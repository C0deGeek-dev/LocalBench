//! A minimal GGUF metadata reader — just enough to tell a dense model from an
//! MoE one and to read its layer count.
//!
//! The tuner's search space turns entirely on two facts the GGUF header already
//! carries: whether the model has expert tensors (`<arch>.expert_count` > 0 ⇒
//! MoE, absent or 0 ⇒ dense) and how many layers it has
//! (`<arch>.block_count`). Reading them is what lets a dense model be tuned on
//! the layer-offload axis instead of the no-op `--n-cpu-moe` one. Only the
//! key/value metadata block is read — the tensor directory that follows it is
//! never touched — so this opens the file, reads a few kilobytes, and stops.
//!
//! The reader is deliberately total: any malformed or unexpected input yields
//! [`ModelShape::UNKNOWN`] (expert count `-1`) so the caller falls back to the
//! catalog heuristic rather than failing the run.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// What the tuner needs from a GGUF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// The model's expert count: `>= 0` when read from the file (`0` = dense,
    /// `> 0` = MoE), `-1` when unknown (unreadable header → fall back to the
    /// catalog heuristic).
    pub expert_count: i64,
    /// The model's transformer layer count (`<arch>.block_count`), or `0` when
    /// unknown.
    pub block_count: i64,
}

impl ModelShape {
    /// The "nothing read" shape: the caller falls back to the catalog heuristic.
    pub const UNKNOWN: ModelShape = ModelShape {
        expert_count: -1,
        block_count: 0,
    };
}

/// Read a GGUF file's expert and layer counts. Never errors: an unreadable or
/// malformed file yields [`ModelShape::UNKNOWN`].
#[must_use]
pub fn read_model_shape(path: &Path) -> ModelShape {
    match File::open(path) {
        Ok(file) => parse(&mut BufReader::new(file)).unwrap_or(ModelShape::UNKNOWN),
        Err(_) => ModelShape::UNKNOWN,
    }
}

/// GGUF metadata value types (spec v2/v3).
mod ty {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

/// A metadata value we bothered to decode: either an unsigned integer (the only
/// kind the tuner reads) or a string (for `general.architecture`). Everything
/// else is skipped over without being retained.
enum Value {
    Uint(i64),
    Text(String),
    Other,
}

fn parse<R: Read>(reader: &mut R) -> Option<ModelShape> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let _version = read_u32(reader)?;
    let _tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;

    let mut arch: Option<String> = None;
    let mut uints: HashMap<String, i64> = HashMap::new();

    // A header with an absurd count is corrupt; cap the loop so a bad file
    // cannot spin. Real models have on the order of tens of metadata entries.
    let kv_count = kv_count.min(100_000);
    for _ in 0..kv_count {
        let key = read_string(reader)?;
        let value = read_value(reader)?;
        match value {
            Value::Text(text) if key == "general.architecture" => arch = Some(text),
            Value::Uint(n) => {
                uints.insert(key, n);
            }
            _ => {}
        }
    }

    // A header that never named `general.architecture` did not parse far enough
    // to be trusted: report it as unknown (fall back to the catalog heuristic),
    // not dense. Only once the architecture is known does an *absent*
    // expert_count key authoritatively mean dense (LocalHub#76).
    let arch = arch?;

    // Resolve the arch-prefixed keys; fall back to any key with the right
    // suffix if the architecture name did not prefix them.
    let by_suffix = |suffix: &str| -> Option<i64> {
        if let Some(v) = uints.get(&format!("{arch}.{suffix}")) {
            return Some(*v);
        }
        uints
            .iter()
            .find(|(k, _)| k.ends_with(&format!(".{suffix}")))
            .map(|(_, v)| *v)
    };

    Some(ModelShape {
        // A file that was read far enough to name its architecture but carries no
        // expert count is dense (0), which is authoritative — not `-1` "unknown".
        expert_count: by_suffix("expert_count").unwrap_or(0),
        block_count: by_suffix("block_count").unwrap_or(0),
    })
}

fn read_value<R: Read>(reader: &mut R) -> Option<Value> {
    let value_type = read_u32(reader)?;
    read_typed(reader, value_type)
}

fn read_typed<R: Read>(reader: &mut R, value_type: u32) -> Option<Value> {
    match value_type {
        ty::UINT8 | ty::BOOL => Some(Value::Uint(i64::from(read_n::<1, R>(reader)?[0]))),
        ty::INT8 => Some(Value::Uint(i64::from(read_n::<1, R>(reader)?[0] as i8))),
        ty::UINT16 => Some(Value::Uint(i64::from(u16::from_le_bytes(read_n(reader)?)))),
        ty::INT16 => Some(Value::Uint(i64::from(i16::from_le_bytes(read_n(reader)?)))),
        ty::UINT32 => Some(Value::Uint(i64::from(u32::from_le_bytes(read_n(reader)?)))),
        ty::INT32 => Some(Value::Uint(i64::from(i32::from_le_bytes(read_n(reader)?)))),
        ty::UINT64 => Some(Value::Uint(
            i64::try_from(u64::from_le_bytes(read_n(reader)?)).unwrap_or(i64::MAX),
        )),
        ty::INT64 => Some(Value::Uint(i64::from_le_bytes(read_n(reader)?))),
        ty::FLOAT32 => {
            let _ = read_n::<4, R>(reader)?;
            Some(Value::Other)
        }
        ty::FLOAT64 => {
            let _ = read_n::<8, R>(reader)?;
            Some(Value::Other)
        }
        ty::STRING => Some(Value::Text(read_string(reader)?)),
        ty::ARRAY => {
            let elem_type = read_u32(reader)?;
            let count = read_u64(reader)?.min(100_000_000);
            for _ in 0..count {
                // Read each element to advance the stream; the values are not
                // retained (the tuner reads no array-valued metadata).
                read_typed(reader, elem_type)?;
            }
            Some(Value::Other)
        }
        _ => None,
    }
}

fn read_string<R: Read>(reader: &mut R) -> Option<String> {
    let len = usize::try_from(read_u64(reader)?).ok()?;
    // A single metadata string over 64 MiB is corrupt; refuse rather than try to
    // allocate it.
    if len > 64 * 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn read_n<const N: usize, R: Read>(reader: &mut R) -> Option<[u8; N]> {
    let mut buf = [0u8; N];
    reader.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn read_u32<R: Read>(reader: &mut R) -> Option<u32> {
    Some(u32::from_le_bytes(read_n(reader)?))
}

fn read_u64<R: Read>(reader: &mut R) -> Option<u64> {
    Some(u64::from_le_bytes(read_n(reader)?))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal GGUF byte stream with the given string and uint metadata.
    fn gguf(strings: &[(&str, &str)], uints: &[(&str, u64)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        let kv = (strings.len() + uints.len()) as u64;
        out.extend_from_slice(&kv.to_le_bytes());
        let put_string = |out: &mut Vec<u8>, s: &str| {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        };
        for (k, v) in strings {
            put_string(&mut out, k);
            out.extend_from_slice(&ty::STRING.to_le_bytes());
            put_string(&mut out, v);
        }
        for (k, v) in uints {
            put_string(&mut out, k);
            out.extend_from_slice(&ty::UINT32.to_le_bytes());
            out.extend_from_slice(&(*v as u32).to_le_bytes());
        }
        out
    }

    #[test]
    fn a_dense_model_reads_zero_experts_and_its_layer_count() {
        // No expert_count key → dense (0, authoritative), block_count read.
        let bytes = gguf(
            &[("general.architecture", "qwen35")],
            &[
                ("qwen35.block_count", 65),
                ("qwen35.attention.head_count", 40),
            ],
        );
        let shape = parse(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(shape.expert_count, 0, "no expert_count key means dense");
        assert_eq!(shape.block_count, 65);
    }

    #[test]
    fn an_moe_model_reads_its_expert_count() {
        let bytes = gguf(
            &[("general.architecture", "qwen3moe")],
            &[("qwen3moe.block_count", 48), ("qwen3moe.expert_count", 128)],
        );
        let shape = parse(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(shape.expert_count, 128);
        assert_eq!(shape.block_count, 48);
    }

    #[test]
    fn an_array_valued_entry_is_skipped_without_derailing_later_keys() {
        // A tokenizer token-type array (common, large) sits between the keys we
        // want; the reader must skip it and still find block_count after it.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes()); // 3 kv
        let put_string = |out: &mut Vec<u8>, s: &str| {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        };
        // arch
        put_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&ty::STRING.to_le_bytes());
        put_string(&mut bytes, "llama");
        // an int32 array of 4 elements
        put_string(&mut bytes, "tokenizer.ggml.token_type");
        bytes.extend_from_slice(&ty::ARRAY.to_le_bytes());
        bytes.extend_from_slice(&ty::INT32.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        for v in [1i32, 2, 3, 4] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // block_count after the array
        put_string(&mut bytes, "llama.block_count");
        bytes.extend_from_slice(&ty::UINT32.to_le_bytes());
        bytes.extend_from_slice(&32u32.to_le_bytes());

        let shape = parse(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(shape.block_count, 32);
        assert_eq!(shape.expert_count, 0);
    }

    #[test]
    fn a_non_gguf_file_is_unknown() {
        let shape = parse(&mut Cursor::new(b"not a gguf".to_vec()));
        assert!(shape.is_none());
        assert_eq!(
            read_model_shape(Path::new("nonexistent.gguf")),
            ModelShape::UNKNOWN
        );
    }

    #[test]
    fn a_truncated_header_is_unknown_not_a_panic() {
        // Claims 5 kv but the stream ends after the first key.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&5u64.to_le_bytes());
        bytes.extend_from_slice(&(4u64).to_le_bytes());
        bytes.extend_from_slice(b"arch");
        assert!(parse(&mut Cursor::new(bytes)).is_none());
    }

    #[test]
    fn a_header_that_never_names_an_architecture_is_unknown_not_dense() {
        // A fully-parsed header carrying a block_count but no
        // `general.architecture` is a file this parser does not understand: it
        // must degrade to unknown so the catalog heuristic decides, never be
        // reported as dense (LocalHub#76 guard).
        let bytes = gguf(&[], &[("qwen35.block_count", 65)]);
        assert!(
            parse(&mut Cursor::new(bytes)).is_none(),
            "no general.architecture ⇒ unknown, not dense"
        );
    }
}
