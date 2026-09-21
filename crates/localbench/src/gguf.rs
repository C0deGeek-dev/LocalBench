//! A minimal GGUF metadata reader — just enough to tell a dense model from an
//! MoE one and to read its layer count.
//!
//! The tuner's search space turns entirely on two facts the GGUF header already
//! carries: whether the model has expert tensors (`<arch>.expert_count` > 0 ⇒
//! MoE, absent or 0 ⇒ dense) and how many layers it has
//! (`<arch>.block_count`). Reading them is what lets a dense model be tuned on
//! the layer-offload axis instead of the no-op `--n-cpu-moe` one. Only the
//! key/value metadata block is read for that, so it opens the file, reads a
//! few kilobytes, and stops.
//!
//! One more fact comes from the tensor directory that follows the metadata:
//! the size of a per-layer embedding table, which a build with `--lazy-mode`
//! reads from disk on demand instead of loading, so it never becomes private
//! memory even when the model is loaded without mmap.
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

/// The tensor a build with `--lazy-mode` reads on demand: the per-layer token
/// embedding table (Gemma 4, Qwen3.8 "qwen4exp").
pub const LAZY_TABLE: &str = "per_layer_token_embd.weight";

/// `--lazy-mode auto` reads that table lazily only when it is larger than this.
pub const LAZY_AUTO_MIN_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// The size in bytes of this file's per-layer embedding table, when the file
/// holds one. A split model keeps it in one shard; the others answer `None`.
/// Never errors: an unreadable or malformed file yields `None`.
#[must_use]
pub fn lazy_table_bytes(path: &Path) -> Option<u64> {
    let file = File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    tensor_bytes(
        &mut Counting {
            inner: BufReader::new(file),
            read: 0,
        },
        file_len,
        LAZY_TABLE,
    )
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

/// The metadata block of a GGUF file: its tensor count, architecture, and
/// every unsigned-integer value by key.
struct Header {
    tensor_count: u64,
    arch: Option<String>,
    uints: HashMap<String, i64>,
}

fn read_header<R: Read>(reader: &mut R) -> Option<Header> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let _version = read_u32(reader)?;
    let tensor_count = read_u64(reader)?;
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
    Some(Header {
        tensor_count,
        arch,
        uints,
    })
}

fn parse<R: Read>(reader: &mut R) -> Option<ModelShape> {
    let Header { arch, uints, .. } = read_header(reader)?;

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

/// A reader that counts the bytes read, so the tensor data section's start
/// (which follows the directory, aligned) is known without seeking.
struct Counting<R> {
    inner: R,
    read: u64,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read += n as u64;
        Ok(n)
    }
}

/// The size of the named tensor's data. The directory gives each tensor's
/// offset into the data section, which follows the directory at the file's
/// alignment; a tensor's data runs to the next offset, or to the end of the
/// file for the last one — so no type-size table is needed.
fn tensor_bytes<R: Read>(reader: &mut Counting<R>, file_len: u64, name: &str) -> Option<u64> {
    let header = read_header(reader)?;
    let alignment = header
        .uints
        .get("general.alignment")
        .and_then(|a| u64::try_from(*a).ok())
        .filter(|a| *a > 0)
        .unwrap_or(32);
    // Real models have hundreds to a few thousand tensors.
    let tensor_count = header.tensor_count.min(1_000_000);
    let mut offsets: Vec<u64> = Vec::new();
    let mut target: Option<u64> = None;
    for _ in 0..tensor_count {
        let tensor = read_string(reader)?;
        let dims = read_u32(reader)?;
        if dims > 8 {
            return None;
        }
        for _ in 0..dims {
            read_u64(reader)?;
        }
        let _type = read_u32(reader)?;
        let offset = read_u64(reader)?;
        if tensor == name {
            target = Some(offset);
        }
        offsets.push(offset);
    }
    let target = target?;
    let data_start = reader.read.div_ceil(alignment) * alignment;
    let data_len = file_len.checked_sub(data_start)?;
    let end = offsets
        .iter()
        .copied()
        .filter(|offset| *offset > target)
        .min()
        .unwrap_or(data_len);
    end.checked_sub(target)
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

    /// A GGUF stream with a tensor directory: `(name, offset)` entries and
    /// `data_len` bytes of tensor data after the aligned directory.
    fn gguf_with_tensors(
        tensors: &[(&str, u64)],
        data_len: u64,
        alignment: Option<u32>,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        let kv = 1 + u64::from(alignment.is_some());
        out.extend_from_slice(&kv.to_le_bytes());
        let put_string = |out: &mut Vec<u8>, s: &str| {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        };
        put_string(&mut out, "general.architecture");
        out.extend_from_slice(&ty::STRING.to_le_bytes());
        put_string(&mut out, "qwen4exp");
        if let Some(a) = alignment {
            put_string(&mut out, "general.alignment");
            out.extend_from_slice(&ty::UINT32.to_le_bytes());
            out.extend_from_slice(&a.to_le_bytes());
        }
        for (name, offset) in tensors {
            put_string(&mut out, name);
            out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&256u64.to_le_bytes());
            out.extend_from_slice(&1024u64.to_le_bytes());
            out.extend_from_slice(&12u32.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }
        let align = u64::from(alignment.unwrap_or(32));
        while out.len() as u64 % align != 0 {
            out.push(0);
        }
        out.resize(out.len() + usize::try_from(data_len).unwrap(), 0);
        out
    }

    fn size_of(bytes: Vec<u8>, name: &str) -> Option<u64> {
        let len = bytes.len() as u64;
        tensor_bytes(
            &mut Counting {
                inner: Cursor::new(bytes),
                read: 0,
            },
            len,
            name,
        )
    }

    #[test]
    fn a_tensor_runs_to_the_next_offset_or_to_the_end_of_the_file() {
        let bytes = gguf_with_tensors(
            &[
                ("token_embd.weight", 0),
                (LAZY_TABLE, 4096),
                ("output.weight", 4096 + 7000),
            ],
            4096 + 7000 + 1200,
            None,
        );
        assert_eq!(size_of(bytes.clone(), LAZY_TABLE), Some(7000));
        assert_eq!(size_of(bytes.clone(), "output.weight"), Some(1200));
        assert_eq!(size_of(bytes, "token_embd.weight"), Some(4096));
    }

    #[test]
    fn the_data_section_starts_at_the_files_own_alignment() {
        let bytes = gguf_with_tensors(&[(LAZY_TABLE, 0)], 5000, Some(64));
        assert_eq!(size_of(bytes, LAZY_TABLE), Some(5000));
    }

    #[test]
    fn a_shard_without_the_table_answers_none() {
        let bytes = gguf_with_tensors(&[("blk.0.ffn_up_exps.weight", 0)], 100, None);
        assert_eq!(size_of(bytes, LAZY_TABLE), None);
        let truncated = gguf_with_tensors(&[(LAZY_TABLE, 0)], 100, None);
        assert_eq!(size_of(truncated[..40].to_vec(), LAZY_TABLE), None);
    }

    /// Live check against a real model file: set `LOCALBENCH_LAZY_GGUF` to the
    /// shard that holds a per-layer embedding table and `LOCALBENCH_LAZY_BYTES`
    /// to its expected size.
    #[test]
    #[ignore = "needs a real GGUF with a per-layer embedding table"]
    fn reads_the_table_size_of_a_real_model() {
        let path = std::env::var("LOCALBENCH_LAZY_GGUF").expect("LOCALBENCH_LAZY_GGUF");
        let bytes = lazy_table_bytes(Path::new(&path)).expect("the table");
        println!("{LAZY_TABLE}: {bytes} bytes");
        if let Ok(expected) = std::env::var("LOCALBENCH_LAZY_BYTES") {
            assert_eq!(bytes, expected.parse::<u64>().unwrap());
        }
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
