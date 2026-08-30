//! OOM-signature and output-quality classification.

use std::sync::OnceLock;

use regex::Regex;

/// The llama.cpp failure patterns that classify a trial as OOM/unviable
/// (matched case-insensitively against server/bench output). Pinned — every
/// entry earned its place from a real failure mode.
pub const FAILURE_PATTERNS: &[&str] = &[
    "cuda error: out of memory",
    "cudaerror_outofmemory",
    "failed to allocate",
    "ggml_cuda_host_malloc",
    "vulkan.*out of memory",
    "cublasstatus_alloc_failed",
    "cudamalloc.*failed",
    "unable to allocate",
    "cannot meet free memory target",
    "failed to fit params to free device memory",
    "failed to load mtp head",
    "using device .* - 0 mib free",
    "failed to lock",
    "mlockall failed",
];

fn failure_regexes() -> &'static Vec<Regex> {
    static REGEXES: OnceLock<Vec<Regex>> = OnceLock::new();
    REGEXES.get_or_init(|| {
        FAILURE_PATTERNS
            .iter()
            .filter_map(|p| Regex::new(p).ok())
            .collect()
    })
}

/// Whether the text carries an OOM/allocation-failure signature.
#[must_use]
pub fn is_oom_message(text: &str) -> bool {
    if text.trim().is_empty() {
        return false;
    }
    let lower = text.to_lowercase();
    failure_regexes().iter().any(|re| re.is_match(&lower))
}

/// Light output sanity check: minimum length plus 4-gram repetition detection.
/// Catches empty responses, looping/stuck models, and crash outputs. The
/// minimums are catalog-tunable for models that legitimately answer short.
#[must_use]
pub fn output_quality_ok(text: &str, min_chars: usize, min_words: usize) -> bool {
    if text.trim().is_empty() || text.len() < min_chars {
        return false;
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < min_words {
        return false;
    }
    let mut fourgrams: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut max_count = 0usize;
    for window in words.windows(4) {
        let gram = window.join(" ");
        let count = fourgrams.entry(gram).or_insert(0);
        *count += 1;
        max_count = max_count.max(*count);
    }
    let total_grams = words.len().saturating_sub(3);
    !(total_grams > 0 && (max_count as f64 / total_grams as f64) > 0.5)
}

/// The default output-quality minimums.
pub const QUALITY_MIN_CHARS: usize = 80;
/// The default output-quality word minimum.
pub const QUALITY_MIN_WORDS: usize = 20;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_the_pinned_oom_signatures() {
        let cases = [
            "ggml_cuda_host_malloc: failed to allocate 1024.00 MiB",
            "CUDA error: out of memory",
            "llama_model_load: unable to allocate backend buffer",
            "vulkan: device ran OUT OF MEMORY during init",
            "cannot meet free memory target of 512 MiB",
            "failed to fit params to free device memory",
            "failed to load MTP head from draft model",
            "using device CUDA0 (NVIDIA GeForce RTX 4090) - 0 MiB free",
            "warning: failed to lock 21474836480-byte buffer",
            "mlockall failed: Cannot allocate memory",
        ];
        for case in cases {
            assert!(is_oom_message(case), "must classify: {case}");
        }
    }

    #[test]
    fn healthy_output_has_no_oom_signature() {
        assert!(!is_oom_message(""));
        assert!(!is_oom_message("   "));
        assert!(!is_oom_message(
            "slot update_slots: prompt processing progress, n_tokens += 1024"
        ));
        assert!(!is_oom_message(
            "main: server is listening on 127.0.0.1:8080"
        ));
    }

    #[test]
    fn quality_rejects_empty_short_and_looping_output() {
        assert!(!output_quality_ok("", QUALITY_MIN_CHARS, QUALITY_MIN_WORDS));
        assert!(!output_quality_ok(
            "ok fine",
            QUALITY_MIN_CHARS,
            QUALITY_MIN_WORDS
        ));
        // A stuck model repeating one 4-gram past half of all grams.
        let looping = "loop ".repeat(120);
        assert!(!output_quality_ok(
            &looping,
            QUALITY_MIN_CHARS,
            QUALITY_MIN_WORDS
        ));
        // Normal prose passes.
        let healthy = "The configuration ran the full soak without incident; decode \
            throughput stayed level across every checkpoint, VRAM headroom never \
            dropped below the configured floor, and no allocation warnings appeared \
            in the server log during the measured window.";
        assert!(output_quality_ok(
            healthy,
            QUALITY_MIN_CHARS,
            QUALITY_MIN_WORDS
        ));
    }

    #[test]
    fn quality_minimums_are_tunable() {
        let short = "Fits fine on this card.";
        assert!(!output_quality_ok(
            short,
            QUALITY_MIN_CHARS,
            QUALITY_MIN_WORDS
        ));
        assert!(output_quality_ok(short, 10, 3));
    }
}
