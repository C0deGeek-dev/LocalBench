//! Override maps: copy/overlay merging and the canonical candidate signature.

use localbench_scoring::score::Overrides;

/// The signature's priority axes, rendered first in this order; any other keys
/// follow sorted, so the same config always yields the same signature.
const SIGNATURE_PRIORITY: &[&str] = &[
    "KvK",
    "KvV",
    "NGpuLayers",
    "NCpuMoe",
    "UbatchSize",
    "BatchSize",
    "Threads",
    "ThreadsBatch",
    "FlashAttn",
    "Mlock",
    "NoMmap",
    "SplitMode",
    "SwaFull",
    "CachePrompt",
    "CacheReuse",
    "SpecType",
    "SpecDraftNMax",
];

/// The canonical signature of a config: `key=value` pairs joined with `;`,
/// priority axes first then the rest sorted, null values skipped, booleans
/// lowercased. The signature is the beam's dedup key — an arm/candidate is its
/// serialized config, never a label.
#[must_use]
pub fn candidate_signature(overrides: &Overrides) -> String {
    let mut keys: Vec<&str> = Vec::new();
    for key in SIGNATURE_PRIORITY {
        if overrides.get(*key).is_some_and(|v| !v.is_null()) {
            keys.push(key);
        }
    }
    // BTreeMap iterates sorted, matching the sorted-extras rule.
    for (key, value) in overrides {
        if !SIGNATURE_PRIORITY.contains(&key.as_str()) && !value.is_null() {
            keys.push(key);
        }
    }

    let parts: Vec<String> = keys
        .iter()
        .map(|key| format!("{key}={}", render_value(&overrides[*key])))
        .collect();
    parts.join(";")
}

/// Render an override value for the signature: bare strings, lowercase
/// booleans, numbers as written.
fn render_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

/// Merge an overlay onto a base config: overlay keys win, base keys survive.
#[must_use]
pub fn join_overrides(base: &Overrides, overlay: &Overrides) -> Overrides {
    let mut merged = base.clone();
    for (key, value) in overlay {
        merged.insert(key.clone(), value.clone());
    }
    merged
}

/// Build an [`Overrides`] map from key/value pairs (test/construction sugar).
#[must_use]
pub fn overrides_of(pairs: &[(&str, serde_json::Value)]) -> Overrides {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_orders_priority_axes_first_then_sorted_extras() {
        let ov = overrides_of(&[
            ("Zeta", 1.into()),
            ("UbatchSize", 1024.into()),
            ("NCpuMoe", 20.into()),
            ("Alpha", "x".into()),
        ]);
        assert_eq!(
            candidate_signature(&ov),
            "NCpuMoe=20;UbatchSize=1024;Alpha=x;Zeta=1"
        );
    }

    #[test]
    fn signature_lowercases_booleans_and_skips_nulls() {
        let ov = overrides_of(&[
            ("Mlock", true.into()),
            ("NoMmap", false.into()),
            ("KvK", serde_json::Value::Null),
        ]);
        assert_eq!(candidate_signature(&ov), "Mlock=true;NoMmap=false");
    }

    #[test]
    fn join_overlay_wins_and_base_survives() {
        let base = overrides_of(&[("NCpuMoe", 20.into()), ("KvK", "q8_0".into())]);
        let overlay = overrides_of(&[("NCpuMoe", 25.into()), ("KvV", "turbo3".into())]);
        let merged = join_overrides(&base, &overlay);
        assert_eq!(merged["NCpuMoe"], serde_json::json!(25));
        assert_eq!(merged["KvK"], serde_json::json!("q8_0"));
        assert_eq!(merged["KvV"], serde_json::json!("turbo3"));
    }
}
