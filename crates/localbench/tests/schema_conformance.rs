//! Wire-format conformance: the JSON the binary actually emits must validate
//! against the committed `schemas/*-v1.schema.json`, and the shipped samples
//! must be renderable emitter output. Ground truth is the emitter itself —
//! reports are built here through the same constructors the commands use, so
//! any drift between the serde structs and the declared contract reds CI
//! instead of aging silently (the previous schemas described the retired
//! PowerShell emitter for a full stack-rewrite).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use localbench::upliftrun::{UpliftArmRow, UpliftReport};
use localbench_measure::runner::{ArmReportRow, CapabilityReport};
use localbench_scoring::uplift::{
    aggregate, assert_injection, significance, ArmResult, MemoryUsed, TaskResult,
};
use localx_llama_core::tuner::{Profile, PromptLength, SearchStrategy};
use localx_llama_core::{
    Mode, Overrides, TunerBestConfig, TunerEntry, CURRENT_TUNER_VERSION, TUNER_SCHEMA,
};

fn used(id: &str) -> Vec<MemoryUsed> {
    vec![MemoryUsed { id: id.to_string() }]
}

fn representative_uplift_report() -> UpliftReport {
    // Two arms built through the real aggregation/injection/significance
    // pipeline — the same calls `run_uplift` makes.
    let baseline = ArmResult {
        arm: "baseline".to_string(),
        is_lesson_arm: false,
        trials: 3,
        tasks: vec![
            TaskResult {
                passes: vec![true, false, true],
                memories_used: vec![vec![], vec![], vec![]],
            },
            TaskResult {
                passes: vec![false, false, true],
                memories_used: vec![vec![], vec![], vec![]],
            },
        ],
    };
    let lesson = ArmResult {
        arm: "lessons".to_string(),
        is_lesson_arm: true,
        trials: 3,
        tasks: vec![
            TaskResult {
                passes: vec![true, true, true],
                memories_used: vec![
                    used("mem-guard-rails"),
                    used("mem-guard-rails"),
                    used("mem-guard-rails"),
                ],
            },
            TaskResult {
                passes: vec![true, false, true],
                memories_used: vec![
                    used("mem-guard-rails"),
                    used("mem-guard-rails"),
                    used("mem-guard-rails"),
                ],
            },
        ],
    };
    let intended = vec!["mem-guard-rails".to_string()];
    let baseline_agg = aggregate(&baseline).unwrap();
    let lesson_agg = aggregate(&lesson).unwrap();
    let uplift = significance(&baseline_agg, &lesson_agg, 0.05);
    UpliftReport {
        schema: 1,
        task_set: "headroom-mini".to_string(),
        model: "example-local-model".to_string(),
        trials: 3,
        arms: vec![
            UpliftArmRow {
                aggregate: baseline_agg,
                injection: assert_injection(&baseline, &[]).unwrap(),
            },
            UpliftArmRow {
                aggregate: lesson_agg,
                injection: assert_injection(&lesson, &intended).unwrap(),
            },
        ],
        uplift,
    }
}

fn representative_capability_report() -> CapabilityReport {
    CapabilityReport {
        schema: 1,
        corpus: "external".to_string(),
        contamination_suspect: true,
        arms: vec![
            ArmReportRow {
                arm: "baseline".to_string(),
                model: "example-local-model".to_string(),
                tasks: 25,
                solved: 14,
                solve_rate: 0.56,
                avg_tool_calls: 18.2,
                avg_redundant: 1.4,
                avg_diff_added: 42.0,
                avg_interventions: 0.0,
                judge_overall: Some(0.61),
                isolation: Some("clean".to_string()),
            },
            ArmReportRow {
                arm: "coached".to_string(),
                model: "example-local-model".to_string(),
                tasks: 25,
                solved: 17,
                solve_rate: 0.68,
                avg_tool_calls: 16.9,
                avg_redundant: 1.1,
                avg_diff_added: 39.5,
                avg_interventions: 1.8,
                judge_overall: Some(0.66),
                isolation: Some("clean".to_string()),
            },
        ],
    }
}

fn compiled_schema(name: &str) -> jsonschema::Validator {
    let raw = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../schemas")
            .join(name),
    )
    .unwrap();
    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    jsonschema::validator_for(&doc).expect("schema must compile")
}

fn assert_valid(schema: &jsonschema::Validator, value: &serde_json::Value, what: &str) {
    let detail: Vec<String> = schema
        .iter_errors(value)
        .map(|e| format!("{} @ {}", e, e.instance_path))
        .collect();
    assert!(
        detail.is_empty(),
        "{what} does not conform: {}",
        detail.join("; ")
    );
}

#[test]
fn the_uplift_emitter_conforms_to_the_declared_schema() {
    let schema = compiled_schema("localbench-uplift-v1.schema.json");
    let report = representative_uplift_report();
    let value = serde_json::to_value(&report).unwrap();
    assert_valid(&schema, &value, "emitted uplift report");
}

#[test]
fn the_capability_emitter_conforms_to_the_declared_schema() {
    let schema = compiled_schema("localbench-capability-v1.schema.json");
    let report = representative_capability_report();
    let value = serde_json::to_value(&report).unwrap();
    assert_valid(&schema, &value, "emitted capability report");
}

#[test]
fn a_prism_autobest_entry_conforms_to_both_store_contracts() {
    let store = TunerBestConfig {
        schema: TUNER_SCHEMA,
        key: "tbonsai27b".to_string(),
        vram_gb: Some(24),
        entries: vec![TunerEntry {
            quant: "q2-0".to_string(),
            context_key: "64k".to_string(),
            context_tokens: Some(65_536),
            mode: Mode::PrismMl,
            vram_gb: 24,
            prompt_length: PromptLength::Short,
            profile: Profile::Pure,
            search_strategy: Some(SearchStrategy::Beam),
            beam_width: Some(3),
            score: 42.0,
            score_unit: "tps".to_string(),
            pure_score: Some(42.0),
            args: Vec::new(),
            overrides: Overrides::default(),
            measured_at: "2026-07-17T00:00:00Z".to_string(),
            tuner_version: CURRENT_TUNER_VERSION,
            trial_count: Some(1),
            gpu_names: None,
            llamacpp_build: Some("prism-b9591-62061f9".to_string()),
        }],
    };
    let value = serde_json::to_value(store).unwrap();
    assert_eq!(value["entries"][0]["mode"], "prismml");
    assert_eq!(value["entries"][0]["searchStrategy"], "beam");
    assert_eq!(value["entries"][0]["beamWidth"], 3);
    assert_valid(
        &compiled_schema("tuner-best-config.schema.json"),
        &value,
        "Prism tuner store",
    );
    assert_valid(
        &compiled_schema("localbox-autobest-v1.schema.json"),
        &value,
        "Prism AutoBest profile",
    );
}

#[test]
fn the_shipped_samples_conform_and_parse_as_emitter_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/localbench");

    let uplift_raw = std::fs::read_to_string(root.join("uplift/sample-uplift.json")).unwrap();
    let parsed: UpliftReport = serde_json::from_str(&uplift_raw)
        .expect("the shipped uplift sample must parse as the emitter's own type");
    assert_valid(
        &compiled_schema("localbench-uplift-v1.schema.json"),
        &serde_json::to_value(&parsed).unwrap(),
        "shipped uplift sample",
    );

    let capability_raw =
        std::fs::read_to_string(root.join("capability/sample-capability-report.json")).unwrap();
    let parsed: CapabilityReport = serde_json::from_str(&capability_raw)
        .expect("the shipped capability sample must parse as the emitter's own type");
    assert_valid(
        &compiled_schema("localbench-capability-v1.schema.json"),
        &serde_json::to_value(&parsed).unwrap(),
        "shipped capability sample",
    );
}

/// Regenerate helper (ignored): prints emitter-built sample JSON so the
/// committed samples can be refreshed from real output when the wire format
/// deliberately changes. Run with:
/// `cargo test -p localbench --test schema_conformance -- --ignored --nocapture`
#[test]
#[ignore = "manual sample regeneration helper, not a check"]
fn print_fresh_samples() {
    println!(
        "--- uplift ---\n{}",
        serde_json::to_string_pretty(&representative_uplift_report()).unwrap()
    );
    println!(
        "--- capability ---\n{}",
        serde_json::to_string_pretty(&representative_capability_report()).unwrap()
    );
}
