//! The network-isolated container grade plan and the Docker-wedge guard.
//!
//! The grade runs in a container with **no network** so a solve can neither
//! fetch its way to green nor reach the model. Integrity of the solved tree is
//! kept by mounting it **read-only** at `/src`; the grade then copies it into a
//! **writable** `/work` and builds/tests there — a compiled-language test
//! command (rust/go/java/C++) must be able to write build artifacts, which the
//! old read-only-`/work` mount blocked, silently grading every compiled cell
//! "0 tests ran → not solved". The command runs under `bash -c` (not `-lc`,
//! which re-sources `/etc/profile` and drops the go/rust toolchain from PATH).
//! The plan is built for inspection and executed by the app layer; the wedge
//! guard keeps a wedged Docker engine from silently burning hours of timeouts.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static CONTAINER_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// One external task to grade.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradeTask {
    pub id: String,
    /// The materialized task workspace on the host.
    pub workspace: String,
    /// The benchmark's own test command, run inside the container.
    pub test_command: String,
}

/// The container invocation the grader would run, returned for inspection
/// without executing it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraderPlan {
    pub runtime: String,
    pub image: String,
    /// Collision-resistant exact identity used for targeted compensation.
    pub container_name: String,
    pub command: Vec<String>,
}

/// Build the network-enabled preparation plan that fills the shared Cargo
/// registry before a Rust grade starts. The generated warm manifest lives
/// inside the host registry at `.localbench-warm/Cargo.toml`; mounting the
/// registry read-write lets `cargo fetch` populate it, while the subsequent
/// grade mounts the same path read-only and disables the network.
#[must_use]
pub fn cargo_warm_plan(task: &GradeTask, image: Option<&str>, cargo_cache: &str) -> GraderPlan {
    let image = image.map_or_else(|| format!("swebench/task:{}", task.id), str::to_string);
    let container_name = container_name("warm", &task.id);
    let command = vec![
        "docker".to_string(),
        "run".to_string(),
        "--rm".to_string(),
        "--name".to_string(),
        container_name.clone(),
        "-e".to_string(),
        "CARGO_HOME=/usr/local/cargo".to_string(),
        "-v".to_string(),
        format!("{cargo_cache}:/usr/local/cargo/registry"),
        image.clone(),
        "cargo".to_string(),
        "fetch".to_string(),
        "--manifest-path".to_string(),
        "/usr/local/cargo/registry/.localbench-warm/Cargo.toml".to_string(),
    ];
    GraderPlan {
        runtime: "docker".to_string(),
        image,
        container_name,
        command,
    }
}

/// Build the grade plan for a task: `docker run --rm --network=none` with the
/// solved tree mounted read-only at `/src`, copied into a writable `/work`, and
/// the test command run there. Image defaults to the per-task SWE-bench image
/// convention. When `cargo_cache` is `Some`, a warmed cargo registry is mounted
/// read-only and cargo is put in offline mode — required for a `--network=none`
/// Rust grade (see [`crate::grade`]); other languages pass `None`.
#[must_use]
pub fn container_grade_plan(
    task: &GradeTask,
    image: Option<&str>,
    cargo_cache: Option<&str>,
) -> GraderPlan {
    let image = image.map_or_else(|| format!("swebench/task:{}", task.id), str::to_string);
    let container_name = container_name("grade", &task.id);
    let mut command: Vec<String> = vec![
        "docker".to_string(),
        "run".to_string(),
        "--rm".to_string(),
        "--name".to_string(),
        container_name.clone(),
        "--network=none".to_string(),
        // Python: don't litter the copied tree with .pyc (matches the recipe).
        "-e".to_string(),
        "PYTHONDONTWRITEBYTECODE=1".to_string(),
        // Solved tree read-only — the grade can't mutate it to game the result.
        "-v".to_string(),
        format!("{}:/src:ro", task.workspace),
    ];
    if let Some(cache) = cargo_cache {
        command.push("-v".to_string());
        command.push(format!("{cache}:/usr/local/cargo/registry:ro"));
        command.push("-e".to_string());
        command.push("CARGO_HOME=/usr/local/cargo".to_string());
        command.push("-e".to_string());
        command.push("CARGO_NET_OFFLINE=true".to_string());
    }
    command.push("-w".to_string());
    command.push("/work".to_string());
    command.push(image.clone());
    command.push("bash".to_string());
    command.push("-c".to_string());
    // Copy the read-only source into the writable workdir, then grade it there.
    command.push(format!("cp -a /src/. /work/ && {}", task.test_command));
    GraderPlan {
        runtime: "docker".to_string(),
        image,
        container_name,
        command,
    }
}

/// The pre-flight health-check invocation: prove the engine actually RUNS a
/// container (bounded by the caller's timeout) before trusting it — `docker
/// info` can answer while `docker run` hangs.
#[must_use]
pub fn docker_healthcheck_plan() -> GraderPlan {
    let image = "busybox".to_string();
    let container_name = container_name("health", "engine");
    GraderPlan {
        runtime: "docker".to_string(),
        image: image.clone(),
        command: vec![
            "docker".to_string(),
            "run".to_string(),
            "--rm".to_string(),
            "--name".to_string(),
            container_name.clone(),
            image,
            "true".to_string(),
        ],
        container_name,
    }
}

/// Exact forced-removal compensation for one inspectable plan identity.
#[must_use]
pub fn docker_cleanup_command(container_name: &str) -> Vec<String> {
    ["docker", "rm", "-f", container_name]
        .iter()
        .map(|part| (*part).to_string())
        .collect()
}

fn container_name(phase: &str, task_id: &str) -> String {
    let slug = task_slug(task_id);
    let sequence = CONTAINER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let epoch_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    format!(
        "localbench-{phase}-{slug}-{}-{epoch_nanos}-{sequence}",
        std::process::id(),
    )
}

fn task_slug(task_id: &str) -> String {
    let mut slug = String::with_capacity(32);
    let mut last_dash = false;
    for character in task_id.chars() {
        let normalized = if character.is_ascii_alphanumeric() {
            character.to_ascii_lowercase()
        } else {
            '-'
        };
        if normalized == '-' && (last_dash || slug.is_empty()) {
            continue;
        }
        slug.push(normalized);
        last_dash = normalized == '-';
        if slug.len() >= 32 {
            break;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        "task".to_string()
    } else {
        slug
    }
}

/// The consecutive-grade-timeout circuit breaker. The Docker engine can wedge
/// (Docker-for-Windows / WSL2): every `docker run` then hangs, so every grade
/// times out and a sweep silently marks cells failed for hours. Three
/// consecutive timeouts trip the breaker — the run yields with its ledger
/// intact instead of wasting itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DockerWedgeBreaker {
    strikes: u32,
    threshold: u32,
}

impl Default for DockerWedgeBreaker {
    fn default() -> Self {
        Self::new(3)
    }
}

impl DockerWedgeBreaker {
    /// A breaker that trips after `threshold` consecutive grade timeouts.
    #[must_use]
    pub fn new(threshold: u32) -> Self {
        Self {
            strikes: 0,
            threshold: threshold.max(1),
        }
    }

    /// Record one grade outcome; any non-timeout resets the streak. Returns
    /// whether the breaker is now tripped.
    pub fn record(&mut self, grade_timed_out: bool) -> bool {
        if grade_timed_out {
            self.strikes += 1;
        } else {
            self.strikes = 0;
        }
        self.tripped()
    }

    /// Whether the engine looks wedged (the streak reached the threshold).
    #[must_use]
    pub fn tripped(&self) -> bool {
        self.strikes >= self.threshold
    }

    /// The current consecutive-timeout streak.
    #[must_use]
    pub fn strikes(&self) -> u32 {
        self.strikes
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn task() -> GradeTask {
        GradeTask {
            id: "astropy__astropy-12907".to_string(),
            workspace: "C:/bench/work/astropy-12907".to_string(),
            test_command: "python -m pytest -q".to_string(),
        }
    }

    #[test]
    fn grade_plan_is_network_isolated_and_grades_a_writable_copy() {
        let plan = container_grade_plan(&task(), None, None);
        assert_eq!(plan.runtime, "docker");
        assert_eq!(plan.image, "swebench/task:astropy__astropy-12907");
        assert!(plan.command.contains(&"--network=none".to_string()));
        // Solved tree mounted read-only at /src (integrity), copied into a
        // writable /work (so compiled languages can build), graded there.
        assert!(plan
            .command
            .contains(&"C:/bench/work/astropy-12907:/src:ro".to_string()));
        assert!(plan.command.windows(2).any(|w| w == ["-w", "/work"]));
        // `bash -c` (not `-lc`, which would drop the toolchain PATH).
        assert!(plan.command.windows(2).any(|w| w == ["bash", "-c"]));
        assert_eq!(
            plan.command.last().unwrap(),
            "cp -a /src/. /work/ && python -m pytest -q"
        );
        // No read-only /work mount that would block the build.
        assert!(!plan.command.iter().any(|a| a.contains(":/work:ro")));
        // An explicit image wins.
        let custom = container_grade_plan(&task(), Some("bench/custom:1"), None);
        assert_eq!(custom.image, "bench/custom:1");
    }

    #[test]
    fn rust_grade_mounts_the_cargo_cache_and_goes_offline() {
        let plan = container_grade_plan(&task(), None, Some("C:/bench/cargo-cache"));
        assert!(plan
            .command
            .contains(&"C:/bench/cargo-cache:/usr/local/cargo/registry:ro".to_string()));
        assert!(plan.command.contains(&"CARGO_NET_OFFLINE=true".to_string()));
        assert!(plan
            .command
            .contains(&"CARGO_HOME=/usr/local/cargo".to_string()));
    }

    #[test]
    fn rust_cache_warm_is_network_enabled_and_writes_the_shared_registry() {
        let plan = cargo_warm_plan(&task(), None, "C:/bench/cargo-cache");
        assert!(!plan.command.contains(&"--network=none".to_string()));
        assert!(plan
            .command
            .contains(&"C:/bench/cargo-cache:/usr/local/cargo/registry".to_string()));
        assert!(plan.command.windows(2).any(|w| w == ["cargo", "fetch"]));
        assert!(plan
            .command
            .contains(&"/usr/local/cargo/registry/.localbench-warm/Cargo.toml".to_string()));
        assert!(plan
            .container_name
            .starts_with("localbench-warm-astropy-astropy-12907-"));
        assert!(plan
            .command
            .windows(2)
            .any(|args| { args == ["--name".to_string(), plan.container_name.clone()] }));
    }

    #[test]
    fn healthcheck_actually_runs_a_container() {
        let plan = docker_healthcheck_plan();
        assert_eq!(plan.command[..2], ["docker".to_string(), "run".to_string()]);
        assert!(plan.command.contains(&"busybox".to_string()));
        assert_eq!(plan.command.last().unwrap(), "true");
        assert!(plan.container_name.starts_with("localbench-health-engine-"));
        assert_eq!(
            docker_cleanup_command(&plan.container_name),
            ["docker", "rm", "-f", plan.container_name.as_str()]
        );
    }

    #[test]
    fn every_phase_gets_a_valid_unique_exact_name() {
        let warm = cargo_warm_plan(&task(), None, "C:/bench/cargo-cache");
        let grade_one = container_grade_plan(&task(), None, None);
        let grade_two = container_grade_plan(&task(), None, None);
        for plan in [&warm, &grade_one, &grade_two] {
            assert!(plan.container_name.len() <= 128);
            assert!(plan.container_name.chars().all(|character| {
                character.is_ascii_lowercase()
                    || character.is_ascii_digit()
                    || matches!(character, '-' | '_' | '.')
            }));
            assert_eq!(
                plan.command
                    .windows(2)
                    .find(|args| args[0] == "--name")
                    .map(|args| args[1].as_str()),
                Some(plan.container_name.as_str())
            );
        }
        assert_ne!(grade_one.container_name, grade_two.container_name);
    }

    #[test]
    fn three_consecutive_timeouts_trip_the_breaker() {
        let mut breaker = DockerWedgeBreaker::default();
        assert!(!breaker.record(true));
        assert!(!breaker.record(true));
        assert!(breaker.record(true), "third consecutive timeout trips");
        assert!(breaker.tripped());
    }

    #[test]
    fn a_successful_grade_resets_the_streak() {
        let mut breaker = DockerWedgeBreaker::default();
        breaker.record(true);
        breaker.record(true);
        assert!(!breaker.record(false), "success resets");
        assert_eq!(breaker.strikes(), 0);
        assert!(!breaker.record(true));
        assert!(!breaker.record(true));
        assert!(breaker.record(true));
    }
}
