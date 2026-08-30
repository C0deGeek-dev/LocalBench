//! The runner ↔ solver seam: LocalBench does not own how a solver drives a
//! model against a workspace — the [`Solver`] trait names that dependency so a
//! mock can satisfy it and the live adapter drives `localpilot eval` headless.
//!
//! A hung solver must never block a matrix, so every live invocation is
//! wall-clock bounded: on expiry the process is killed and the cell records a
//! failure instead of the sweep hanging.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_millis(25);
const TREE_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);
const HELPER_REAP_TIMEOUT: Duration = Duration::from_secs(1);

/// One solver invocation's identity and knobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveSpec {
    pub model: String,
    pub arm: String,
    /// The per-task id, so every scorecard is labelled per task.
    pub task: String,
    /// The solver edits its own throwaway task workspace, so `bypass` is the
    /// default — untrusted-code isolation is the grader's job (a
    /// network-isolated, read-only container), not the solver's.
    pub permission: String,
    /// Enable the verify-before-done gate for this run (the verify arm).
    pub verify: bool,
    /// Close the run out into review-gated memory (the warm/teaching arm);
    /// off keeps the run clean-room.
    pub learn: bool,
    /// Path to a coach script: drive the run through the solver's MCP serve
    /// surface with a deterministic scripted coach instead of a headless eval.
    pub coach: Option<String>,
    /// The problem statement (the trailing positional argument).
    pub problem: String,
}

impl SolveSpec {
    /// A spec with the default permission posture.
    #[must_use]
    pub fn new(model: &str, arm: &str, task: &str, problem: &str) -> Self {
        Self {
            model: model.to_string(),
            arm: arm.to_string(),
            task: task.to_string(),
            permission: "bypass".to_string(),
            verify: false,
            learn: false,
            coach: None,
            problem: problem.to_string(),
        }
    }
}

/// The `localpilot eval` argument vector for a spec. Pure, so the arg shape
/// is testable without a binary; the problem statement stays one argv entry.
#[must_use]
pub fn eval_args(spec: &SolveSpec) -> Vec<String> {
    let mut argv = vec![
        "eval".to_string(),
        "--model".to_string(),
        spec.model.clone(),
        "--arm".to_string(),
        spec.arm.clone(),
        "--task".to_string(),
        spec.task.clone(),
        "--permission".to_string(),
        spec.permission.clone(),
    ];
    if spec.verify {
        argv.push("--verify".to_string());
    }
    if spec.learn {
        argv.push("--learn".to_string());
    }
    argv.push(spec.problem.clone());
    argv
}

/// A bounded child-process outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedRun {
    /// Whether the process exited zero (false on timeout).
    pub exit_ok: bool,
    pub stdout: String,
    pub stderr: String,
    /// Whether the wall-clock bound expired (the process was killed).
    pub timed_out: bool,
    /// A secondary process-tree cleanup problem. This never changes the
    /// primary exit or timeout outcome.
    pub cleanup_diagnostic: Option<String>,
}

/// Run a command to completion under a wall-clock bound, capturing stdout and
/// stderr in regular temporary files. Descendants may inherit those handles,
/// but unlike pipes they cannot keep a reader blocked waiting for EOF. Each
/// child owns a process group/tree; on expiry that exact tree is terminated.
///
/// # Errors
/// A plain-language message when the process cannot be spawned.
pub fn run_bounded(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    timeout: Duration,
) -> Result<BoundedRun, String> {
    let mut stdout_file = tempfile::tempfile()
        .map_err(|error| format!("could not create stdout capture for {program}: {error}"))?;
    let mut stderr_file = tempfile::tempfile()
        .map_err(|error| format!("could not create stderr capture for {program}: {error}"))?;
    let child_stdout = stdout_file
        .try_clone()
        .map_err(|error| format!("could not clone stdout capture for {program}: {error}"))?;
    let child_stderr = stderr_file
        .try_clone()
        .map_err(|error| format!("could not clone stderr capture for {program}: {error}"))?;
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(child_stdout))
        .stderr(Stdio::from(child_stderr));
    configure_process_tree(&mut command);
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("could not start {program}: {e}"))?;

    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let mut cleanup_diagnostic = None;
    let exit_ok = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.success(),
            Ok(None) => {
                if Instant::now() >= deadline {
                    timed_out = true;
                    cleanup_diagnostic = terminate_and_reap_process_tree(&mut child, program);
                    break false;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) => return Err(format!("waiting on {program}: {e}")),
        }
    };
    if timed_out && cleanup_diagnostic.is_some() {
        cleanup_diagnostic = cleanup_diagnostic.map(|diagnostic| {
            format!("{diagnostic}; timeout output is a bounded snapshot and may be truncated")
        });
    }
    let stdout = read_capture(&mut stdout_file, "stdout", program)?;
    let stderr = read_capture(&mut stderr_file, "stderr", program)?;
    Ok(BoundedRun {
        exit_ok,
        stdout,
        stderr,
        timed_out,
        cleanup_diagnostic,
    })
}

fn read_capture(file: &mut File, stream: &str, program: &str) -> Result<String, String> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("could not seek {stream} capture for {program}: {error}"))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("could not read {stream} capture for {program}: {error}"))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(unix)]
pub(crate) fn configure_process_tree(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(windows)]
pub(crate) fn configure_process_tree(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
}

#[cfg(unix)]
fn terminate_process_tree(child: &mut Child) -> Option<String> {
    let group = format!("-{}", child.id());
    terminate_with_helper("kill", &["-s", "KILL", "--", &group], child)
}

pub(crate) fn terminate_and_reap_process_tree(child: &mut Child, program: &str) -> Option<String> {
    merge_diagnostic(
        terminate_process_tree(child),
        reap_child_bounded(child, program),
    )
}

#[cfg(windows)]
fn terminate_process_tree(child: &mut Child) -> Option<String> {
    let pid = child.id().to_string();
    terminate_with_helper("taskkill", &["/PID", &pid, "/T", "/F"], child)
}

fn terminate_with_helper(program: &str, args: &[&str], child: &mut Child) -> Option<String> {
    let mut helper = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(helper) => helper,
        Err(error) => {
            let fallback = child.kill();
            return Some(format!(
                "exact process-tree cleanup could not start {program}: {error}; direct-child fallback: {}",
                cleanup_result(fallback)
            ));
        }
    };
    let deadline = Instant::now() + TREE_CLEANUP_TIMEOUT;
    loop {
        match helper.try_wait() {
            Ok(Some(status)) if status.success() => return None,
            Ok(Some(status)) => {
                let fallback = child.kill();
                return Some(format!(
                    "exact process-tree cleanup {program} exited {status}; direct-child fallback: {}",
                    cleanup_result(fallback)
                ));
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(POLL_INTERVAL),
            Ok(None) => {
                let helper_diagnostic = stop_helper_bounded(&mut helper, program);
                let fallback = child.kill();
                return merge_diagnostic(Some(format!(
                    "exact process-tree cleanup {program} timed out after {}s; direct-child fallback: {}",
                    TREE_CLEANUP_TIMEOUT.as_secs(),
                    cleanup_result(fallback)
                )), helper_diagnostic);
            }
            Err(error) => {
                let fallback = child.kill();
                return Some(format!(
                    "waiting for exact process-tree cleanup {program} failed: {error}; direct-child fallback: {}",
                    cleanup_result(fallback)
                ));
            }
        }
    }
}

fn stop_helper_bounded(helper: &mut Child, program: &str) -> Option<String> {
    let kill_result = helper.kill();
    let deadline = Instant::now() + HELPER_REAP_TIMEOUT;
    loop {
        match helper.try_wait() {
            Ok(Some(_)) => return None,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(POLL_INTERVAL),
            Ok(None) => {
                return Some(format!(
                    "cleanup helper {program} did not reap within {}s after kill ({})",
                    HELPER_REAP_TIMEOUT.as_secs(),
                    cleanup_result(kill_result)
                ));
            }
            Err(error) => {
                return Some(format!(
                    "could not reap cleanup helper {program} after kill: {error}"
                ));
            }
        }
    }
}

fn reap_child_bounded(child: &mut Child, program: &str) -> Option<String> {
    let deadline = Instant::now() + TREE_CLEANUP_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return None,
            Ok(None) if Instant::now() < deadline => {
                let _ = child.kill();
                std::thread::sleep(POLL_INTERVAL);
            }
            Ok(None) => {
                return Some(format!(
                    "direct child {program} did not reap within {}s of tree termination",
                    TREE_CLEANUP_TIMEOUT.as_secs()
                ));
            }
            Err(error) => {
                return Some(format!(
                    "could not reap direct child {program} after tree termination: {error}"
                ));
            }
        }
    }
}

fn merge_diagnostic(first: Option<String>, second: Option<String>) -> Option<String> {
    match (first, second) {
        (Some(first), Some(second)) => Some(format!("{first}; {second}")),
        (Some(diagnostic), None) | (None, Some(diagnostic)) => Some(diagnostic),
        (None, None) => None,
    }
}

fn cleanup_result(result: std::io::Result<()>) -> &'static str {
    if result.is_ok() {
        "succeeded"
    } else {
        "failed"
    }
}

/// Drives the solver-under-test against one task workspace and returns its
/// capability scorecard JSON.
pub trait Solver {
    /// Solve one task in `workspace`, returning the raw scorecard JSON.
    ///
    /// # Errors
    /// A plain-language message; the matrix isolates it as a failed cell.
    fn solve(&mut self, workspace: &Path, spec: &SolveSpec) -> Result<String, String>;
}

/// The live solver: `localpilot eval` headless in the task workspace.
pub struct LocalPilotSolver {
    /// The binary to invoke (a name on PATH or an explicit path).
    pub bin: String,
    /// Wall-clock bound per task.
    pub timeout: Duration,
}

impl Solver for LocalPilotSolver {
    fn solve(&mut self, workspace: &Path, spec: &SolveSpec) -> Result<String, String> {
        if !workspace.is_dir() {
            return Err(format!("workspace not found: {}", workspace.display()));
        }
        // A coached cell is driven through the solver's MCP serve surface by
        // the deterministic scripted coach instead of a headless eval.
        if let Some(script) = &spec.coach {
            return crate::coach::drive_coached(
                &self.bin,
                workspace,
                spec,
                Path::new(script),
                self.timeout,
            );
        }
        let run = run_bounded(&self.bin, &eval_args(spec), Some(workspace), self.timeout)?;
        if run.timed_out {
            let cleanup = run
                .cleanup_diagnostic
                .as_deref()
                .map_or_else(String::new, |diagnostic| format!("; {diagnostic}"));
            return Err(format!(
                "'{} eval' timed out after {}s (arm '{}', task '{}'){cleanup}",
                self.bin,
                self.timeout.as_secs(),
                spec.arm,
                spec.task
            ));
        }
        if !run.exit_ok {
            return Err(format!(
                "'{} eval' failed (arm '{}', task '{}'): {}",
                self.bin,
                spec.arm,
                spec.task,
                run.stderr.trim()
            ));
        }
        if run.stdout.trim().is_empty() {
            return Err(format!(
                "'{} eval' produced no scorecard on stdout (arm '{}', task '{}')",
                self.bin, spec.arm, spec.task
            ));
        }
        Ok(run.stdout)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    struct ChildGuard(Child);

    impl ChildGuard {
        fn spawn_sleeper() -> Self {
            let mut command = if cfg!(windows) {
                let mut command = Command::new("pwsh");
                command.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 60"]);
                command
            } else {
                let mut command = Command::new("sleep");
                command.arg("60");
                command
            };
            Self(command.spawn().expect("spawn unrelated sibling"))
        }

        fn is_alive(&mut self) -> bool {
            self.0.try_wait().expect("query sibling").is_none()
        }
    }

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[test]
    fn the_eval_argv_shape_is_pinned() {
        let mut spec = SolveSpec::new("apex", "full", "t-1", "Fix the failing test in two words");
        assert_eq!(
            eval_args(&spec),
            [
                "eval",
                "--model",
                "apex",
                "--arm",
                "full",
                "--task",
                "t-1",
                "--permission",
                "bypass",
                "Fix the failing test in two words"
            ]
        );
        // The verify and warm arms add their flags before the positional.
        spec.verify = true;
        spec.learn = true;
        let argv = eval_args(&spec);
        assert_eq!(
            argv[9..],
            [
                "--verify".to_string(),
                "--learn".to_string(),
                spec.problem.clone()
            ]
        );
        // The multi-word problem stays a single argv entry.
        assert_eq!(argv.last().unwrap(), &spec.problem);
    }

    #[test]
    fn a_bounded_run_captures_output_and_exit() {
        let (program, args) = if cfg!(windows) {
            (
                "cmd",
                vec![
                    "/C".to_string(),
                    "echo hello & echo expected-error 1>&2 & exit /b 7".to_string(),
                ],
            )
        } else {
            (
                "sh",
                vec![
                    "-c".to_string(),
                    "echo hello; echo expected-error >&2; exit 7".to_string(),
                ],
            )
        };
        let run = run_bounded(program, &args, None, Duration::from_secs(30)).unwrap();
        assert!(!run.exit_ok);
        assert!(!run.timed_out);
        assert!(run.stdout.contains("hello"));
        assert!(run.stderr.contains("expected-error"));
        assert_eq!(run.cleanup_diagnostic, None);
    }

    #[test]
    fn a_hung_descendant_tree_is_killed_without_touching_an_unrelated_sibling() {
        let mut sibling = ChildGuard::spawn_sleeper();
        let (program, args) = if cfg!(windows) {
            (
                "pwsh",
                vec![
                    "-NoProfile".to_string(),
                    "-Command".to_string(),
                    "$child = Start-Process -FilePath pwsh -ArgumentList @('-NoProfile', \
                     '-Command', 'Start-Sleep -Seconds 60') -NoNewWindow -PassThru; \
                     [Console]::Out.WriteLine(\"descendant-pid:$($child.Id)\"); \
                     [Console]::Error.WriteLine('grandchild-stderr'); \
                     Wait-Process -Id $child.Id"
                        .to_string(),
                ],
            )
        } else {
            (
                "sh",
                vec![
                    "-c".to_string(),
                    "sleep 60 & child=$!; echo descendant-pid:$child; \
                     echo grandchild-stderr >&2; wait $child"
                        .to_string(),
                ],
            )
        };
        let started = Instant::now();
        let run = run_bounded(program, &args, None, Duration::from_secs(1)).unwrap();
        assert!(run.timed_out);
        assert!(!run.exit_ok);
        assert!(started.elapsed() < Duration::from_secs(15));
        assert!(run.stderr.contains("grandchild-stderr"));
        let descendant_pid = run
            .stdout
            .lines()
            .find_map(|line| line.trim().strip_prefix("descendant-pid:"))
            .expect("fixture reports its descendant pid")
            .parse::<u32>()
            .expect("descendant pid is numeric");
        let gone_deadline = Instant::now() + Duration::from_secs(5);
        while process_is_alive(descendant_pid) && Instant::now() < gone_deadline {
            std::thread::sleep(Duration::from_millis(25));
        }
        assert!(
            !process_is_alive(descendant_pid),
            "the exact descendant tree must be gone"
        );
        assert!(sibling.is_alive(), "an unrelated sibling must survive");
        assert_eq!(run.cleanup_diagnostic, None);
    }

    #[test]
    fn inherited_output_handles_cannot_delay_a_completed_parent() {
        let (program, args) = if cfg!(windows) {
            (
                "pwsh",
                vec![
                    "-NoProfile".to_string(),
                    "-Command".to_string(),
                    "$child = Start-Process -FilePath pwsh -ArgumentList @('-NoProfile', \
                     '-Command', 'Start-Sleep -Seconds 5') -NoNewWindow -PassThru; \
                     [Console]::Out.WriteLine(\"descendant-pid:$($child.Id)\"); \
                     [Console]::Error.WriteLine('parent-complete')"
                        .to_string(),
                ],
            )
        } else {
            (
                "sh",
                vec![
                    "-c".to_string(),
                    "sleep 5 & child=$!; echo descendant-pid:$child; \
                     echo parent-complete >&2"
                        .to_string(),
                ],
            )
        };
        let started = Instant::now();
        let run = run_bounded(program, &args, None, Duration::from_secs(30)).unwrap();
        let descendant_pid = run
            .stdout
            .lines()
            .find_map(|line| line.trim().strip_prefix("descendant-pid:"))
            .expect("fixture reports its inherited-handle descendant")
            .parse::<u32>()
            .expect("descendant pid is numeric");
        let descendant_was_alive = process_is_alive(descendant_pid);
        terminate_fixture(descendant_pid);
        assert!(
            descendant_was_alive,
            "fixture descendant must still be alive"
        );
        assert!(run.exit_ok);
        assert!(!run.timed_out);
        assert!(run.stderr.contains("parent-complete"));
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "a descendant retaining output handles must not delay its completed parent"
        );
    }

    #[cfg(unix)]
    fn process_is_alive(pid: u32) -> bool {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[cfg(unix)]
    fn terminate_fixture(pid: u32) {
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    }
    #[cfg(windows)]
    fn process_is_alive(pid: u32) -> bool {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .is_ok_and(|output| {
                String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\""))
            })
    }

    #[cfg(windows)]
    fn terminate_fixture(pid: u32) {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[test]
    fn the_live_solver_refuses_a_missing_workspace() {
        let mut solver = LocalPilotSolver {
            bin: "localpilot".to_string(),
            timeout: Duration::from_secs(1),
        };
        let spec = SolveSpec::new("m", "full", "t", "p");
        let err = solver
            .solve(Path::new("Z:/definitely/not/here"), &spec)
            .unwrap_err();
        assert!(err.contains("workspace not found"));
    }
}
