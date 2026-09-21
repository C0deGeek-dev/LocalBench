//! How much memory the host can still commit.
//!
//! Windows has a hard commit limit — RAM plus page file — and an allocation
//! past it fails: llama.cpp's CUDA initialisation fails that way when a model
//! loaded without mmap makes its CPU-side weights private while the display
//! driver backs the GPU's memory with host commit. The OS reports the free part
//! directly (`Win32_OperatingSystem.FreeVirtualMemory`). sysinfo sees it only
//! while commit exceeds physical RAM, because it reports commit above RAM as
//! "used swap"; below that it cannot tell the commit charge at all.
//!
//! Other systems have no such limit: the kernel can hand out the available RAM
//! plus the free swap.

use std::path::Path;
use std::time::Duration;

use localx_llama_runtime::tool::run_tool;

const GIB: f64 = 1_073_741_824.0;

/// Longest the OS query may take; it normally answers in about 0.2 s.
const QUERY_TIMEOUT: Duration = Duration::from_secs(10);

/// The host's commit headroom in GiB, asked of the OS where it has a limit
/// (Windows spawns one short query). `None` when it cannot be read.
#[must_use]
pub fn probe_gb(system: &sysinfo::System) -> Option<f64> {
    if cfg!(windows) {
        windows_free_virtual_kib().map(|kib| kib as f64 / 1_048_576.0)
    } else {
        from_sysinfo_gb(system)
    }
}

/// The commit headroom sysinfo alone can tell, in GiB, without spawning
/// anything: on Windows only while commit exceeds RAM (then exact), elsewhere
/// always. `None` when it cannot tell.
#[must_use]
pub fn from_sysinfo_gb(system: &sysinfo::System) -> Option<f64> {
    from_counters(
        cfg!(windows),
        system.available_memory(),
        system.free_swap(),
        system.used_swap(),
    )
    .map(|bytes| bytes as f64 / GIB)
}

/// [`from_sysinfo_gb`] over raw counters, in bytes.
#[must_use]
pub fn from_counters(windows: bool, available: u64, free_swap: u64, used_swap: u64) -> Option<u64> {
    if windows {
        // sysinfo's swap on Windows is the commit limit and charge minus
        // physical RAM, clamped at zero: with nothing "used", the charge is
        // below RAM and unknown, and the free part is just the page file.
        (used_swap > 0).then_some(free_swap)
    } else {
        Some(available.saturating_add(free_swap))
    }
}

/// Free virtual memory (commit limit minus commit charge) in KiB, from the OS.
fn windows_free_virtual_kib() -> Option<u64> {
    let args = [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "(Get-CimInstance Win32_OperatingSystem -Property FreeVirtualMemory).FreeVirtualMemory",
    ]
    .map(str::to_string);
    let output = run_tool(Path::new("powershell"), &args, QUERY_TIMEOUT)?;
    if !output.success {
        return None;
    }
    parse_free_virtual_kib(&output.stdout)
}

/// Parse the query's answer: one positive integer (KiB).
#[must_use]
pub fn parse_free_virtual_kib(stdout: &str) -> Option<u64> {
    stdout.trim().parse::<u64>().ok().filter(|kib| *kib > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB_BYTES: u64 = 1_073_741_824;

    #[test]
    fn sysinfo_only_knows_windows_commit_headroom_while_commit_exceeds_ram() {
        // Commit charge below RAM: sysinfo reports no used swap and a free
        // swap equal to the page file, which is not the headroom.
        assert_eq!(from_counters(true, 53 * GIB_BYTES, 57 * GIB_BYTES, 0), None);
        // Commit above RAM: the free part is exactly the headroom.
        assert_eq!(
            from_counters(true, 2 * GIB_BYTES, 5 * GIB_BYTES, 11 * GIB_BYTES),
            Some(5 * GIB_BYTES)
        );
    }

    #[test]
    fn without_a_commit_limit_the_headroom_is_available_ram_plus_free_swap() {
        assert_eq!(
            from_counters(false, 40 * GIB_BYTES, 12 * GIB_BYTES, 0),
            Some(52 * GIB_BYTES)
        );
        assert_eq!(from_counters(false, u64::MAX, 1, 0), Some(u64::MAX));
    }

    #[test]
    fn the_os_answer_is_one_positive_kib_count() {
        assert_eq!(parse_free_virtual_kib("110761856\r\n"), Some(110_761_856));
        assert_eq!(parse_free_virtual_kib(""), None);
        assert_eq!(parse_free_virtual_kib("0"), None);
        assert_eq!(
            parse_free_virtual_kib("Get-CimInstance : Access denied"),
            None
        );
    }

    /// Live check: prints the probe's answer for comparison with the OS's own
    /// counters (on Windows: Commit Limit − Committed Bytes).
    #[test]
    #[ignore = "reads the real host"]
    fn prints_this_hosts_commit_headroom() {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        println!(
            "commit headroom: probe {:?} GiB, sysinfo {:?} GiB",
            probe_gb(&system),
            from_sysinfo_gb(&system)
        );
    }
}
