# Reap leaked/runaway grading + solution procs. On these exercises any host
# python/node/java invocation (the model running its own tests or inline
# evaluation via run_shell) finishes in seconds; one still alive after MaxAgeMin
# is a hung or runaway solution (infinite loop / memory blowup) whose process
# orphaned. Left alone they accumulate and exhaust RAM — enough to OOM the model
# server. Matches by runtime + age (not just `pytest`/`unittest`, since a runaway
# is often `python -c "..."`), and never touches the no-think proxy. LocalPilot's
# run_shell now reaps its own process tree on timeout, so this is the backstop for
# anything that escapes (the grader path, an over-long tool timeout). Exits when
# the supervisor dies.
param([int]$SupervisorPid, [int]$MaxAgeMin = 3, [int]$IntervalSec = 60)
$runtimes = @('python.exe', 'node.exe', 'java.exe')
while ($true) {
    if ($SupervisorPid -and -not (Get-Process -Id $SupervisorPid -ErrorAction SilentlyContinue)) {
        "reaper: supervisor $SupervisorPid gone, exiting"; break
    }
    $cut = (Get-Date).AddMinutes(-$MaxAgeMin)
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='node.exe' OR Name='java.exe'" |
        Where-Object {
            $_.CreationDate -lt $cut -and
            $_.CommandLine -notmatch 'no-think-proxy' -and
            $runtimes -contains $_.Name
        } |
        ForEach-Object {
            "reaper: kill $($_.Name) pid $($_.ProcessId) ($([int]($_.WorkingSetSize/1MB))MB, started $($_.CreationDate))"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    Start-Sleep -Seconds $IntervalSec
}
