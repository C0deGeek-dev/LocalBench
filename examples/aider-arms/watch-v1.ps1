# Watch the v1.0 4-arm re-run. Exit (re-invoking the agent) when the supervisor
# dies, all agentic cells land, or progress stalls 45 min. Reports per-arm
# standings + the warm store's accumulated-lesson count.
param([int]$SupervisorPid, [int]$AgenticTarget = 900)  # full+fair+claude-code+warm x 225
$ledger = "C:\Users\David\.localbench\runs\aider-arms-v1\cells.jsonl"
$warmDb = "C:\Users\David\.localbench\runs\aider-arms-v1\warm-global\memory\.localmind\localmind.sqlite"
$last = -1; $stall = 0; $elapsed = 0; $maxSec = 96 * 3600
function Get-Cells { @(Get-Content $ledger -ErrorAction SilentlyContinue | ForEach-Object { try { $_ | ConvertFrom-Json } catch {} }) }
function Agentic { param($c) @($c | Where-Object { $_.arm -in @('full', 'fair', 'claude-code', 'warm') }) }
while ($true) {
    Start-Sleep -Seconds 180; $elapsed += 180
    $c = Get-Cells; $n = @(Agentic $c).Count
    $alive = Get-Process -Id $SupervisorPid -ErrorAction SilentlyContinue
    if ($n -ne $last) { $last = $n; $stall = 0 } else { $stall += 180 }
    if (-not $alive) { "WATCH: supervisor exited. agentic=$n/$AgenticTarget"; break }
    if ($n -ge $AgenticTarget) { "WATCH: complete. agentic=$n"; break }
    if ($stall -ge 2700) { "WATCH: stalled ${stall}s. agentic=$n"; break }
    if ($elapsed -ge $maxSec) { "WATCH: max watch time. agentic=$n"; break }
}
$c = Get-Cells
"--- v1 standings ---"
foreach ($arm in 'baseline', 'full', 'fair', 'claude-code', 'warm') {
    $g = @($c | Where-Object { $_.arm -eq $arm })
    "{0,-12} {1}/{2}" -f $arm, @($g | Where-Object { $_.solved }).Count, $g.Count
}
# Warm accumulation: how many lessons made it into the global store (the 'smarter
# as you use it' signal). 0 after many exercises => warm is inert; flag it.
if (Test-Path $warmDb) {
    $q = "import sqlite3; c=sqlite3.connect(r'$warmDb'); cur=c.cursor();" +
    "print('warm accepted memory:', cur.execute('select count(*) from memory_index').fetchone()[0]);" +
    "print('warm review items:', cur.execute('select count(*) from review_items').fetchone()[0]); c.close()"
    $q | python - 2>$null
}
