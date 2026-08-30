# v1.0 re-run status table: arms per language + the key deltas.
# Read-only (ledger), GPU-free. Run: pwsh C:\Users\David\.localbench\status-v1.ps1
$ledger = "C:\Users\David\.localbench\runs\aider-arms-v1\cells.jsonl"
$warmDb = "C:\Users\David\.localbench\runs\aider-arms-v1\warm-global\localmind.sqlite"
$arms = 'baseline', 'full', 'claude-code', 'warm'
$langs = 'python', 'go', 'rust', 'cpp', 'javascript', 'java'
$corpRoot = "C:\Users\David\.localbench\external-corpus\aider-polyglot"
function Total { param($l) @(Get-ChildItem "$corpRoot\$l\exercises\practice" -Directory -ErrorAction SilentlyContinue).Count }
$c = @(Get-Content $ledger -ErrorAction SilentlyContinue | ForEach-Object { try { $_ | ConvertFrom-Json } catch {} })

function Rate { param($cells) $n = @($cells).Count; $p = @($cells | Where-Object { $_.solved }).Count; @{ p = $p; n = $n } }
function Cell { param($r) if ($r.n -eq 0) { '   -   ' } else { '{0,2}/{1,-2}' -f $r.p, $r.n } }

"`n  v1.0 re-run — solved/total per language, model pinned (APEX)`n"
"  {0,-11} {1,-6} {2,-9} {3,-9} {4,-9} {5,-9}" -f 'lang', 'total', 'baseline', 'full', 'claude', 'warm'
"  " + ('-' * 64)
foreach ($l in $langs) {
    $row = foreach ($a in $arms) { Cell (Rate (@($c | Where-Object { $_.lang -eq $l -and $_.arm -eq $a }))) }
    "  {0,-11} {1,-6} {2,-9} {3,-9} {4,-9} {5,-9}" -f $l, (Total $l), $row[0], $row[1], $row[2], $row[3]
}
"  " + ('-' * 64)
$corpusTotal = ($langs | ForEach-Object { Total $_ } | Measure-Object -Sum).Sum
$tot = foreach ($a in $arms) { Rate (@($c | Where-Object { $_.arm -eq $a })) }
$totRow = $tot | ForEach-Object { Cell $_ }
"  {0,-11} {1,-6} {2,-9} {3,-9} {4,-9} {5,-9}" -f 'TOTAL', $corpusTotal, $totRow[0], $totRow[1], $totRow[2], $totRow[3]

function Pct { param($r) if ($r.n) { [math]::Round(100 * $r.p / $r.n) } else { 0 } }
function Sgn { param($x) if ($x -ge 0) { "+$x" } else { "$x" } }
$b = $tot[0]; $f = $tot[1]; $cc = $tot[2]; $w = $tot[3]
"`n  Key deltas (whole-corpus, on the cells run so far):"
"    harness vs raw  (full - baseline)    : $(Pct $f)% - $(Pct $b)% = $(Sgn ((Pct $f)-(Pct $b)))"
"    learning lift   (warm - full)        : $(Pct $w)% - $(Pct $f)% = $(Sgn ((Pct $w)-(Pct $f)))"
"    LP vs CC        (warm - claude-code) : $(Pct $w)% - $(Pct $cc)% = $(Sgn ((Pct $w)-(Pct $cc)))"

# Warm accumulation (the 'smarter as you use it' signal; 0 after many cells = inert)
if (Test-Path $warmDb) {
    $q = "import sqlite3;c=sqlite3.connect(r'$warmDb');x=c.cursor();" +
    "print('  warm store: ',x.execute('select count(*) from memory_index').fetchone()[0],'accepted lessons,'," +
    "x.execute('select count(*) from review_items').fetchone()[0],'in review');c.close()"
    "`n" + ($q | python - 2>$null)
}
# Grader health: a wedged Docker engine makes every grade time out — the failure
# that silently burns hours. Surface it: a bounded engine probe + a count of grades
# that timed out (the runner's circuit breaker yields the run after 3 in a row).
$timeouts = @($c | Where-Object { $_.note -match 'grade timed out' }).Count
$dexe = "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe"
$dj = Start-Job -ArgumentList $dexe { param($d) & $d run --rm busybox true *>$null; $LASTEXITCODE }
$dres = if (Wait-Job $dj -Timeout 20) { Receive-Job $dj } else { Stop-Job $dj; 999 }
Remove-Job $dj -Force -ErrorAction SilentlyContinue
$dstate = if ($dres -eq 0) { 'OK (ran a container)' } else { 'WEDGED/DOWN  <-- fix Docker before running' }
"`n  grader: docker $dstate ; grade-timeout cells: $timeouts"
# Fairness caveat: a claude-code cell that hit its wall-clock budget is counted
# unsolved but may be budget-bound, not capability-bound — surface it so the cc
# rate is read as a floor, not a pure capability number.
$ccTo = @($c | Where-Object { $_.arm -eq 'claude-code' -and $_.note -match "claude -p' timed out" }).Count
if ($ccTo -gt 0) { "  caveat: $ccTo claude-code cell(s) hit the wall-clock budget (counted unsolved) — cc rate is a floor" }
$sup = Get-Content "C:\Users\David\.localbench\runs\aider-arms-v1\.supervisor-pid" -ErrorAction SilentlyContinue
"  supervisor pid $sup : $((Get-Process -Id $sup -ErrorAction SilentlyContinue) ? 'running' : 'GONE')`n"
