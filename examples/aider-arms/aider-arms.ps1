param(
    [string]$Phase = 'selftest',                 # selftest | solve
    [string]$Languages = 'python,go,rust,cpp',
    [string]$Arms = 'baseline,full',
    [int]$Limit = 0,                              # 0 = all exercises in each language
    [int]$EvalTimeout = 900,                      # external per-exercise backstop (s); ABOVE the internal 600s turn-timeout so the loop's clean finalize wins the race (no external SIGKILL masking a scorecard)
    [string]$Model = 'q3635ba3bapex',
    [string]$Proxy = 'http://127.0.0.1:11435/v1/messages',
    [string]$EmbedBaseUrl = 'http://127.0.0.1:8090',  # CPU embedding server (LocalBox llmembedserve); warm arm only, no GPU VRAM
    [string]$RunTag = ''                          # suffix for the run dir/ledger (e.g. '-ablation') to isolate a side run
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module D:\repos\LocalX\LocalBench\src\LocalBench.psm1 -Force
. D:\repos\LocalX\LocalBench\src\adapters\localpilot-eval-solver.ps1

$dbin = "$env:ProgramFiles\Docker\Docker\resources\bin"; $env:PATH = "$dbin;$env:PATH"
$docker = "$dbin\docker.exe"
$corpusRoot = Join-Path $HOME '.localbench\external-corpus\aider-polyglot'
$runRoot = Join-Path $HOME ".localbench\runs\aider-arms$RunTag"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

# split on comma OR whitespace — a comma-list can collapse to a space crossing
# Start-Process into a child pwsh, which once mislabelled arms as one "baseline full".
$langs = @($Languages -split '[,\s]+' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$armList = @($Arms -split '[,\s]+' | ForEach-Object { $_.Trim() } | Where-Object { $_ })

# ---- per-language config -----------------------------------------------------
# stub:  rel paths the solver edits (detected from the exercise dir)
# gold:  @{from=<.meta path>; to=<stub rel path>} pairs for the gold (self-test pass)
# image / test: container + in-container test command (cwd=/work, --network=none)
$cfg = @{
    python = @{
        image = 'python:3.12-slim'
        stub  = { param($d) Get-ChildItem $d -Filter '*.py' | Where-Object { $_.Name -notlike '*_test.py' } | ForEach-Object { $_.Name } }
        gold  = { param($d) @(@{ from = '.meta/example.py'; to = (Get-ChildItem $d -Filter '*.py' | Where-Object { $_.Name -notlike '*_test.py' } | Select-Object -First 1).Name }) }
        test  = 'f=$(ls *_test.py | head -1); python -B -m unittest "${f%.py}"'
    }
    go = @{
        image = 'golang:1.23-bookworm'
        stub  = { param($d) Get-ChildItem $d -Filter '*.go' | Where-Object { $_.Name -notlike '*_test.go' } | ForEach-Object { $_.Name } }
        gold  = { param($d) @(@{ from = '.meta/example.go'; to = (Get-ChildItem $d -Filter '*.go' | Where-Object { $_.Name -notlike '*_test.go' } | Select-Object -First 1).Name }) }
        test  = 'GOFLAGS=-mod=mod GOPROXY=off GOCACHE=/tmp/gc GOPATH=/tmp/gp go test ./...'
    }
    rust = @{
        # Current stable rust. The corpus's declared deps resolve to their latest
        # compatible versions (e.g. `time = "0.3"` -> time-core 0.1.9, which needs
        # the edition2024 cargo feature) and those versions' MSRV has risen past
        # the old pinned 1.82 image, so even the offline cache warm failed on 1.82.
        # The bump is backward-compatible for the edition-2021 exercise code; it
        # only lets the modern dependency crates build. The warm step uses this
        # same image, so the cached crates always match the grade toolchain.
        image = 'rust:1.96-slim-bookworm'
        stub  = { param($d) , 'src/lib.rs' }
        gold  = { param($d) @(@{ from = '.meta/example.rs'; to = 'src/lib.rs' }) }
        # Offline grade: --network=none + CARGO_NET_OFFLINE=true. The exercises'
        # declared deps (time/anyhow/rand) + crates an agentic arm may add are
        # pre-vendored into the shared warmed cargo registry volume (warmed once
        # by Initialize-CargoCache), exactly mirroring the gradle cache — so a
        # fresh workspace compiles + tests fully offline.
        test  = 'CARGO_NET_OFFLINE=true CARGO_TARGET_DIR=/tmp/t cargo test -- --include-ignored'
        mounts = @('-v', 'aider-cargo-cache:/usr/local/cargo/registry')   # shared warmed cache
    }
    cpp = @{
        image = 'gcc:14'
        stub  = { param($d) Get-ChildItem $d -File | Where-Object { ($_.Name -like '*.cpp' -or $_.Name -like '*.h') -and $_.Name -notlike '*_test.cpp' } | ForEach-Object { $_.Name } }
        gold  = { param($d)
            $g = @()
            if (Test-Path (Join-Path $d '.meta/example.cpp')) { $g += @{ from = '.meta/example.cpp'; to = (Get-ChildItem $d -Filter '*.cpp' | Where-Object { $_.Name -notlike '*_test.cpp' } | Select-Object -First 1).Name } }
            if (Test-Path (Join-Path $d '.meta/example.h')) { $g += @{ from = '.meta/example.h'; to = (Get-ChildItem $d -Filter '*.h' | Select-Object -First 1).Name } }
            $g }
        test  = 'g++ -std=c++17 -DEXERCISM_RUN_ALL_TESTS -I. -Itest *.cpp test/tests-main.cpp -o /tmp/run 2>&1 && /tmp/run'
    }
    javascript = @{
        image  = 'node:22-bookworm-slim'
        stub   = { param($d) , "$((Split-Path $d -Leaf)).js" }
        gold   = { param($d) @(@{ from = '.meta/proof.ci.js'; to = "$((Split-Path $d -Leaf)).js" }) }
        enable = { param($ws) Get-ChildItem $ws -Filter '*.spec.js' | ForEach-Object {
                (Get-Content $_.FullName -Raw) -replace '\bxtest\(', 'test(' -replace '\bxit\(', 'it(' -replace '\.skip\(', '(' |
                    Set-Content -LiteralPath $_.FullName -Encoding utf8 } }
        prep   = 'npm install --no-audit --no-fund --loglevel=error'
        test   = 'npx jest'
    }
    java = @{
        image  = 'gradle:8.10-jdk21'
        stub   = { param($d) Get-ChildItem (Join-Path $d 'src/main/java') -Recurse -Filter '*.java' | ForEach-Object { $_.FullName.Substring($d.Length).TrimStart('\', '/') -replace '\\', '/' } }
        gold   = { param($d)
            $ref = Get-ChildItem (Join-Path $d '.meta/src/reference/java') -Recurse -Filter '*.java' -ErrorAction SilentlyContinue
            $main = Get-ChildItem (Join-Path $d 'src/main/java') -Recurse -Filter '*.java'
            $g = @()
            foreach ($r in $ref) { $m = $main | Where-Object { $_.Name -eq $r.Name } | Select-Object -First 1; if ($m) { $g += @{ from = ($r.FullName.Substring($d.Length).TrimStart('\', '/') -replace '\\', '/'); to = ($m.FullName.Substring($d.Length).TrimStart('\', '/') -replace '\\', '/') } } }
            $g }
        enable = { param($ws) Get-ChildItem (Join-Path $ws 'src/test') -Recurse -Filter '*.java' | ForEach-Object {
                (Get-Content $_.FullName -Raw) -replace '(?m)^\s*@Disabled.*$', '' |
                    Set-Content -LiteralPath $_.FullName -Encoding utf8 } }
        # No per-exercise network prep: the shared gradle cache is warmed once
        # (gradle dist + junit) so each fresh workspace compiles + runs fully offline.
        test   = 'gradle --no-daemon --offline test'   # no pipe — $LASTEXITCODE must reflect gradle
        mounts = @('-v', 'aider-gradle-cache:/home/gradle/.gradle')   # shared warmed cache
    }
}

function New-Workspace {
    param([string]$Lang, [string]$Ex, [ValidateSet('stub', 'gold', 'solve')][string]$Kind)
    $src = Join-Path $corpusRoot "$Lang/exercises/practice/$Ex"
    $ws = Join-Path $runRoot ("$Lang-$Ex-$Kind-" + [System.IO.Path]::GetRandomFileName().Substring(0, 6))
    if (Test-Path $ws) { Remove-Item $ws -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $ws | Out-Null
    # copy everything except the gold (.meta) — keep stub, tests, build files, .docs
    Get-ChildItem $src -Force | Where-Object { $_.Name -ne '.meta' } | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $ws $_.Name) -Recurse -Force
    }
    if (Test-Path (Join-Path $src '.docs/instructions.md')) {
        Copy-Item (Join-Path $src '.docs/instructions.md') (Join-Path $ws 'INSTRUCTIONS.md') -Force
    }
    if ($Kind -eq 'gold') {
        foreach ($g in (& $cfg[$Lang].gold $src)) {
            Copy-Item (Join-Path $src $g.from) (Join-Path $ws $g.to) -Force
        }
    }
    # enable all tests (Exercism ships most disabled; the benchmark enables them)
    if ($cfg[$Lang].ContainsKey('enable')) { & $cfg[$Lang].enable $ws }
    return $ws
}

# Run one `docker` invocation bounded by a timeout. A graded test that hangs —
# an infinite loop in the model's solution, or (seen on Docker-for-Windows) a
# wedged engine that leaves the `docker run` CLI blocked with no container — must
# never freeze the whole sweep. On expiry the named container is killed (freeing
# the engine) and the wedged CLI is reaped, and the call returns a non-zero code
# so the cell is recorded failed and the loop continues. Mirrors the solver
# adapter's bounded-process pattern; reads both pipes async to avoid a full-buffer
# deadlock.
function Invoke-DockerBounded {
    param([string[]]$Argv, [string]$ContainerName, [int]$TimeoutSec = 420)
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $docker
    foreach ($a in $Argv) { $psi.ArgumentList.Add([string]$a) }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $o = $proc.StandardOutput.ReadToEndAsync()
    $e = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        try { & $docker kill $ContainerName 2>&1 | Out-Null } catch { }
        try { & $docker rm -f $ContainerName 2>&1 | Out-Null } catch { }
        try { $proc.Kill($true) } catch { }
        return @{ code = 124; out = "grade timed out after ${TimeoutSec}s" }
    }
    $out = ($o.GetAwaiter().GetResult()) + "`n" + ($e.GetAwaiter().GetResult())
    return @{ code = $proc.ExitCode; out = $out }
}

function Invoke-Grade {
    param([string]$Lang, [string]$Workspace, [int]$TimeoutSec = 420)
    # grade on a throwaway COPY mounted rw (compiled langs write build artifacts),
    # network-isolated + ephemeral — the real safety controls. Every docker run is
    # bounded (Invoke-DockerBounded) so a hung test or wedged engine fails the cell
    # instead of freezing the sweep.
    $copy = Join-Path $runRoot ('grade-' + [System.IO.Path]::GetRandomFileName().Substring(0, 8))
    Copy-Item $Workspace $copy -Recurse -Force
    $name = 'lbgrade-' + (Split-Path $copy -Leaf)
    try {
        $img = $cfg[$Lang].image
        $cmd = $cfg[$Lang].test
        # extra mounts (e.g. a shared gradle cache so the dist/deps download once)
        $mounts = if ($cfg[$Lang].ContainsKey('mounts')) { $cfg[$Lang].mounts } else { @() }
        # optional network-ON dependency prep (no model code runs here — only the
        # package manager resolves trusted, pinned test deps). The graded test run
        # below is always --network=none, so model code never has network. Prep can
        # download a toolchain on a cold cache, so it gets a longer bound.
        if ($cfg[$Lang].ContainsKey('prep')) {
            $null = Invoke-DockerBounded -ContainerName "$name-prep" -TimeoutSec 600 -Argv (
                @('run', '--rm', '--name', "$name-prep", '--network=bridge', '-v', "${copy}:/work", '-w', '/work') +
                $mounts + @($img, 'bash', '-c', $cfg[$Lang].prep))
        }
        # `bash -c` (not `-lc`): a login shell re-sources /etc/profile and drops the
        # golang/rust toolchain (/usr/local/go/bin, /usr/local/cargo/bin) from PATH.
        $r = Invoke-DockerBounded -ContainerName $name -TimeoutSec $TimeoutSec -Argv (
            @('run', '--rm', '--name', $name, '--network=none', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-v', "${copy}:/work", '-w', '/work') +
            $mounts + @($img, 'bash', '-c', $cmd))
        $tail = (($r.out -split "`n") | Where-Object { $_.Trim() } | Select-Object -Last 2) -join ' | '
        # A cell is solved only if the test command exits 0 AND tests actually ran.
        # A compile-with-zero-tests (rust/xorcism: `0 passed; 0 failed`) exits 0 but
        # proves nothing — counting it solved inflates the arm. Parse the FULL output
        # (not the 2-line display $tail; the count line is often not in the last two,
        # e.g. rust's doc-test result). Fail closed + log if exit 0 yet no test ran.
        $testsRun = Get-TestCount -Lang $Lang -Output $r.out
        $passed = ($r.code -eq 0) -and ($testsRun -gt 0)
        if (($r.code -eq 0) -and ($testsRun -le 0)) {
            Write-Warning "grade: exit 0 but 0 tests ran in $Lang — not solved (compiled-with-zero-tests or unparsed test output). tail: $tail"
        }
        return @{ passed = $passed; tail = $tail; timedout = ($r.code -eq 124); tests_run = $testsRun }
    }
    finally { Remove-Item $copy -Recurse -Force -ErrorAction SilentlyContinue }
}

function Initialize-CargoCache {
    # Warm the shared `aider-cargo-cache` cargo registry volume so the offline
    # rust grade (--network=none + CARGO_NET_OFFLINE=true) builds every exercise's
    # declared deps + the common crates an agentic arm may add. This is the ONE
    # network step (--network=bridge); it runs once before the rust grades and is
    # idempotent (`cargo fetch` is a no-op when the volume already holds the
    # crates) — mirroring the warmed gradle cache. A vendored cache, never a
    # network path into a graded run (those stay --network=none). The dep set is
    # the recorded, Pester-pinned union from Get-RustCargoCacheDeps.
    param([string]$CacheVolume = 'aider-cargo-cache', [int]$TimeoutSec = 600)
    $img = $cfg['rust'].image
    $deps = Get-RustCargoCacheDeps -CorpusRoot $corpusRoot
    Write-Host ("cargo cache: warming '$CacheVolume' with $($deps.Count) crates ($img)...")
    foreach ($d in $deps) { Write-Host ('  - {0} = {1} ({2})' -f $d.name, $d.version, $d.source) }
    $warm = Join-Path $runRoot ('cargo-warm-' + [System.IO.Path]::GetRandomFileName().Substring(0, 6))
    New-Item -ItemType Directory -Force -Path (Join-Path $warm 'src') | Out-Null
    try {
        New-CargoWarmManifest -Dependency $deps | Set-Content (Join-Path $warm 'Cargo.toml') -Encoding utf8
        '' | Set-Content (Join-Path $warm 'src/lib.rs') -Encoding utf8
        # cargo fetch resolves + downloads the full dependency graph (incl.
        # transitive crates) into the mounted registry; no compile, no offline flag.
        $r = Invoke-DockerBounded -ContainerName 'lbcargo-warm' -TimeoutSec $TimeoutSec -Argv (
            @('run', '--rm', '--name', 'lbcargo-warm', '--network=bridge',
                '-v', "${CacheVolume}:/usr/local/cargo/registry", '-v', "${warm}:/warm", '-w', '/warm',
                $img, 'cargo', 'fetch'))
        if ($r.code -ne 0) {
            Write-Warning "cargo cache warm failed (code $($r.code)) — the offline rust grade may fail loud for some deps:`n$($r.out)"
            return $false
        }
        Write-Host 'cargo cache: warm OK (deps vendored; the grade stays --network=none + offline).'
        return $true
    }
    finally { Remove-Item $warm -Recurse -Force -ErrorAction SilentlyContinue }
}

function Test-DockerHealthy {
    # The engine can wedge (Docker-for-Windows / WSL2): `docker info` and every
    # `docker run` then hang, so every grade times out and the sweep silently burns
    # hours marking cells failed. Prove the engine actually RUNS a container,
    # bounded, before trusting it — fail loud and early instead.
    param([int]$TimeoutSec = 60)
    $r = Invoke-DockerBounded -ContainerName 'lbgrade-healthcheck' -TimeoutSec $TimeoutSec `
        -Argv @('run', '--rm', '--name', 'lbgrade-healthcheck', 'busybox', 'true')
    return ($r.code -eq 0)
}

function Test-EmbedEndpointHealthy {
    # The warm arm depends on the CPU embedding server (embedding-backed semantic
    # dedup + retrieval rerank over the shared global store). A down/wedged embed
    # endpoint would silently degrade the warm arm to lexical-only mid-run. Prove
    # it actually returns a vector before the first cell — the embeddings analogue
    # of Test-DockerHealthy. CPU-only, so this touches no GPU VRAM.
    param([Parameter(Mandatory)][string]$BaseUrl, [int]$TimeoutSec = 15)
    $body = @{ model = 'embed'; input = @('embedding pre-flight probe') } | ConvertTo-Json -Depth 4
    try {
        $r = Invoke-RestMethod -Uri "$BaseUrl/v1/embeddings" -Method Post -Body $body `
            -ContentType 'application/json' -TimeoutSec $TimeoutSec
        return (@($r.data[0].embedding).Count -gt 0)
    }
    catch { return $false }
}

function Invoke-RawModel {
    param([string]$Prompt)
    $body = @{ model = $Model; max_tokens = 6000; messages = @(@{ role = 'user'; content = $Prompt }) } | ConvertTo-Json -Depth 6
    $r = Invoke-RestMethod $Proxy -Method Post -Body $body -ContentType 'application/json' `
        -Headers @{ 'x-api-key' = 'local'; 'anthropic-version' = '2023-06-01' } -TimeoutSec 300
    return [string]$r.content[0].text
}

function Invoke-ClaudeCodeSolver {
    # The `claude-code` arm: run Claude Code headless against the SAME local apex
    # model (via the no-think proxy), in the exercise workspace, with file edits
    # allowed. Same model as the `full`/`baseline` arms, so the full-vs-claude-code
    # delta compares the two HARNESSES with the model pinned. Bounded by a timeout.
    param(
        [Parameter(Mandatory = $true)][string]$Workspace,
        [Parameter(Mandatory = $true)][string]$Problem,
        [string]$ModelName = $Model,
        [string]$BaseUrl = 'http://127.0.0.1:11435',
        [string]$AuthToken = 'local',
        # Wall-clock budget for one `claude -p` solve. Defaults to the same
        # per-exercise backstop the other arms get ($EvalTimeout, 900s) so the cc
        # arm is never silently penalised by a tighter default — an earlier 600s
        # default produced 14 cc timeouts that depressed the cc solve rate. The
        # solve loop passes $EvalTimeout explicitly; this default just keeps a
        # bare call fair. --max-turns (40) is the matched turn cap, not the binding
        # limit here (the timeouts were wall-clock, not turn-count).
        [int]$TimeoutSeconds = 900,
        [int]$MaxTurns = 40,
        [string]$ClaudePath = 'claude'
    )
    if (-not (Test-Path -LiteralPath $Workspace)) { throw "Invoke-ClaudeCodeSolver: workspace not found: $Workspace" }
    $exe = (Get-Command -Name $ClaudePath -ErrorAction SilentlyContinue)?.Source
    if (-not $exe) { throw "Invoke-ClaudeCodeSolver: '$ClaudePath' not on PATH; pass -ClaudePath." }

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $exe
    foreach ($a in @('-p', $Problem, '--permission-mode', 'bypassPermissions', '--model', $ModelName, '--max-turns', [string]$MaxTurns)) {
        $psi.ArgumentList.Add([string]$a)
    }
    $psi.WorkingDirectory = $Workspace
    # point Claude Code at the local apex model via the proxy (Anthropic-style)
    $null = $psi.Environment   # materialize the inherited environment before overlaying
    $psi.Environment['ANTHROPIC_BASE_URL'] = $BaseUrl
    $psi.Environment['ANTHROPIC_AUTH_TOKEN'] = $AuthToken
    $psi.Environment['ANTHROPIC_API_KEY'] = $AuthToken
    $psi.Environment['ANTHROPIC_MODEL'] = $ModelName
    $psi.Environment['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] = '1'
    $psi.Environment['DISABLE_TELEMETRY'] = '1'
    $psi.Environment['DISABLE_AUTOUPDATER'] = '1'
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false

    $proc = [System.Diagnostics.Process]::Start($psi)
    $so = $proc.StandardOutput.ReadToEndAsync()
    $null = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
        try { $proc.Kill($true) } catch {}
        throw "Invoke-ClaudeCodeSolver: 'claude -p' timed out after ${TimeoutSeconds}s"
    }
    # claude edits files in-place; we grade the workspace, so its stdout is advisory only
    return $so.GetAwaiter().GetResult()
}

function Write-BaselineFiles {
    # Parse `FILE: <path>` + fenced blocks from a single-shot reply and write each.
    param([string]$Reply, [string]$Workspace, [string[]]$StubFiles)
    $wrote = 0
    $matches = [regex]::Matches($Reply, '(?ms)FILE:\s*(?<path>[^\r\n`]+?)\s*\r?\n```[a-zA-Z0-9+]*\r?\n(?<body>.*?)```')
    foreach ($m in $matches) {
        $rel = $m.Groups['path'].Value.Trim()
        if ($StubFiles -notcontains $rel) { continue }   # only editable stubs
        $dest = Join-Path $Workspace $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        $m.Groups['body'].Value | Set-Content -LiteralPath $dest -Encoding utf8 -NoNewline
        $wrote++
    }
    if ($wrote -eq 0) {
        # fallback: single fenced block -> first stub file
        $one = [regex]::Match($Reply, '(?ms)```[a-zA-Z0-9+]*\r?\n(?<body>.*?)```')
        if ($one.Success -and $StubFiles.Count -ge 1) {
            $one.Groups['body'].Value | Set-Content -LiteralPath (Join-Path $Workspace $StubFiles[0]) -Encoding utf8 -NoNewline
            $wrote = 1
        }
    }
    return $wrote
}

# ---- local LLM server lifecycle (pause frees the GPU; resume restarts) -------
function Test-ServerUp {
    try { return ((Invoke-RestMethod 'http://127.0.0.1:8080/health' -TimeoutSec 3).status -eq 'ok') } catch { return $false }
}
function Start-Server {
    # Resume: bring the tuned apex-i-quality server back (detached; survives this runner).
    if (Test-ServerUp) { return $true }
    Write-Host 'server down — starting llmdefaultserve (resume)...'
    Start-Process pwsh -ArgumentList '-NoProfile', '-Command', ". `"$HOME\.local-llm\LocalLLMProfile.ps1`"; llmdefaultserve" -WindowStyle Hidden
    for ($i = 0; $i -lt 45; $i++) { Start-Sleep -Seconds 8; if (Test-ServerUp) { Write-Host "server up (~$([int](($i+1)*8))s)"; Start-Sleep -Seconds 4; return $true } }
    return $false
}
function Test-ServerGenerates {
    # A REAL generation probe — `/health` can read ok while llama-server is wedged
    # (no tokens produced). This catches the wedge `/health` misses.
    param([int]$TimeoutSec = 45)
    try {
        $b = @{ model = $Model; max_tokens = 8; messages = @(@{ role = 'user'; content = 'ok' }) } | ConvertTo-Json -Depth 5
        $r = Invoke-RestMethod 'http://127.0.0.1:11435/v1/messages' -Method Post -Body $b -ContentType 'application/json' `
            -Headers @{ 'x-api-key' = 'local'; 'anthropic-version' = '2023-06-01' } -TimeoutSec $TimeoutSec
        return (-not [string]::IsNullOrWhiteSpace([string]$r.content[0].text))
    } catch { return $false }
}
function Restart-Server {
    # Recover a wedged server: kill llama-server, relaunch llmdefaultserve (proxy persists).
    Write-Host 'restarting wedged server...'
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
    Start-Process pwsh -ArgumentList '-NoProfile', '-Command', ". `"$HOME\.local-llm\LocalLLMProfile.ps1`"; llmdefaultserve" -WindowStyle Hidden
    for ($i = 0; $i -lt 45; $i++) { Start-Sleep -Seconds 8; if (Test-ServerUp) { Start-Sleep -Seconds 6; return (Test-ServerGenerates -TimeoutSec 60) } }
    return $false
}

# ---- resumable cell ledger ---------------------------------------------------
$ledger = Join-Path $runRoot 'cells.jsonl'
function Get-DoneCells {
    $h = @{}
    if (Test-Path $ledger) {
        foreach ($line in Get-Content $ledger) {
            if (-not $line.Trim()) { continue }
            try { $c = $line | ConvertFrom-Json; $h["$($c.lang)|$($c.ex)|$($c.arm)"] = $c } catch {}
        }
    }
    return $h
}
function Add-Cell { param($Cell) ($Cell | ConvertTo-Json -Compress -Depth 5) | Add-Content -LiteralPath $ledger -Encoding utf8 }
function Get-Rate { param([object[]]$Cells) $n = @($Cells).Count; $p = @($Cells | Where-Object { $_.solved }).Count; return @{ p = $p; n = $n; rate = ($n ? $p / $n : 0) } }
function Show-Standings {
    $cells = @((Get-DoneCells).Values)
    if ($cells.Count -eq 0) { Write-Host 'ledger empty — nothing run yet.'; return }
    $order = 'python', 'go', 'rust', 'cpp', 'javascript', 'java'
    $ls = @($order | Where-Object { $l = $_; $cells | Where-Object { $_.lang -eq $l } })
    $ls += @($cells | ForEach-Object { $_.lang } | Sort-Object -Unique | Where-Object { $_ -notin $order })

    Write-Host ''
    Write-Host ('  Corpus sweep — APEX two-arm (full=localpilot eval  vs  baseline=raw model)')
    Write-Host ('  {0,-11} {1,-15} {2,-15} {3,-7} {4}' -f 'lang', 'full', 'baseline', 'delta', 'exercises')
    Write-Host ('  ' + ('-' * 66))
    # full-corpus denominator: all six languages x 2 arms (not just langs seen so far)
    $targetTotal = (@($order | ForEach-Object { @(Get-ChildItem (Join-Path $corpusRoot "$_/exercises/practice") -Directory -ErrorAction SilentlyContinue).Count }) | Measure-Object -Sum).Sum * 2
    foreach ($l in $ls) {
        $total = @(Get-ChildItem (Join-Path $corpusRoot "$l/exercises/practice") -Directory -ErrorAction SilentlyContinue).Count
        $f = Get-Rate (@($cells | Where-Object { $_.lang -eq $l -and $_.arm -eq 'full' }))
        $b = Get-Rate (@($cells | Where-Object { $_.lang -eq $l -and $_.arm -eq 'baseline' }))
        # an exercise is "done" only when BOTH arms ran (a complete pair), not any-arm
        $bex = @($cells | Where-Object { $_.lang -eq $l -and $_.arm -eq 'baseline' } | ForEach-Object { $_.ex } | Sort-Object -Unique)
        $fex = @($cells | Where-Object { $_.lang -eq $l -and $_.arm -eq 'full' } | ForEach-Object { $_.ex } | Sort-Object -Unique)
        $exDone = @($bex | Where-Object { $_ -in $fex }).Count
        $delta = if ($f.n -and $b.n) { '{0:+0%;-0%; 0%}' -f ($f.rate - $b.rate) } else { '--' }
        $done = ($exDone -ge $total -and $total -gt 0) ? 'DONE' : "$exDone/$total"
        Write-Host ('  {0,-11} {1,-15} {2,-15} {3,-7} {4}' -f $l,
            ('{0}/{1} ({2:P0})' -f $f.p, $f.n, $f.rate), ('{0}/{1} ({2:P0})' -f $b.p, $b.n, $b.rate), $delta, $done)
    }
    Write-Host ('  ' + ('-' * 66))
    $F = Get-Rate (@($cells | Where-Object { $_.arm -eq 'full' }))
    $B = Get-Rate (@($cells | Where-Object { $_.arm -eq 'baseline' }))
    $D = if ($F.n -and $B.n) { $F.rate - $B.rate } else { 0 }
    Write-Host ('  {0,-11} {1,-15} {2,-15} {3:+0%;-0%}' -f 'TOTAL', ('{0}/{1} ({2:P0})' -f $F.p, $F.n, $F.rate), ('{0}/{1} ({2:P0})' -f $B.p, $B.n, $B.rate), $D)
    Write-Host ('  LocalPilot harness delta (full - baseline): {0:+0%;-0%}' -f $D)
    if ($targetTotal -gt 0) { Write-Host ('  progress: {0}/{1} cells ({2:P0})' -f $cells.Count, $targetTotal, ($cells.Count / $targetTotal)) }

    # 3-way harness comparison (model pinned): LocalPilot (full) vs Claude Code
    $ccAll = @($cells | Where-Object { $_.arm -eq 'claude-code' })
    if ($ccAll.Count -gt 0) {
        Write-Host ''
        Write-Host '  Harness comparison — apex pinned: LocalPilot (full) vs Claude Code'
        Write-Host ('  {0,-11} {1,-15} {2,-15} {3}' -f 'lang', 'LocalPilot', 'Claude Code', 'delta (LP-CC)')
        foreach ($l in $ls) {
            $lp = Get-Rate (@($cells | Where-Object { $_.lang -eq $l -and $_.arm -eq 'full' }))
            $cc = Get-Rate (@($cells | Where-Object { $_.lang -eq $l -and $_.arm -eq 'claude-code' }))
            if ($cc.n -eq 0) { continue }
            $d = if ($lp.n) { '{0:+0%;-0%; 0%}' -f ($lp.rate - $cc.rate) } else { '--' }
            Write-Host ('  {0,-11} {1,-15} {2,-15} {3}' -f $l, ('{0}/{1} ({2:P0})' -f $lp.p, $lp.n, $lp.rate), ('{0}/{1} ({2:P0})' -f $cc.p, $cc.n, $cc.rate), $d)
        }
        $LP = Get-Rate (@($cells | Where-Object { $_.arm -eq 'full' }))
        $CC = Get-Rate (@($cells | Where-Object { $_.arm -eq 'claude-code' }))
        Write-Host ('  {0,-11} {1,-15} {2,-15} {3:+0%;-0%}' -f 'TOTAL', ('{0}/{1} ({2:P0})' -f $LP.p, $LP.n, $LP.rate), ('{0}/{1} ({2:P0})' -f $CC.p, $CC.n, $CC.rate), ($LP.n -and $CC.n ? $LP.rate - $CC.rate : 0))
        # Fairness caveat: a cc cell that hit the wall-clock budget is an unsolved
        # cell that may be budget-bound, not capability-bound. Surface the count so
        # the cc rate is not read as a pure capability number.
        $ccTimeouts = @($ccAll | Where-Object { $_.note -match "claude -p' timed out" }).Count
        if ($ccTimeouts -gt 0) { Write-Host ('  caveat: {0} claude-code cell(s) hit the wall-clock budget (counted unsolved) — cc rate is a floor' -f $ccTimeouts) }
    }

    $lastLine = (Get-Content $ledger | Where-Object { $_.Trim() } | Select-Object -Last 1)
    if ($lastLine) { $lc = $lastLine | ConvertFrom-Json; $age = [math]::Round(((Get-Date) - (Get-Item $ledger).LastWriteTime).TotalSeconds)
        Write-Host ('  last: {0}/{1} {2} solved={3}  ({4}s ago)' -f $lc.lang, $lc.ex, $lc.arm, $lc.solved, $age) }
    Write-Host ''
}

# ============================ PHASES =========================================
if ($Phase -eq 'summary') { Show-Standings; return }
if ($Phase -eq 'selftest') {
    foreach ($lang in $langs) {
        $exAll = Get-ChildItem (Join-Path $corpusRoot "$lang/exercises/practice") -Directory | Select-Object -First 1
        $ex = $exAll.Name
        $gws = New-Workspace -Lang $lang -Ex $ex -Kind gold
        $g = Invoke-Grade -Lang $lang -Workspace $gws
        $sws = New-Workspace -Lang $lang -Ex $ex -Kind stub
        $s = Invoke-Grade -Lang $lang -Workspace $sws
        $ok = ($g.passed -and -not $s.passed)
        Write-Host ("[{0}/{1}] gold={2} stub={3} -> grader {4}" -f $lang, $ex, $g.passed, $s.passed, ($ok ? 'OK' : 'BROKEN'))
        if (-not $ok) { Write-Host ("    gold: $($g.tail)`n    stub: $($s.tail)") }
    }
}
elseif ($Phase -eq 'solve') {
    $done = Get-DoneCells
    Write-Host ("resuming: {0} cells already in the ledger" -f $done.Count)
    if (-not (Start-Server)) { throw 'local LLM server is not up and llmdefaultserve did not start it — aborting (ledger is intact; resume to retry).' }
    # Pre-flight the grader: a wedged/stopped Docker engine makes every grade time
    # out, which would burn the whole run as false failures. Catch it before the
    # first cell, not after hours.
    if (-not (Test-DockerHealthy)) { throw 'Docker engine did not run a container (wedged or stopped) — aborting before the run wastes itself on grade timeouts. Fix Docker (start Docker Desktop, or `wsl --shutdown` then start it), then resume; the ledger is intact.' }
    Write-Host 'docker: engine healthy (ran a container).'
    # Warm the shared cargo registry once before the rust grades so the offline
    # rust build (--network=none + CARGO_NET_OFFLINE=true) finds the exercises'
    # declared deps + the common crates an agentic arm may add. Idempotent; only
    # when rust is in the matrix. A warm failure is non-fatal (each affected cell
    # then fails loud with a cargo error) — but it is the difference between
    # honest rust numbers and all-arm offline-dep false fails.
    if ('rust' -in $langs) {
        if (-not (Initialize-CargoCache)) { Write-Host 'cargo cache warm incomplete — continuing; offline rust deps may fail loud per cell.' }
    }
    # The warm arm depends on the CPU embedding server (semantic dedup + rerank over
    # the shared global store). Pre-flight it the same way as docker — prove it
    # returns a vector before the first cell, not after a wasted run. Only the warm
    # arm uses it, so this gate is skipped entirely when 'warm' is not in the matrix
    # (the other arms are unaffected). The embed server is CPU-only, so this adds no
    # GPU VRAM and the chat model stays identical across arms.
    if ('warm' -in $armList) {
        if (-not (Test-EmbedEndpointHealthy -BaseUrl $EmbedBaseUrl)) { throw "Embedding endpoint $EmbedBaseUrl did not return a vector — the warm arm needs the CPU embedding server. Start it with LocalBox ``llmembedserve`` (CPU-only, -ngl 0, no GPU VRAM), then resume; the ledger is intact." }
        Write-Host "embed: endpoint healthy (returned a vector) — $EmbedBaseUrl (CPU, no GPU VRAM)."
    }
    $gradeTimeouts = 0   # consecutive grade-timeout streak -> docker-wedge circuit breaker
    $yielded = $false
    foreach ($lang in $langs) {
        if ($yielded) { break }
        $exs = Get-ChildItem (Join-Path $corpusRoot "$lang/exercises/practice") -Directory | Sort-Object Name
        if ($Limit -gt 0) { $exs = $exs | Select-Object -First $Limit }
        foreach ($exDir in $exs) {
            $ex = $exDir.Name
            # Resume fast: if every arm for this exercise is already in the ledger,
            # skip it entirely — no server probe, no model touch.
            if (@($armList | Where-Object { -not $done.ContainsKey("$lang|$ex|$_") }).Count -eq 0) { continue }
            # Health + wedge check before each exercise (only when there is real work).
            if (-not (Test-ServerUp)) {
                # /health down -> server gone (e.g. you took the GPU) -> yield, don't fight.
                Write-Host 'server gone (health down) — yielding GPU (ledger intact; resume to continue).'
                $yielded = $true; break
            }
            if (-not (Test-ServerGenerates)) {
                # /health ok but no tokens -> wedged. This is the runner's own server, so
                # restart it (vs yield, which is for "you took the GPU").
                Write-Host 'server wedged (health ok, generation stalled) — restarting...'
                if (-not (Restart-Server)) {
                    Write-Host 'still wedged after restart — yielding (ledger intact; resume to continue).'
                    $yielded = $true; break
                }
                Write-Host 'server recovered; continuing.'
            }
            $ex = $exDir.Name
            $src = $exDir.FullName
            $stubFiles = @(& $cfg[$lang].stub $src)
            $instr = if (Test-Path (Join-Path $src '.docs/instructions.md')) { Get-Content (Join-Path $src '.docs/instructions.md') -Raw } else { '' }
            foreach ($arm in $armList) {
                $key = "$lang|$ex|$arm"
                if ($done.ContainsKey($key)) { continue }   # resume: skip completed cell
                $ws = New-Workspace -Lang $lang -Ex $ex -Kind solve
                $solved = $false; $note = ''; $testsRun = $null
                try {
                    if ($arm -eq 'full') {
                        # full agentic harness
                        @"
[provider]
default = "local"
[providers.local]
kind = "openai-compatible"
base_url = "http://localhost:11435/v1"
model = "$Model"
context_window = 32000
"@ | Set-Content (Join-Path $ws '.localpilot.toml') -Encoding utf8
                        # Learning is on by default now, so disable it for this clean-room
                        # measurement arm — the full-vs-raw delta must not read/write memory.
                        New-LocalMindMeasurementConfig | Set-Content (Join-Path $ws '.localmind.toml') -Encoding utf8
                        git -C $ws init -q; git -C $ws add -A 2>$null; git -C $ws -c user.email=x@x -c user.name=x commit -qm init 2>$null
                        $prob = "$instr`n`nImplement the solution in: $($stubFiles -join ', '). Edit only those files; do not modify the tests."
                        $raw = Invoke-LocalPilotSolver -Workspace $ws -Problem $prob -Model $Model -Arm 'full' -Task "$lang-$ex" -LocalPilotPath 'localpilot' -TimeoutSeconds $EvalTimeout
                        # (scorecard parsed only for diagnostics; pass/fail is the container grade)
                    }
                    elseif ($arm -eq 'claude-code') {
                        # claude-code arm: Claude Code headless driving the SAME apex model
                        # (via the proxy) — full vs claude-code = LocalPilot harness vs
                        # Claude Code harness, model pinned.
                        git -C $ws init -q; git -C $ws add -A 2>$null; git -C $ws -c user.email=x@x -c user.name=x commit -qm init 2>$null
                        $prob = "$instr`n`nImplement the solution in: $($stubFiles -join ', '). Edit only those files; do not modify the tests."
                        $null = Invoke-ClaudeCodeSolver -Workspace $ws -Problem $prob -TimeoutSeconds $EvalTimeout
                    }
                    elseif ($arm -in @('fair', 'verify', 'warm')) {
                        # HarnessConvergence arms: the exact config is emitted by the
                        # LocalBench arm-config functions (recorded, Pester-pinned), so
                        # the arm is a config, not a label. fair = rails matched to CC +
                        # verify on; verify = full+verify only; warm = fair + persistent
                        # global learning shared across exercises.
                        New-LocalPilotArmConfig -Arm $arm -Model $Model | Set-Content (Join-Path $ws '.localpilot.toml') -Encoding utf8
                        if ($arm -eq 'warm') {
                            # The global store lives OUTSIDE the per-exercise workspace so
                            # lessons accumulate across all 225 exercises (the workspace is
                            # wiped each cell). Closeout writeback for the live run is wired
                            # in 06.4 (DEFERRED, D008).
                            $warmGlobal = Join-Path $runRoot 'warm-global\memory'
                            New-Item -ItemType Directory -Force -Path $warmGlobal | Out-Null
                            New-LocalMindWarmConfig -GlobalMemoryRoot $warmGlobal -Model $Model | Set-Content (Join-Path $ws '.localmind.toml') -Encoding utf8
                        }
                        else {
                            # fair / verify are clean-room measurement arms — disable the
                            # now-default-on learning so they read/write no accumulated memory.
                            New-LocalMindMeasurementConfig | Set-Content (Join-Path $ws '.localmind.toml') -Encoding utf8
                        }
                        git -C $ws init -q; git -C $ws add -A 2>$null; git -C $ws -c user.email=x@x -c user.name=x commit -qm init 2>$null
                        $prob = "$instr`n`nImplement the solution in: $($stubFiles -join ', '). Edit only those files; do not modify the tests."
                        # Only the warm arm closes out into LocalMind (`--learn`) so lessons
                        # accumulate across exercises; fair/verify stay clean-room.
                        $raw = Invoke-LocalPilotSolver -Workspace $ws -Problem $prob -Model $Model -Arm $arm -Task "$lang-$ex" -LocalPilotPath 'localpilot' -TimeoutSeconds $EvalTimeout -Learn:($arm -eq 'warm')
                    }
                    else {
                        # baseline: raw single-shot model, no harness/tools
                        $stubContent = ($stubFiles | ForEach-Object { "FILE: $_`n```` `n" + (Get-Content (Join-Path $ws $_) -Raw) + "`n```` " }) -join "`n"
                        $prompt = "You are solving a coding exercise. Instructions:`n$instr`n`nCurrent file(s) to implement:`n$stubContent`n`n" +
                        "Reply with the COMPLETE final content of each file to edit. For each file output a line `FILE: <path>` then a fenced code block with the full file content. Edit only: $($stubFiles -join ', '). Do not include tests or commentary."
                        $reply = Invoke-RawModel -Prompt $prompt
                        $w = Write-BaselineFiles -Reply $reply -Workspace $ws -StubFiles $stubFiles
                        if ($w -eq 0) { $note = 'no parseable file in reply' }
                    }
                    $grade = Invoke-Grade -Lang $lang -Workspace $ws
                    $solved = $grade.passed
                    $testsRun = $grade.tests_run
                    if (-not $note) { $note = $grade.tail }
                    if (($grade.tests_run -le 0) -and -not $grade.timedout) { $note = "0 tests ran (not solved) | $note" }
                }
                catch { $note = "ERR: $($_.Exception.Message)" }
                finally { Remove-Item $ws -Recurse -Force -ErrorAction SilentlyContinue }
                $cell = [ordered]@{ lang = $lang; ex = $ex; arm = $arm; solved = $solved; tests_run = $testsRun; note = $note }
                Add-Cell $cell                 # persist immediately -> pause/resume safe
                $done[$key] = $cell
                Write-Host ("[{0}/{1}] {2,-8} solved={3} {4}" -f $lang, $ex, $arm, $solved, ($solved ? '' : "($note)".Substring(0, [Math]::Min(60, "($note)".Length))))
                # Docker-wedge circuit breaker: a graded test that times out means a
                # hung container or wedged engine, not a wrong solution. Three in a
                # row = the engine is gone; stop now (ledger intact) instead of marking
                # every remaining cell a false failure over hours. A real pass/fail
                # resets the streak.
                if ($cell.note -match 'grade timed out') { $gradeTimeouts++ } else { $gradeTimeouts = 0 }
                if ($gradeTimeouts -ge 3) {
                    Write-Host 'DOCKER WEDGED: 3 grade timeouts in a row — yielding (ledger intact; fix Docker, then resume).'
                    $yielded = $true; break
                }
            }
            if ($yielded) { break }
        }
        Write-Host "--- standings after $lang ---"; Show-Standings
    }
    Write-Host "`n===================== ARM SUMMARY ====================="
    Show-Standings
    Write-Host "ledger: $ledger"
    if ($yielded) { Write-Host 'STATUS: yielded (server taken) — resume to continue.' }
    else { Write-Host 'STATUS: pass complete for the requested languages/arms.' }
}
