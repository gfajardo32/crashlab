# ============================================================
#  CrashLab harness  v3
# ============================================================

param(
    [string]$VMName   = "crashlab-01",
    [int]   $Cycles   = 10,
    [string]$Snapshot = "clean"
)

$script:VBM = "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
$GUSER   = "CRASHLAB-GOLDEN"
$GPASS   = "password"
$OutRoot = "C:\CrashLab\runs"
$LogFile = "C:\CrashLab\harness.log"

# argv is ARGUMENTS ONLY. VBoxManage does not want the program name repeated.
$Faults = @(
    @{ name = "mem-slow-50mb"; proc = "Testlimit64"
       exe  = "C:\CrashData\sysinternals\Testlimit64.exe"
       argv = @("-accepteula","-d","50","-e","1") },

    @{ name = "handles"; proc = "Testlimit64"
       exe  = "C:\CrashData\sysinternals\Testlimit64.exe"
       argv = @("-accepteula","-h","-u") }
)

$BaselineMinMin  = 5
$BaselineMaxMin  = 11
$DeathThresholdS = 60
$MaxRunMinutes   = 45
$BootTimeoutS    = 300
$MinFreeGB       = 20

function Invoke-VBox {
    param([string[]]$VArgs, [int]$TimeoutSec = 120)
    $so = [System.IO.Path]::GetTempFileName()
    $se = [System.IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath $script:VBM -ArgumentList $VArgs -NoNewWindow -PassThru `
                 -RedirectStandardOutput $so -RedirectStandardError $se
        if (-not $p.WaitForExit($TimeoutSec * 1000)) {
            try { $p.Kill() } catch { }
            return @{ ok=$false; timedout=$true; exit=$null; out=""; err="TIMEOUT after ${TimeoutSec}s" }
        }
        $p.WaitForExit()
        $out = if (Test-Path $so) { (Get-Content $so -Raw) } else { "" }
        $err = if (Test-Path $se) { (Get-Content $se -Raw) } else { "" }
        $code = $null; try { $code = $p.ExitCode } catch { }
        $failed = ("$out`n$err" -match 'VBoxManage\.exe:\s*error')
        if ($null -ne $code -and $code -ne 0) { $failed = $true }
        return @{ ok=(-not $failed); timedout=$false; exit=$code; out=$out; err=$err }
    }
    finally { Remove-Item $so,$se -Force -ErrorAction SilentlyContinue }
}

function Get-Heartbeat {
    param([string]$vm)
    for ($t=0; $t -lt 4; $t++) {
        $r = Invoke-VBox @("guestproperty","get",$vm,"/CrashLab/Heartbeat") -TimeoutSec 30
        if ($r.out -match 'Value:\s*(\d+)') { return [long]$Matches[1] }
        Start-Sleep -Milliseconds 800
    }
    return $null
}
function Wait-ForHeartbeat {
    param([string]$vm, [int]$timeoutSec)
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    $last = $null
    while ((Get-Date) -lt $deadline) {
        $hb = Get-Heartbeat $vm
        if ($null -ne $hb) {
            if ($null -ne $last -and $hb -gt $last) { return $true }
            $last = $hb
        }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Test-FaultRunning {
    param([string]$vm, [string]$procName)
    $r = Invoke-VBox @("guestcontrol",$vm,"--username",$GUSER,"--password",$GPASS,
                       "run","--exe","C:\Windows\System32\tasklist.exe","--wait-stdout","--",
                       "/fi","imagename eq $procName.exe") -TimeoutSec 60
    return ($r.out -match $procName)
}

function Sync-Git {
    param([string]$repo, [string]$msg)
    try {
        Start-Process -FilePath "git" -ArgumentList @("-C",$repo,"add","-A") -NoNewWindow -Wait
        Start-Process -FilePath "git" -ArgumentList @("-C",$repo,"commit","-m",$msg) -NoNewWindow -Wait
        Start-Process -FilePath "git" -ArgumentList @("-C",$repo,"push") -NoNewWindow -Wait
    } catch { }
}

function Say { param($msg, $color = "Gray") Write-Host $msg -ForegroundColor $color }

New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null
Start-Transcript -Path $LogFile -Append | Out-Null

$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
Say "free disk: $freeGB GB" "DarkGray"
if ($freeGB -lt $MinFreeGB) { Say "ABORT: less than $MinFreeGB GB free" "Red"; Stop-Transcript | Out-Null; return }

for ($i = 1; $i -le $Cycles; $i++) {
  try {
    $fault  = $Faults[(Get-Random -Maximum $Faults.Count)]
    $runId  = "{0}_{1}_{2}" -f $VMName, (Get-Date -Format "yyyyMMdd_HHmmss"), $fault.name
    $runDir = Join-Path $OutRoot $runId
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    Say ""
    Say "=== cycle $i of $Cycles : $runId ===" "Cyan"
    $status = "unknown"

    Invoke-VBox @("controlvm",$VMName,"poweroff") -TimeoutSec 60 | Out-Null
    for ($w=0; $w -lt 20; $w++) {
        $st = Invoke-VBox @("showvminfo",$VMName,"--machinereadable") -TimeoutSec 30
        if ($st.out -match 'VMState="(poweroff|aborted|saved)"') { break }
        Start-Sleep -Seconds 2
    }

    $r = Invoke-VBox @("snapshot",$VMName,"restore",$Snapshot) -TimeoutSec 180
    if (-not $r.ok) { Say "  snapshot restore FAILED: $($r.err.Trim())" "Red"; $status = "restore_failed"; throw "restore failed" }
    Start-Sleep -Seconds 3

    # start, retrying; if the session is wedged, clear stale VBox processes
    $started = $false
    for ($try=1; $try -le 6; $try++) {
        $r = Invoke-VBox @("startvm",$VMName,"--type","headless") -TimeoutSec 120
        if ($r.ok) { $started = $true; break }
        if ($try -ge 2) {
            Get-Process VBoxHeadless,VirtualBoxVM,VBoxSVC,VBoxSDS -ErrorAction SilentlyContinue |
                Stop-Process -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 8
        } else {
            Start-Sleep -Seconds 5
        }
    }
    if (-not $started) {
        Say "  startvm FAILED: $($r.err.Trim())" "Red"
        $status = "start_failed"
        throw "start failed"
    }

    if (-not (Wait-ForHeartbeat $VMName $BootTimeoutS)) { Say "  boot timeout" "Red"; $status = "boot_timeout"; throw "boot timeout" }
    $bootTime = Get-Date
    Say ("  booted {0}" -f $bootTime.ToString("HH:mm:ss"))

    $baseline = Get-Random -Minimum $BaselineMinMin -Maximum $BaselineMaxMin
    Say "  baseline $baseline min"
    Start-Sleep -Seconds ($baseline * 60)

    $injectTime = Get-Date
    $injectHb   = Get-Heartbeat $VMName
    Say ("  inject {0} at {1}" -f $fault.name, $injectTime.ToString("HH:mm:ss"))

    $ctlArgs = @("guestcontrol",$VMName,"--username",$GUSER,"--password",$GPASS,
                 "run","--exe",$fault.exe,"--wait-stdout","--wait-stderr","--") + $fault.argv
    Start-Process -FilePath $script:VBM -ArgumentList $ctlArgs -NoNewWindow `
        -RedirectStandardOutput (Join-Path $runDir "inject_stdout.txt") `
        -RedirectStandardError  (Join-Path $runDir "inject_stderr.txt") | Out-Null

    # --- VERIFY the fault launched, by reading its own stdout (no guest call) ---
    Start-Sleep -Seconds 25
    $ilog = ""
    try { $ilog = Get-Content (Join-Path $runDir "inject_stdout.txt") -Raw -ErrorAction SilentlyContinue } catch { }
    if ($ilog -notmatch 'Creating|Leaking|Leaked|Reserving|Allocating') {
        Say "  LAUNCH FAILED, see inject_stdout.txt" "Red"
        $status = "launch_failed"
        throw "launch failed"
    }
    Say "  fault confirmed running" "DarkGray"

    $lastHb     = $injectHb
    $lastChange = Get-Date
    $deadline   = (Get-Date).AddMinutes($MaxRunMinutes)
    $died       = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        $hb = Get-Heartbeat $VMName
        if ($null -ne $hb -and $hb -ne $lastHb) { $lastHb = $hb; $lastChange = Get-Date }
        elseif (((Get-Date) - $lastChange).TotalSeconds -ge $DeathThresholdS) { $died = $true; break }
    }

    $ttf = if ($died -and $lastHb -and $injectHb) { $lastHb - $injectHb } else { $null }
    if ($died) { Say ("  DIED  ttf={0}s" -f $ttf) "Yellow"; $status = "died" }
    else       { Say "  SURVIVED (hit max run time)" "DarkGray"; $status = "survived" }

    Invoke-VBox @("controlvm",$VMName,"screenshotpng",(Join-Path $runDir "death.png")) -TimeoutSec 60 | Out-Null
    Invoke-VBox @("controlvm",$VMName,"reset") -TimeoutSec 60 | Out-Null

    if (Wait-ForHeartbeat $VMName $BootTimeoutS) {
        $r = Invoke-VBox @("guestcontrol",$VMName,"--username",$GUSER,"--password",$GPASS,
                           "run","--exe","C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                           "--wait-stdout","--",
                           "-NoProfile","-ExecutionPolicy","Bypass",
                           "-File","C:\CrashData\collect-logs.ps1") -TimeoutSec 300
        if (-not $r.ok) { Say "  collector error: $($r.err.Trim())" "Red" }
        $r = Invoke-VBox @("guestcontrol",$VMName,"--username",$GUSER,"--password",$GPASS,
                           "copyfrom","--target-directory",$runDir,
                           "C:\CrashData\exports","--recursive") -TimeoutSec 300
        if ($r.ok) { Say "  harvested" "Green" } else { Say "  copyfrom error: $($r.err.Trim())" "Red" }
    } else { Say "  harvest boot failed" "Red" }
  }
  catch { Say "  cycle aborted: $($_.Exception.Message)" "Red" }
  finally {
    try {
        [pscustomobject]@{
            run_id            = $runId
            vm                = $VMName
            fault             = $fault.name
            fault_cmd         = ($fault.exe + " " + ($fault.argv -join " "))
            status            = $status
            baseline_minutes  = $baseline
            boot_time_local   = if ($bootTime)   { $bootTime.ToString("o") }   else { $null }
            inject_time_local = if ($injectTime) { $injectTime.ToString("o") } else { $null }
            inject_heartbeat  = $injectHb
            died              = $died
            last_heartbeat    = $lastHb
            death_time_utc    = if ($lastHb) { [DateTimeOffset]::FromUnixTimeSeconds($lastHb).UtcDateTime.ToString("o") } else { $null }
            ttf_seconds       = $ttf
        } | ConvertTo-Json | Set-Content (Join-Path $runDir "run.json") -Encoding UTF8
    } catch { }
    Sync-Git $OutRoot ("run {0} {1} ttf={2}" -f $runId, $status, $ttf)
    Invoke-VBox @("controlvm",$VMName,"poweroff") -TimeoutSec 60 | Out-Null
    Start-Sleep -Seconds 5
    Get-Process VBoxManage -ErrorAction SilentlyContinue |
        Where-Object { $_.StartTime -lt (Get-Date).AddMinutes(-2) } |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Remove-Variable bootTime,injectTime,injectHb,lastHb,ttf,died,baseline -ErrorAction SilentlyContinue
  }
}

Say ""
Say "done. runs under $OutRoot, log at $LogFile" "Green"
Stop-Transcript | Out-Null