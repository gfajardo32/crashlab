#!/usr/bin/env python3
"""
crashloop.py - bare-metal crash harness driver (runs on the Mac).

Drives the Windows laptop over SSH, one cycle at a time:
  up -> baseline (normal operation) -> inject memory fault ->
  watchdog bugchecks + auto-reboots -> harvest event logs ->
  write run.json -> (git commit) -> repeat.

Each run folder lands in crashlab-data in the exact shape build_dataset reads.
Usage:  python3 crashloop.py [cycles] [--surface-data] [--healthy-control]
"""
import argparse
import base64
import csv
import hashlib
from pathlib import Path
import re
import subprocess, time, json, random, os, sys, datetime
import tempfile
import threading
import uuid

HOST      = "user@100.96.227.33"            # laptop over Tailscale
RUNS_DIR  = os.path.expanduser("~/4700Project/crashlab-data")
MACHINE   = "DESKTOP-DN6KTF8"
TOOL      = r"C:\CrashLab\tools\Testlimit64.exe"
FAULT_ARG = "-accepteula -d 40 -e 1"        # ~40 MB/s leak
FAULT     = "mem-40mb"
BASELINE_MIN, BASELINE_MAX = 4, 8           # minutes of normal operation
FIRST_BASELINE_MIN = 8
HEALTHY_CONTROL_MIN = 12
MEMORY_SAMPLE_SECONDS = 5
MEMORY_COLUMNS = ("timestamp_utc", "available_bytes", "committed_bytes", "commit_limit_bytes", "pages_per_sec")
PROBE_COLUMNS = ("probe_timestamp_utc", "probe_status", "probe_latency_ms")
PROBE_COST_COLUMNS = ("probe_peak_commit_bytes", "probe_cpu_ms")
RESPONSIVENESS_FILES = {
    "memory": "memory.csv",
    "powershell": "probe_powershell.csv",
    "cmd": "probe_cmd.csv",
}
RESPONSIVENESS_TIMEOUT_MS = 4000
RESPONSIVENESS_FAILURE_MS = 1365
RAMP_LIMIT_MINUTES = {5: 25, 10: 15, 15: 12}
STEPPED_PATTERNS = {"A": (75, 60), "B": (60, 360)}
STEPPED_LIMIT_MIN = 25
PROTOCOL_HEALTHY_MIN = 30
PROTOCOL_SETTLE_MIN = 5
PLATEAU_CHUNK_MIB = 10
PLATEAU_MARGIN_CHUNKS = 3
PLATEAU_HOLD_LIMIT_MIN = 22
HIGH_PLATEAU_PERCENT = 96
HIGH_PLATEAU_GUARD_PERCENT = 98
PLATEAU_TOPUP_MAX_CHUNKS = 25
PLATEAU_STABILITY_MIB = 64
MAX_RUN_MIN   = 20                          # give up on a cycle after this
BOOT_WAIT_MIN = 7                           # max wait for the machine to return
GIT_PUSH      = True                        # commit+push each run to crashlab-data

SSH = ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
       "-o", "BatchMode=yes", HOST]


def ps(cmd, timeout=60):
    """Return success and stdout, or failure and the command's diagnostic."""
    script = (
        "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; "
        "trap { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }; "
        + cmd
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        r = subprocess.run(SSH + ["powershell", "-NoProfile", "-EncodedCommand", encoded],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    if r.returncode:
        return False, r.stderr.strip() or r.stdout.strip() or f"SSH exited with status {r.returncode}"
    return True, r.stdout.strip()


def reachable():
    ok, out = ps("echo ok", timeout=15)
    return ok and "ok" in out


def boot_time(*, required=False, timeout=60):
    ok, out = ps("(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('o')", timeout=timeout)
    if ok and out:
        return out
    if required:
        detail = out or "PowerShell returned no boot time"
        raise SystemExit(f"Surface boot check failed: {detail}")
    return None


def read_last_death():
    ok, out = ps(r"Get-Content C:\CrashLab\last_death.txt")
    return out.strip() if ok else None


def wait_until(predicate, timeout_s, poll_s=15):
    end = time.time() + timeout_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def git_sync(msg, runid=None):
    if runid is not None:
        for args in (["add", "--", runid], ["commit", "-m", msg, "--", runid], ["push"]):
            result = subprocess.run(["git", "-C", RUNS_DIR] + args, capture_output=True, text=True)
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
                raise RuntimeError(f"git {args[0]} failed for {runid}: {detail}")
        print(f"  pushed -> {runid}", flush=True)
        return
    for args in (["add", "-A"], ["commit", "-m", msg], ["push"]):
        subprocess.run(["git", "-C", RUNS_DIR] + args,
                       capture_output=True, text=True)


def start_workload(task_name):
    processes = (
        f"Get-CimInstance Win32_Process | Where-Object {{ $_.ExecutablePath -eq '{TOOL}' }}"
    )
    ok, result = ps(
        f"if ({processes}) {{ throw 'A workload process is already running' }}; "
        f"$action = New-ScheduledTaskAction -Execute '{TOOL}' -Argument '{FAULT_ARG}'; "
        "$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name; "
        "$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U; "
        "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries "
        f"-ExecutionTimeLimit (New-TimeSpan -Minutes {MAX_RUN_MIN + 2}); "
        f"Register-ScheduledTask -TaskName '{task_name}' -Action $action "
        "-Principal $principal -Settings $settings | Out-Null; "
        f"Start-ScheduledTask -TaskName '{task_name}'"
    )
    if not ok:
        raise SystemExit(f"Surface scheduled launch failed: {result}")

    ok, result = ps(
        f"for ($i = 0; $i -lt 10; $i++) {{ "
        f"$task = Get-ScheduledTask -TaskName '{task_name}'; "
        f"$process = {processes} | Select-Object -First 1; "
        "if ($task.State -eq 'Running' -and $process) { $process.ProcessId; exit 0 }; "
        "Start-Sleep -Milliseconds 500 }; "
        f"$result = (Get-ScheduledTaskInfo -TaskName '{task_name}').LastTaskResult; "
        'throw "Testlimit did not stay running after SSH disconnected (task result $result)"'
    )
    if not ok or not result.isdigit():
        raise SystemExit(f"Surface Testlimit verification failed: {result or 'no process ID'}")
    return result


def remove_workload_task(task_name):
    ok, result = ps(
        f"$task = Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue; "
        "if ($task) { "
        "if ($task.State -eq 'Running') { $task | Stop-ScheduledTask }; "
        "$task | Unregister-ScheduledTask -Confirm:$false }; 'removed'",
        timeout=20,
    )
    if not ok or result != "removed":
        print(f"  Surface task cleanup failed for {task_name}: {result or 'no confirmation'}", flush=True)
        return False
    return True


def start_memory_sampling(task_name, remote_csv):
    sampler = rf"""
$ErrorActionPreference = 'Stop'
$stream = [IO.FileStream]::new('{remote_csv}', 'CreateNew', 'Write', 'Read')
$writer = [IO.StreamWriter]::new($stream)
$probe = [Diagnostics.ProcessStartInfo]::new()
$probe.FileName = Join-Path $env:SystemRoot 'System32\cmd.exe'
$probe.Arguments = '/d /c exit 0'
$probe.UseShellExecute = $false
$probe.CreateNoWindow = $true
try {{
    $writer.WriteLine('{",".join(MEMORY_COLUMNS + PROBE_COLUMNS)}')
    while ($true) {{
        $tick = [Diagnostics.Stopwatch]::StartNew()
        $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
        $values = @($memory.AvailableBytes, $memory.CommittedBytes, $memory.CommitLimit, $memory.PagesPersec)
        if ($values.Count -ne 4 -or $values -contains $null) {{ throw 'Missing memory counters' }}
        $row = @([DateTime]::UtcNow.ToString('o')) + $values
        $probeTime = [DateTime]::UtcNow.ToString('o')
        $latency = [Diagnostics.Stopwatch]::StartNew()
        $child = $null
        $status = 'spawn_error'
        try {{
            $child = [Diagnostics.Process]::Start($probe)
            if ($null -ne $child) {{
                $status = 'timeout'
                if ($child.WaitForExit(2000)) {{
                    if ($child.ExitCode -eq 0) {{ $status = 'ok' }} else {{ $status = 'exit_nonzero' }}
                }}
            }}
        }} catch {{
            if ($null -ne $child) {{ $status = 'probe_error' }}
        }} finally {{
            $elapsed = $latency.ElapsedMilliseconds
            if ($null -ne $child) {{
                if ($status -in @('timeout', 'probe_error')) {{
                    try {{ $child.Kill() }} catch {{ }}
                }}
                $child.Dispose()
            }}
        }}
        $row += @($probeTime, $status, $elapsed)
        $writer.WriteLine($row -join ',')
        $writer.Flush()
        $stream.Flush($true)
        $sleep = [Math]::Max(0, {MEMORY_SAMPLE_SECONDS * 1000} - $tick.ElapsedMilliseconds)
        Start-Sleep -Milliseconds $sleep
    }}
}} finally {{
    $writer.Dispose()
}}
"""
    encoded = base64.b64encode(sampler.encode("utf-16-le")).decode("ascii")
    ok, result = ps(
        "$exe = Join-Path $env:SystemRoot 'System32\\WindowsPowerShell\\v1.0\\powershell.exe'; "
        f"$action = New-ScheduledTaskAction -Execute $exe -Argument '-NoProfile -EncodedCommand {encoded}'; "
        "$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name; "
        "$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U; "
        "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries "
        f"-ExecutionTimeLimit (New-TimeSpan -Minutes {max(FIRST_BASELINE_MIN, BASELINE_MAX) + MAX_RUN_MIN + 3}); "
        f"Register-ScheduledTask -TaskName '{task_name}' -Action $action "
        "-Principal $principal -Settings $settings | Out-Null; "
        f"Start-ScheduledTask -TaskName '{task_name}'"
    )
    if not ok:
        raise RuntimeError(f"Surface memory sampler could not start: {result}")

    ok, result = ps(
        "for ($i = 0; $i -lt 20; $i++) { "
        f"$sample = @(Get-Content -LiteralPath '{remote_csv}' -TotalCount 2 -ErrorAction SilentlyContinue); "
        f"if ($sample.Count -eq 2 -and (Get-ScheduledTask -TaskName '{task_name}').State -eq 'Running') "
        "{ ($sample[1] -split ',')[0]; exit 0 }; "
        "Start-Sleep -Seconds 1 }; "
        f"$result = (Get-ScheduledTaskInfo -TaskName '{task_name}').LastTaskResult; "
        'throw "No live memory samples (task result $result)"'
    )
    if not ok:
        raise RuntimeError(f"Surface memory sampling verification failed: {result}")
    parse_utc(result)
    return result


def parse_utc(value):
    normalized = re.sub(r"(\.\d{6})\d+(?=Z$|[+-]\d{2}:\d{2}$)", r"\1", value)
    timestamp = datetime.datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    if timestamp.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"Expected a UTC timestamp, got {value!r}")
    return timestamp


def surface_utc_now():
    ok, result = ps("[DateTime]::UtcNow.ToString('o')")
    if not ok:
        raise RuntimeError(f"Could not read Surface clock: {result}")
    parse_utc(result)
    return result


def responsiveness_grid_bounds(clock, minutes):
    tick = datetime.timedelta(seconds=MEMORY_SAMPLE_SECONDS)
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    start = epoch + ((parse_utc(clock) - epoch) // tick + 1) * tick
    return start, start + datetime.timedelta(minutes=minutes)


def responsiveness_sampler_script(kind, remote_csv, timeout_ms):
    """Build one Surface script whose slots use the shared five-second UTC grid."""
    if kind == "memory":
        header = MEMORY_COLUMNS
        initialize = """
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class CrashLabCounters {
    [StructLayout(LayoutKind.Sequential)]
    struct PerformanceInfo {
        public uint Size;
        public UIntPtr CommitTotal, CommitLimit, CommitPeak, PhysicalTotal, PhysicalAvailable;
        public UIntPtr SystemCache, KernelTotal, KernelPaged, KernelNonpaged, PageSize;
        public uint HandleCount, ProcessCount, ThreadCount;
    }
    [DllImport("psapi.dll")]
    static extern bool GetPerformanceInfo(out PerformanceInfo info, uint size);
    public static string Read() {
        PerformanceInfo info;
        if (!GetPerformanceInfo(out info, (uint) Marshal.SizeOf(typeof(PerformanceInfo)))) return ",,,";
        var size = info.PageSize.ToUInt64();
        return String.Format(System.Globalization.CultureInfo.InvariantCulture, "{0},{1},{2},",
            info.PhysicalAvailable.ToUInt64() * size, info.CommitTotal.ToUInt64() * size,
            info.CommitLimit.ToUInt64() * size);
    }
}
'@
"""
        sample = """
        $counters = ',,,'
        try { $counters = [CrashLabCounters]::Read() } catch { }
        $row = $stamp + ',' + $counters
"""
    elif kind in ("powershell", "cmd"):
        header = PROBE_COLUMNS + PROBE_COST_COLUMNS
        executable = r"System32\WindowsPowerShell\v1.0\powershell.exe" if kind == "powershell" else r"System32\cmd.exe"
        arguments = "-NoLogo -NoProfile -NonInteractive -Command exit" if kind == "powershell" else "/d /c exit 0"
        initialize = r"""
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class CrashLabProcessCost {
    [StructLayout(LayoutKind.Sequential)]
    struct MemoryCounters {
        public uint Size, PageFaultCount;
        public UIntPtr PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage;
        public UIntPtr QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage;
    }
    [DllImport("psapi.dll")]
    static extern bool GetProcessMemoryInfo(IntPtr handle, out MemoryCounters counters, uint size);
    public static string PeakCommit(IntPtr handle) {
        MemoryCounters counters;
        return GetProcessMemoryInfo(handle, out counters, (uint) Marshal.SizeOf(typeof(MemoryCounters)))
            ? counters.PeakPagefileUsage.ToUInt64().ToString(System.Globalization.CultureInfo.InvariantCulture)
            : "";
    }
}
'@
$probe = [Diagnostics.ProcessStartInfo]::new()
$probe.UseShellExecute = $false
$probe.CreateNoWindow = $true
""" + f"$probe.FileName = Join-Path $env:SystemRoot '{executable}'\n$probe.Arguments = '{arguments}'\n"
        sample = f"""
        $latency = [Diagnostics.Stopwatch]::StartNew()
        $child = $null
        $status = 'spawn_error'
        $peakCommit = ''
        $cpu = ''
        try {{
            $child = [Diagnostics.Process]::Start($probe)
            if ($null -ne $child) {{
                $status = 'timeout'
                $remaining = {timeout_ms} - $latency.ElapsedMilliseconds
                if ($remaining -gt 0 -and $child.WaitForExit([int]$remaining)) {{
                    if ($child.ExitCode -eq 0) {{ $status = 'ok' }} else {{ $status = 'exit_nonzero' }}
                }}
            }}
        }} catch {{
            if ($null -ne $child) {{ $status = 'probe_error' }}
        }} finally {{
            $elapsed = $latency.ElapsedMilliseconds
            if ($status -eq 'timeout' -and $elapsed -lt {timeout_ms}) {{ $status = 'probe_error' }}
            if ($status -eq 'ok' -and $elapsed -ge {timeout_ms}) {{ $status = 'timeout' }}
            if ($status -eq 'timeout') {{ $elapsed = {timeout_ms} }}
            if ($null -ne $child) {{
                if ($status -in @('timeout', 'probe_error')) {{
                    try {{ $child.Kill() }} catch {{ }}
                }}
                try {{ $peakCommit = [CrashLabProcessCost]::PeakCommit($child.Handle) }} catch {{ }}
                try {{ $cpu = [Math]::Round($child.TotalProcessorTime.TotalMilliseconds).ToString() }} catch {{ }}
                $child.Dispose()
            }}
        }}
        $row = @($stamp, $status, $elapsed, $peakCommit, $cpu) -join ','
"""
    else:
        raise ValueError(f"Unknown responsiveness stream {kind!r}")

    return "$ErrorActionPreference = 'Stop'\n" + initialize + f"""
$stream = [IO.FileStream]::new('{remote_csv}', 'CreateNew', 'Write', 'Read')
$writer = [IO.StreamWriter]::new($stream)
$period = {MEMORY_SAMPLE_SECONDS * 1000}
try {{
    $writer.WriteLine('{','.join(header)}')
    while ($true) {{
        $now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $wait = $period - ($now % $period)
        Start-Sleep -Milliseconds $wait
        $observed = [DateTimeOffset]::UtcNow
        if ($observed.ToUnixTimeMilliseconds() - $now - $wait -ge ($period / 2)) {{ continue }}
        $stamp = $observed.UtcDateTime.ToString('o')
{sample}
        $writer.WriteLine($row)
        $writer.Flush()
        $stream.Flush($true)
    }}
}} finally {{
    $writer.Dispose()
}}
"""


def copy_remote(source, destination, *, checked=False):
    command = ["scp", "-r"]
    if checked:
        command.extend(SSH[1:-1])
    result = subprocess.run(command + [f"{HOST}:{source}", destination],
                            capture_output=True, text=True, timeout=300)
    if checked and result.returncode:
        raise RuntimeError(f"Could not copy {source}: {result.stderr.strip() or result.stdout.strip()}")


def collect_event_exports(destination, *, checked=False):
    ps(r"Remove-Item C:\CrashLab\exports\* -Force -ErrorAction SilentlyContinue")
    ok, result = ps(r"powershell -ExecutionPolicy Bypass -File C:\CrashLab\collect-logs.ps1", timeout=300)
    if checked and not ok:
        raise RuntimeError(f"Surface event export failed: {result}")

    copy_remote("C:/CrashLab/exports", destination, checked=checked)
    exports = os.path.join(destination, "exports")
    if os.path.isdir(exports):
        for filename in os.listdir(exports):
            os.replace(os.path.join(exports, filename), os.path.join(destination, filename))
        os.rmdir(exports)
    if checked and not any("System" in name and name.endswith(".csv") for name in os.listdir(destination)):
        raise RuntimeError("Surface event export did not produce a System CSV")


def write_run_metadata(directory, metadata):
    with open(os.path.join(directory, "run.json"), "w") as stream:
        json.dump(metadata, stream, indent=2)


def launch_responsiveness_sampler(kind, task, remote_script, remote_csv, timeout_ms, *, limit_minutes=40):
    launch_responsiveness_script(
        task, remote_script, responsiveness_sampler_script(kind, remote_csv, timeout_ms), limit_minutes,
    )


def launch_responsiveness_script(task, remote_script, script, limit_minutes):
    upload_path = remote_script.replace("\\", "/")
    with tempfile.TemporaryDirectory(prefix="crashlab-sampler-") as directory:
        source = os.path.join(directory, "sampler.ps1")
        with open(source, "w", encoding="ascii") as stream:
            stream.write(script)
        result = subprocess.run(
            ["scp", *SSH[1:-1], source, f"{HOST}:{upload_path}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode:
            raise RuntimeError(f"Surface sampler upload failed: {result.stderr.strip()}")
    ok, result = ps(
        "$exe = Join-Path $env:SystemRoot 'System32\\WindowsPowerShell\\v1.0\\powershell.exe'; "
        f"$action = New-ScheduledTaskAction -Execute $exe -Argument '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File {remote_script}'; "
        "$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name; "
        "$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U; "
        "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        f"-DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes {limit_minutes}); "
        f"Register-ScheduledTask -TaskName '{task}' -Action $action -Principal $principal -Settings $settings | Out-Null; "
        f"Start-ScheduledTask -TaskName '{task}'",
    )
    if not ok:
        raise RuntimeError(f"Surface task {task} could not start: {result}")


def plateau_chunk_count(percent, committed, limit):
    if percent not in (*range(75, 81), HIGH_PLATEAU_PERCENT):
        raise ValueError("The capped plateau must be 75–80% or 96% commit")
    chunk = PLATEAU_CHUNK_MIB * 1024 ** 2
    count = (limit * percent // 100 - committed) // chunk - PLATEAU_MARGIN_CHUNKS
    if committed <= 0 or limit <= committed or count <= 0:
        raise ValueError("Surface has insufficient commit headroom for the capped plateau")
    return count


def plateau_workload_script(remote_csv, percent, memory_csv):
    # Start the workload with a hard allocation cap and stop it locally even if SSH is lost.
    return rf"""
$ErrorActionPreference = 'Stop'
if ((Get-ScheduledTask -TaskName CrashLabWatchdog).State -ne 'Disabled') {{
    throw 'CrashLabWatchdog must be disabled before a responsiveness workload'
}}
if (Get-Process Testlimit64 -ErrorAction SilentlyContinue) {{ throw 'Testlimit is already running' }}
$stream = [IO.FileStream]::new('{remote_csv}', 'CreateNew', 'Write', 'Read')
$writer = [IO.StreamWriter]::new($stream)
function Record($event) {{
    $writer.WriteLine((@([DateTime]::UtcNow.ToString('o'), $event, $committed, $limit, $count) -join ','))
    $writer.Flush()
    $stream.Flush($true)
}}
$child = $null
try {{
    $writer.WriteLine('timestamp_utc,event,committed_bytes,commit_limit_bytes,allocation_chunk_count')
    $process = [Diagnostics.ProcessStartInfo]::new()
    $process.FileName = '{TOOL}'
    $process.UseShellExecute = $false
    $process.CreateNoWindow = $true
    $reading = (Get-Content -LiteralPath '{memory_csv}' -Tail 1) -split ','
    if ($reading.Count -ne 5 -or $reading[2] -notmatch '^\d+$' -or $reading[3] -notmatch '^\d+$') {{
        throw 'Surface commit sample is invalid'
    }}
    $time = [DateTimeOffset]::Parse($reading[0], [Globalization.CultureInfo]::InvariantCulture)
    $age = ([DateTimeOffset]::UtcNow - $time).TotalSeconds
    if ($age -lt 0 -or $age -gt {2 * MEMORY_SAMPLE_SECONDS}) {{ throw 'Surface commit sample is stale' }}
    $committed = [long]$reading[2]
    $limit = [long]$reading[3]
    $target = [Math]::Floor([decimal]$limit * {percent} / 100)
    $count = [long]([Math]::Floor(($target - $committed) / {PLATEAU_CHUNK_MIB * 1024 ** 2}) - {PLATEAU_MARGIN_CHUNKS})
    if ($committed -le 0 -or $limit -le $committed -or $count -le 0) {{ throw 'Insufficient commit headroom for the capped plateau' }}
    $process.Arguments = "-accepteula -d {PLATEAU_CHUNK_MIB} -e 1 -c $count"
    $child = [Diagnostics.Process]::Start($process)
    Record 'started'
    if ($child.WaitForExit({PLATEAU_HOLD_LIMIT_MIN * 60000})) {{
        Record 'workload_exit'
    }} else {{
        Record 'intervention'
        $child.Kill()
        if (-not $child.WaitForExit(5000)) {{ throw 'Workload exit could not be verified' }}
    }}
}} finally {{
    if ($null -ne $child) {{
        try {{ if (-not $child.HasExited) {{ $child.Kill() }} }} finally {{ $child.Dispose() }}
    }}
    $writer.Dispose()
}}
"""


def launch_capped_plateau(percent, memory_csv, task, remote_script, remote_csv):
    if not 75 <= percent <= 80:
        raise ValueError("The initial capped plateau must be 75–80% commit")
    launch_responsiveness_script(task, remote_script, plateau_workload_script(remote_csv, percent, memory_csv), 23)
    ok, result = ps(
        "for ($i=0; $i -lt 20; $i++) { "
        f"$rows = @(Get-Content -LiteralPath '{remote_csv}' -TotalCount 2 -ErrorAction SilentlyContinue); "
        "if ($rows.Count -eq 2 -and ($rows[1] -split ',')[1] -eq 'started') { "
        "$rows[1]; exit 0 }; Start-Sleep -Seconds 1 }; "
        "throw 'Capped workload did not report its start'",
        timeout=30,
    )
    if not ok:
        raise RuntimeError(result)
    timestamp, _, committed, limit, count = result.split(",")
    committed, limit, count = int(committed), int(limit), int(count)
    if count != plateau_chunk_count(percent, committed, limit):
        raise RuntimeError("Capped workload reported an inconsistent allocation limit")
    return {
        "inject_time_utc": parse_utc(timestamp).isoformat(),
        "target_commit_percent": percent,
        "commit_bytes_before_workload": committed,
        "commit_limit_before_workload": limit,
        "allocation_chunk_mib": PLATEAU_CHUNK_MIB,
        "allocation_chunk_count": count,
        "allocation_margin_chunks": PLATEAU_MARGIN_CHUNKS,
    }


def ramp_stop_tracker_script():
    """Track adjacent observed slots without depending on the Mac or SSH."""
    return rf"""
$script:previousSlot = $null
$script:consecutive = 0
$script:onsetSlot = $null
function Update-RampProbe($line) {{
    $fields = $line -split ',', 4
    if ($fields.Count -lt 3 -or $fields[0] -eq 'probe_timestamp_utc') {{ return }}
    $stamp = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse($fields[0], [ref]$stamp) -or $stamp -lt $observationStart) {{ return }}
    $slot = [long][Math]::Floor(($stamp.ToUnixTimeMilliseconds() + 2500) / 5000)
    if ($null -ne $previousSlot -and $slot -le $previousSlot) {{ return }}
    $failed = $fields[1] -in @('spawn_error', 'timeout', 'exit_nonzero', 'probe_error')
    if ($fields[1] -eq 'ok') {{
        $latency = 0L
        if (-not [long]::TryParse($fields[2], [ref]$latency) -or $latency -lt 0) {{ return }}
        $failed = $latency -gt {RESPONSIVENESS_FAILURE_MS}
    }} elseif (-not $failed) {{ return }}
    if ($slot -ne ($previousSlot + 1)) {{ $script:consecutive = 0 }}
    $script:previousSlot = $slot
    if ($failed) {{ $script:consecutive++ }} else {{ $script:consecutive = 0 }}
    if ($consecutive -eq 3) {{ $script:onsetSlot = $slot - 2 }}
}}
"""


def plateau_topup_count_script():
    return f"""
    function Get-PlateauTopUpCount($stableCommit, $currentCommit, $currentLimit) {{
        $target = [Math]::Floor([decimal]$currentLimit * {HIGH_PLATEAU_PERCENT} / 100)
        $remaining = $target - [Math]::Max($stableCommit, $currentCommit)
        $chunks = [long]([Math]::Floor($remaining / {PLATEAU_CHUNK_MIB * 1024 ** 2}) - {PLATEAU_MARGIN_CHUNKS})
        return [Math]::Max(0L, [Math]::Min({PLATEAU_TOPUP_MAX_CHUNKS}L, $chunks))
    }}
"""


def ramp_workload_script(remote_csv, probe_csv, start=None, *, ramp_rate=10, plateau_memory_csv=None):
    observation_start = start.isoformat() if start else "1970-01-01T00:00:00+00:00"
    limit_minutes = RAMP_LIMIT_MINUTES[ramp_rate]
    arguments = f"$process.Arguments = '-accepteula -d {ramp_rate} -e 1'"
    scenario = "supervised_ramp"
    extra_header = extra_fields = guard = ""
    if plateau_memory_csv is not None:
        limit_minutes = PLATEAU_HOLD_LIMIT_MIN
        scenario = "supervised_plateau"
        extra_header = ",committed_bytes,commit_limit_bytes,allocation_chunk_count"
        extra_fields = ", $recordCommit, $recordLimit, $recordCount"
        arguments = rf"""
    {plateau_topup_count_script()}
    function Read-Commit {{
        $reading = (Get-Content -LiteralPath '{plateau_memory_csv}' -Tail 1) -split ','
        if ($reading.Count -ne 5 -or $reading[2] -notmatch '^\d+$' -or $reading[3] -notmatch '^\d+$') {{
            throw 'Surface commit sample is invalid'
        }}
        $stamp = [DateTimeOffset]::Parse($reading[0], [Globalization.CultureInfo]::InvariantCulture)
        $age = ([DateTimeOffset]::UtcNow - $stamp).TotalSeconds
        if ($age -lt 0 -or $age -gt {2 * MEMORY_SAMPLE_SECONDS} -or [long]$reading[3] -le 0) {{
            throw 'Surface commit sample is stale or invalid'
        }}
        return @([long]$reading[2], [long]$reading[3])
    }}
    $committed, $limit = Read-Commit
    $target = [Math]::Floor([decimal]$limit * {HIGH_PLATEAU_PERCENT} / 100)
    $count = [long]([Math]::Floor(($target - $committed) / {PLATEAU_CHUNK_MIB * 1024 ** 2}) - {PLATEAU_MARGIN_CHUNKS})
    if ($committed -le 0 -or $limit -le $committed -or $count -le 0) {{
        throw 'Insufficient commit headroom for the capped plateau'
    }}
    $process.Arguments = "-accepteula -d {PLATEAU_CHUNK_MIB} -e 1 -c $count"
    $nextGuardCheck = 0
    $topUpDecided = $false
    $topUpAllocationEnds = $null
"""
        guard = rf"""
        if ($null -ne $topUpChild -and $topUpChild.HasExited) {{
            $guardReason = 'topup_exit'
            break
        }}
        if ($timer.ElapsedMilliseconds -ge $nextGuardCheck) {{
            $nextGuardCheck = $timer.ElapsedMilliseconds + 1000
            try {{
                $current, $currentLimit = Read-Commit
                if ($current * 100 -ge $currentLimit * {HIGH_PLATEAU_GUARD_PERCENT}) {{
                    $guardReason = 'commit_guard'
                    break
                }}
            }} catch {{
                $guardReason = 'counter_unavailable'
                break
            }}
            if (-not $topUpDecided -and $timer.ElapsedMilliseconds -ge (($count + 60) * 1000) -and
                $timer.ElapsedMilliseconds -le ({limit_minutes * 60000} - 360000)) {{
                $samples = @(Get-Content -LiteralPath '{plateau_memory_csv}' -Tail 13)
                if ($samples.Count -eq 13) {{
                    $rates = @()
                    $previousTime = $null
                    $firstTime = $null
                    foreach ($sample in $samples) {{
                        $fields = $sample -split ','
                        $sampleTime = [DateTimeOffset]::MinValue
                        if ($fields.Count -ne 5 -or $fields[2] -notmatch '^\d+$' -or
                            $fields[3] -ne [string]$currentLimit -or
                            -not [DateTimeOffset]::TryParse($fields[0], [ref]$sampleTime) -or
                            ($null -ne $previousTime -and
                             [Math]::Abs(($sampleTime - $previousTime).TotalSeconds - {MEMORY_SAMPLE_SECONDS}) -gt 1)) {{
                            break
                        }}
                        if ($null -eq $firstTime) {{ $firstTime = $sampleTime }}
                        $previousTime = $sampleTime
                        $rates += [long]$fields[2]
                    }}
                    if ($rates.Count -eq 13 -and ($previousTime - $firstTime).TotalSeconds -ge 59) {{
                        $sorted = @($rates | Sort-Object)
                        if ($sorted[-1] - $sorted[0] -le {PLATEAU_STABILITY_MIB * 1024 ** 2}) {{
                            $topUpDecided = $true
                            Record 'plateau_settle' 'stable_commit' $sorted[6] $currentLimit $count $firstTime.UtcDateTime
                            $topUpCount = Get-PlateauTopUpCount $sorted[6] $current $currentLimit
                            if ($topUpCount -gt 0) {{
                                $process.Arguments = "-accepteula -d {PLATEAU_CHUNK_MIB} -e 1 -c $topUpCount"
                                $topUpChild = [Diagnostics.Process]::Start($process)
                                [CrashLabRampJob]::Add($job, $topUpChild.Handle)
                                Record 'topup_started' 'settled_commit' $sorted[6] $currentLimit $topUpCount
                                $topUpAllocationEnds = $timer.ElapsedMilliseconds + $topUpCount * 1000
                            }} else {{
                                Record 'topup_skipped' 'no_headroom' $sorted[6] $currentLimit 0
                            }}
                        }}
                    }}
                }}
            }}
            if ($topUpAllocationEnds -and $timer.ElapsedMilliseconds -ge $topUpAllocationEnds) {{
                Record 'topup_allocation_end' 'nominal_duration' $sorted[6] $currentLimit $topUpCount
                $topUpAllocationEnds = $null
            }}
        }}
"""
    return rf"""
$ErrorActionPreference = 'Stop'
if ((Get-ScheduledTask -TaskName CrashLabWatchdog).State -ne 'Disabled') {{
    throw 'CrashLabWatchdog must be disabled before the ramp'
}}
if (Get-Process Testlimit64 -ErrorAction SilentlyContinue) {{ throw 'Testlimit is already running' }}
$observationStart = [DateTimeOffset]::Parse('{observation_start}')
{ramp_stop_tracker_script()}
Add-Type @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class CrashLabRampJob {{
    [StructLayout(LayoutKind.Sequential)]
    struct Limits {{
        public long PerProcessTime, PerJobTime;
        public uint Flags;
        public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }}
    [StructLayout(LayoutKind.Sequential)]
    struct Extended {{
        public Limits Basic;
        public ulong ReadOperations, WriteOperations, OtherOperations;
        public ulong ReadBytes, WriteBytes, OtherBytes;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }}
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern SafeFileHandle CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(SafeFileHandle job, int infoClass, ref Extended info, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(SafeFileHandle job, IntPtr process);
    public static SafeFileHandle Create() {{
        SafeFileHandle job = CreateJobObject(IntPtr.Zero, null);
        if (job.IsInvalid) {{
            int error = Marshal.GetLastWin32Error();
            job.Dispose();
            throw new Win32Exception(error);
        }}
        Extended info = new Extended();
        info.Basic.Flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (!SetInformationJobObject(job, 9, ref info, (uint)Marshal.SizeOf(typeof(Extended)))) {{
            int error = Marshal.GetLastWin32Error();
            job.Dispose();
            throw new Win32Exception(error);
        }}
        return job;
    }}
    public static void Add(SafeFileHandle job, IntPtr process) {{
        if (!AssignProcessToJobObject(job, process)) throw new Win32Exception(Marshal.GetLastWin32Error());
    }}
}}
'@
$source = [IO.FileStream]::new('{probe_csv}', 'Open', 'Read', 'ReadWrite')
$reader = [IO.StreamReader]::new($source)
$stream = [IO.FileStream]::new('{remote_csv}', 'CreateNew', 'Write', 'Read')
$writer = [IO.StreamWriter]::new($stream)
$child = $null
$topUpChild = $null
$job = $null
$script:pending = ''
function Read-RampProbes {{
    $received = $reader.ReadToEnd()
    if (-not $received) {{ return }}
    $lines = ($pending + $received) -split "`n"
    $script:pending = $lines[-1]
    for ($i = 0; $i -lt ($lines.Count - 1); $i++) {{
        Update-RampProbe ($lines[$i].TrimEnd("`r"))
        if ($null -ne $onsetSlot) {{ return }}
    }}
}}
function Record($event, $reason, $recordCommit=$committed, $recordLimit=$limit, $recordCount=$count,
                $recordTime=[DateTime]::UtcNow) {{
    $onset = if ($null -ne $onsetSlot) {{
        [DateTimeOffset]::FromUnixTimeMilliseconds($onsetSlot * 5000).UtcDateTime.ToString('o')
    }} else {{ '' }}
    $writer.WriteLine((@($recordTime.ToString('o'), $event, $reason, $onset{extra_fields}) -join ','))
    $writer.Flush()
    $stream.Flush($true)
}}
try {{
    $writer.WriteLine('timestamp_utc,event,reason,onset_utc{extra_header}')
    Read-RampProbes
    if ($null -ne $onsetSlot) {{ throw 'PowerShell responsiveness already failed during baseline' }}
    $process = [Diagnostics.ProcessStartInfo]::new()
    $process.FileName = '{TOOL}'
    {arguments}
    $process.UseShellExecute = $false
    $process.CreateNoWindow = $true
    $job = [CrashLabRampJob]::Create()
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $child = [Diagnostics.Process]::Start($process)
    [CrashLabRampJob]::Add($job, $child.Handle)
    Record 'started' '{scenario}'
    $guardReason = $null
    while ($timer.ElapsedMilliseconds -lt {limit_minutes * 60000} -and -not $child.HasExited) {{
        Read-RampProbes
        if ($null -ne $onsetSlot) {{ break }}
{guard}
        Start-Sleep -Milliseconds 250
    }}
    if ($child.HasExited) {{
        Record 'workload_exit' 'natural_exit'
        if ($null -ne $topUpChild -and -not $topUpChild.HasExited) {{ $topUpChild.Kill() }}
    }} else {{
        $reason = if ($null -ne $onsetSlot) {{ 'probe_onset' }} elseif ($guardReason) {{ $guardReason }} else {{ 'time_cap' }}
        Record 'intervention' $reason
        $child.Kill()
        if ($null -ne $topUpChild -and -not $topUpChild.HasExited) {{ $topUpChild.Kill() }}
        if (-not $child.WaitForExit(5000)) {{ throw 'Ramp workload exit could not be verified' }}
        if ($null -ne $topUpChild -and -not $topUpChild.WaitForExit(5000)) {{
            throw 'Plateau top-up exit could not be verified'
        }}
        Record 'stopped' $reason
    }}
}} finally {{
    if ($null -ne $job) {{ $job.Dispose() }}
    if ($null -ne $topUpChild) {{
        try {{ if (-not $topUpChild.HasExited) {{ $topUpChild.Kill() }} }} finally {{ $topUpChild.Dispose() }}
    }}
    if ($null -ne $child) {{
        try {{ if (-not $child.HasExited) {{ $child.Kill() }} }} finally {{ $child.Dispose() }}
    }}
    $writer.Dispose()
    $reader.Dispose()
}}
"""


def launch_ramp(start, probe_csv, task, remote_script, remote_csv, *, ramp_rate=10, plateau_memory_csv=None):
    limit_minutes = PLATEAU_HOLD_LIMIT_MIN if plateau_memory_csv is not None else RAMP_LIMIT_MINUTES[ramp_rate]
    launch_responsiveness_script(
        task, remote_script, ramp_workload_script(
            remote_csv, probe_csv, start, ramp_rate=ramp_rate, plateau_memory_csv=plateau_memory_csv,
        ),
        limit_minutes + 1,
    )
    ok, result = ps(
        "for ($i=0; $i -lt 20; $i++) { "
        f"$rows = @(Get-Content -LiteralPath '{remote_csv}' -TotalCount 2 -ErrorAction SilentlyContinue); "
        "if ($rows.Count -eq 2 -and ($rows[1] -split ',')[1] -eq 'started') { "
        "$rows[1]; exit 0 }; Start-Sleep -Seconds 1 }; "
        "throw 'Supervised ramp workload did not report its start'",
        timeout=30,
    )
    if not ok:
        raise RuntimeError(result)
    metadata = {
        "inject_time_utc": parse_utc(result.split(",", 1)[0]).isoformat(),
        "allocation_chunk_mib": ramp_rate,
        "allocation_chunk_interval_seconds": 1,
        "workload_time_cap_minutes": limit_minutes,
    }
    if plateau_memory_csv is not None:
        committed, limit, count = map(int, result.split(",")[4:])
        if count != plateau_chunk_count(HIGH_PLATEAU_PERCENT, committed, limit):
            raise RuntimeError("Capped workload reported an inconsistent allocation limit")
        metadata.update(
            target_commit_percent=HIGH_PLATEAU_PERCENT,
            commit_guard_percent=HIGH_PLATEAU_GUARD_PERCENT,
            commit_bytes_before_workload=committed,
            commit_limit_before_workload=limit,
            allocation_chunk_count=count,
            allocation_margin_chunks=PLATEAU_MARGIN_CHUNKS,
        )
    return metadata


def summarize_ramp(staging, metadata):
    with open(os.path.join(staging, "workload.csv"), newline="") as stream:
        events = list(csv.DictReader(stream))
    stop = next((row for row in events if row["event"] in ("intervention", "workload_exit")), None)
    if not stop:
        if not metadata["reboot_times_utc"]:
            raise RuntimeError("The supervised ramp has no recorded intervention or exit")
        return
    metadata["intervention_times_utc"] = [parse_utc(stop["timestamp_utc"]).isoformat()]
    metadata["workload_stop_reason"] = stop["reason"]
    metadata["local_probe_onset_utc"] = parse_utc(stop["onset_utc"]).isoformat() if stop["onset_utc"] else None
    metadata["workload_stop_verified"] = stop["event"] == "workload_exit" or any(
        row["event"] == "stopped" for row in events
    )


def record_ssh_probe(stop, output):
    """Record transport availability independently of the local Surface probes."""
    target = time.monotonic()
    while not stop.is_set():
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        started = time.monotonic()
        ok = reachable()
        output.append((stamp, "ok" if ok else "fail", round((time.monotonic() - started) * 1000)))
        target += 15
        stop.wait(max(0, target - time.monotonic()))


def verify_protocol_freeze(project=None):
    """Return the freeze commit after verifying all pinned files against it."""
    project = Path(project) if project is not None else Path(__file__).resolve().parent
    protocol_files = ("protocol/PROTOCOL_v3.md", "protocol/protocol_v3.json")
    source_files = ("crashloop.py", "surface_responsiveness.py", "make_surface_dataset.py",
                    "score_slope.py", "make_protocol.py")
    files = protocol_files + source_files

    def git(*args):
        result = subprocess.run(["git", "-C", str(project), *args], capture_output=True)
        if result.returncode:
            raise RuntimeError("Protocol v3 must be frozen in Git before running replicates")
        return result.stdout

    revision = git("log", "-1", "--format=%H", "--", *protocol_files).decode().strip()
    if not revision:
        raise RuntimeError("Protocol v3 must be frozen in Git before running replicates")
    message = git("show", "-s", "--format=%B", revision)
    if git("diff", "--cached", "--name-only", "HEAD", "--", *files):
        raise RuntimeError("Protocol v3 pinned files have staged changes")
    if git("diff", "--name-only", "HEAD", "--", *files):
        raise RuntimeError("Protocol v3 pinned files have working-tree changes")
    for name in files:
        frozen = git("show", f"{revision}:{name}")
        local = project / name
        if (git("ls-tree", revision, "--", name) != git("ls-tree", "HEAD", "--", name)
                or local.is_symlink() or not local.is_file() or frozen != local.read_bytes()):
            raise RuntimeError(f"Protocol v3 freeze does not match pinned file {name}")
        if name in protocol_files and hashlib.sha256(frozen).hexdigest().encode() not in message:
            raise RuntimeError(f"Protocol v3 freeze does not match {name} and its SHA-256")
    return revision


def restart_for_protocol():
    freeze_commit = verify_protocol_freeze()
    boot = boot_time(required=True)
    ok, reason = ps(
        "if ((Get-ScheduledTask -TaskName CrashLabWatchdog).State -ne 'Disabled' -or "
        "    (Get-Process Testlimit64 -ErrorAction SilentlyContinue) -or "
        "    (Get-ScheduledTask -TaskName 'CrashLab-Response-*' -ErrorAction SilentlyContinue "
        "     | Where-Object State -eq 'Running')) { throw 'Surface is not ready for a protocol restart' }; "
        "Restart-Computer -Force",
        timeout=30,
    )
    if not ok and not any(word in reason.lower() for word in ("reset", "closed", "broken pipe")):
        raise RuntimeError(f"Could not request the Surface restart: {reason}")
    new_boot = None

    def reboot_ready():
        nonlocal new_boot
        new_boot = verified_boot_time()
        return new_boot is not None and (
            datetime.datetime.fromisoformat(new_boot) - datetime.datetime.fromisoformat(boot)
        ).total_seconds() > 5

    if not wait_until(reboot_ready, BOOT_WAIT_MIN * 60):
        raise RuntimeError("Could not verify a fresh Surface boot")
    now = parse_utc(surface_utc_now())
    earliest = datetime.datetime.fromisoformat(new_boot) + datetime.timedelta(minutes=PROTOCOL_SETTLE_MIN)
    time.sleep(max(0, (earliest - now).total_seconds()))
    return new_boot, freeze_commit


def healthy_burst_schedule(seed):
    rng = random.Random(seed)
    schedule = []
    offset = 0
    while True:
        offset += rng.randint(60, 180)
        size = rng.randint(200, 600)
        hold = rng.randint(15, 60)
        if offset + hold > PROTOCOL_HEALTHY_MIN * 60:
            return schedule
        schedule.append((offset, size, hold))


def protocol_workload_script(remote_csv, probe_csv, memory_csv, start, limit_minutes, reason,
                             preflight_limit, plan, body):
    return rf"""
$ErrorActionPreference = 'Stop'
if ((Get-ScheduledTask -TaskName CrashLabWatchdog).State -ne 'Disabled') {{
    throw 'CrashLabWatchdog must be disabled before the protocol workload'
}}
if (Get-Process Testlimit64 -ErrorAction SilentlyContinue) {{ throw 'Testlimit is already running' }}
$observationStart = [DateTimeOffset]::Parse('{start.isoformat()}')
{ramp_stop_tracker_script()}
Add-Type @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
public sealed class CrashLabAllocation {{
    readonly List<IntPtr> blocks = new List<IntPtr>();
    const uint CommitAndReserve = 0x3000, ReadWrite = 4, Release = 0x8000;
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr VirtualAlloc(IntPtr address, UIntPtr size, uint type, uint protection);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool VirtualFree(IntPtr address, UIntPtr size, uint type);
    public void Add(int bytes) {{
        IntPtr block = VirtualAlloc(IntPtr.Zero, (UIntPtr)(uint)bytes, CommitAndReserve, ReadWrite);
        if (block == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        blocks.Add(block);
        for (int offset = 0; offset < bytes; offset += Environment.SystemPageSize)
            Marshal.WriteByte(block, offset, 1);
    }}
    public void Clear() {{
        for (int i = blocks.Count - 1; i >= 0; i--) {{
            if (!VirtualFree(blocks[i], UIntPtr.Zero, Release))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            blocks.RemoveAt(i);
        }}
    }}
    public static void CpuPulse() {{
        var timer = Stopwatch.StartNew();
        while (timer.ElapsedMilliseconds < 5) {{ System.Threading.Thread.SpinWait(64); }}
    }}
}}
'@
$allocation = [CrashLabAllocation]::new()
$reader = [IO.StreamReader]::new([IO.FileStream]::new('{probe_csv}', 'Open', 'Read', 'ReadWrite'))
$stream = [IO.FileStream]::new('{remote_csv}', 'CreateNew', 'Write', 'Read')
$writer = [IO.StreamWriter]::new($stream)
$script:pending = ''
function Read-ProtocolProbes {{
    $received = $reader.ReadToEnd()
    if (-not $received) {{ return }}
    $lines = ($pending + $received) -split '\n'
    $script:pending = $lines[-1]
    for ($i = 0; $i -lt ($lines.Count - 1); $i++) {{
        Update-RampProbe ($lines[$i].TrimEnd([char]13))
        if ($null -ne $onsetSlot) {{ return }}
    }}
}}
function Read-ProtocolCommit {{
    $fields = (Get-Content -LiteralPath '{memory_csv}' -Tail 1) -split ','
    if ($fields.Count -ne 5 -or $fields[2] -notmatch '^\d+$' -or $fields[3] -notmatch '^\d+$' -or
        [long]$fields[2] -le 0 -or [long]$fields[3] -le 0) {{ throw 'Invalid commit sample' }}
    $stamp = [DateTimeOffset]::Parse($fields[0], [Globalization.CultureInfo]::InvariantCulture)
    $age = ([DateTimeOffset]::UtcNow - $stamp).TotalSeconds
    if ($age -lt 0 -or $age -gt {2 * MEMORY_SAMPLE_SECONDS}) {{ throw 'Stale commit sample' }}
    return @($stamp, [long]$fields[2], [long]$fields[3])
}}
function Record($event, $reason, $burstSeconds='', $burstMiB='', $burstHold='') {{
    $onset = if ($null -ne $onsetSlot) {{
        [DateTimeOffset]::FromUnixTimeMilliseconds($onsetSlot * 5000).UtcDateTime.ToString('o')
    }} else {{ '' }}
    $writer.WriteLine((@([DateTime]::UtcNow.ToString('o'), $event, $reason, $onset,
        $burstSeconds, $burstMiB, $burstHold) -join ','))
    $writer.Flush()
    $stream.Flush($true)
}}
try {{
    $writer.WriteLine('timestamp_utc,event,reason,onset_utc,burst_offset_seconds,burst_mib,burst_hold_seconds')
    Read-ProtocolProbes
    if ($null -ne $onsetSlot) {{ throw 'PowerShell responsiveness already failed during baseline' }}
    $initialTime, $initialCommit, $initialLimit = Read-ProtocolCommit
    if ($initialCommit * 100 -ge $initialLimit * {preflight_limit}) {{
        throw 'Insufficient commit headroom before the protocol workload'
    }}
    $timer = [Diagnostics.Stopwatch]::StartNew()
    Record 'started' '{reason}'
    {plan}
    $stopReason = 'time_cap'
    while ($timer.ElapsedMilliseconds -lt {limit_minutes * 60000}) {{
        Read-ProtocolProbes
        if ($null -ne $onsetSlot) {{ $stopReason = 'probe_onset'; break }}
        {body}
        Start-Sleep -Milliseconds 250
    }}
    Record 'intervention' $stopReason
    $allocation.Clear()
    Record 'stopped' $stopReason
}} finally {{
    try {{ $allocation.Clear() }} finally {{ $writer.Dispose(); $reader.Dispose() }}
}}
"""


def stepped_workload_script(remote_csv, probe_csv, memory_csv, start, pattern):
    target, hold = STEPPED_PATTERNS[pattern]
    plan = f"""
    $pauseTarget = {target}
    $pauseSeconds = {hold}
    $phase = 'initial_rise'
    $nextAllocation = 0L
    $nextRead = 0L
    $lastSample = $null
    $sinceSample = 0L
    $chunkBytes = {10 * 1024 ** 2}
"""
    body = """
        $elapsed = $timer.ElapsedMilliseconds
        if ($phase -eq 'pause' -and $elapsed -ge $resumeAt) {
            Record 'resume' 'pause_complete'
            $phase = 'post_pause_rise'
        }
        if ($phase -ne 'post_pause_rise' -and $elapsed -ge $nextRead) {
            $nextRead = $elapsed + 1000
            try { $observed, $committed, $limit = Read-ProtocolCommit }
            catch { $stopReason = 'counter_unavailable'; break }
            if ($observed -ne $lastSample) {
                $lastSample = $observed
                $sinceSample = 0L
            }
            if ($committed * 100 -ge $limit * 98) {
                $stopReason = 'commit_guard'
                break
            }
        }
        if ($phase -eq 'initial_rise' -and ($committed + $sinceSample + $chunkBytes) * 100 -gt $limit * $pauseTarget) {
            if ($committed * 100 -ge $limit * ($pauseTarget - 1)) {
                Record 'pause_start' 'target_reached'
                $resumeAt = $elapsed + $pauseSeconds * 1000
                $phase = 'pause'
            }
        } elseif ($phase -ne 'pause' -and $elapsed -ge $nextAllocation) {
            $nextAllocation = $elapsed + 1000
            try { $allocation.Add($chunkBytes); $sinceSample += $chunkBytes }
            catch {
                # Preserve pressure until the independent probe reports onset or the cap expires.
                $nextAllocation = $elapsed + 5000
            }
        }
"""
    return protocol_workload_script(
        remote_csv, probe_csv, memory_csv, start, STEPPED_LIMIT_MIN, f"stepped_{pattern}", target, plan, body,
    )


def healthy_workload_script(remote_csv, probe_csv, memory_csv, start, seed):
    schedule = "; ".join(f"{offset}; {mib}; {hold}" for offset, mib, hold in healthy_burst_schedule(seed))
    plan = f"""
    $bursts = @({schedule})
    for ($i = 0; $i -lt $bursts.Count; $i += 3) {{
        Record 'burst_planned' 'seeded' $bursts[$i] $bursts[$i + 1] $bursts[$i + 2]
    }}
    $index = 0
    $releaseAt = $null
    $nextRead = 0L
"""
    body = """
        $elapsed = $timer.ElapsedMilliseconds
        if ($null -ne $releaseAt -and $elapsed -ge $releaseAt) {
            $allocation.Clear()
            $releaseAt = $null
            Record 'burst_release' 'duration_complete'
        }
        if ($elapsed -ge $nextRead) {
            $nextRead = $elapsed + 1000
            try { $observed, $committed, $limit = Read-ProtocolCommit }
            catch { $stopReason = 'counter_unavailable'; break }
            if ($committed * 100 -ge $limit * 85) {
                $stopReason = 'commit_guard'
                break
            }
        }
        if ($null -eq $releaseAt -and $index -lt $bursts.Count -and $elapsed -ge $bursts[$index] * 1000) {
            $offset = $bursts[$index]
            $mib = $bursts[$index + 1]
            $hold = $bursts[$index + 2]
            $index += 3
            # A one-point margin covers drift between counter samples.
            if (($committed + $mib * 1MB) * 100 -ge $limit * 84) {
                Record 'burst_skipped' 'insufficient_headroom' $offset $mib $hold
                $stopReason = 'burst_guard'
                break
            }
            try { $allocation.Add($mib * 1MB) }
            catch {
                Record 'burst_skipped' 'allocation_failed' $offset $mib $hold
                $stopReason = 'allocation_failed'
                break
            }
            $releaseAt = $timer.ElapsedMilliseconds + $hold * 1000
            Record 'burst_start' 'seeded' $offset $mib $hold
        }
        [CrashLabAllocation]::CpuPulse()
"""
    return protocol_workload_script(
        remote_csv, probe_csv, memory_csv, start, PROTOCOL_HEALTHY_MIN, "healthy_bursts", 84, plan, body,
    )


def launch_protocol_workload(start, probe_csv, memory_csv, task, remote_script, remote_csv, *, stepped, burst_seed):
    if stepped is not None:
        script = stepped_workload_script(remote_csv, probe_csv, memory_csv, start, stepped)
        duration = STEPPED_LIMIT_MIN
        target, hold = STEPPED_PATTERNS[stepped]
        metadata = {"stepped_pattern": stepped, "pause_commit_percent": target, "pause_duration_seconds": hold}
    else:
        script = healthy_workload_script(remote_csv, probe_csv, memory_csv, start, burst_seed)
        duration = PROTOCOL_HEALTHY_MIN
        metadata = {"burst_seed": burst_seed}
    launch_responsiveness_script(task, remote_script, script, duration + 1)
    ok, result = ps(
        "for ($i=0; $i -lt 30; $i++) { "
        f"$rows = @(Get-Content -LiteralPath '{remote_csv}' -TotalCount 2 -ErrorAction SilentlyContinue); "
        "if ($rows.Count -eq 2 -and ($rows[1] -split ',')[1] -eq 'started') { "
        "$rows[1]; exit 0 }; Start-Sleep -Seconds 1 }; "
        "throw 'Protocol workload did not report its start'",
        timeout=40,
    )
    if not ok:
        raise RuntimeError(result)
    return {"inject_time_utc": parse_utc(result.split(",", 1)[0]).isoformat(),
            "workload_time_cap_minutes": duration, **metadata}


def collect_responsiveness_run(*, timeout_ms, plateau_percent=None, ramp_rate=None,
                               stepped=None, burst_seed=None, protocol_v3=False, pilot=False):
    if protocol_v3 and pilot:
        raise ValueError("Pilot and protocol v3 runs are mutually exclusive")
    if protocol_v3:
        boot, freeze_commit = restart_for_protocol()
    else:
        boot = boot_time(required=True)
    mode = "stepped" if stepped is not None else (
        "ramp" if ramp_rate is not None else ("plateau" if plateau_percent is not None else "control")
    )
    has_workload = mode != "control" or burst_seed is not None
    high_plateau = plateau_percent == HIGH_PLATEAU_PERCENT
    runid = f"laptop_{datetime.datetime.now():%Y%m%d_%H%M%S}_responsiveness-{mode}"
    if pilot:
        runid += "_pilot"
    tag = f"CrashLab-Response-{uuid.uuid4().hex}"
    resources = {}
    workload = {}
    print(f"\n=== {runid} ===", flush=True)
    try:
        for kind in RESPONSIVENESS_FILES:
            task = f"{tag}-{kind}"
            script, remote_csv = rf"C:\CrashLab\{task}.ps1", rf"C:\CrashLab\{task}.csv"
            resources[kind] = (task, script, remote_csv)
            launch_responsiveness_sampler(
                kind, task, script, remote_csv, timeout_ms if kind == "powershell" else 2000,
                limit_minutes=45 if protocol_v3 or pilot else 40,
            )
        checks = [
            f"(@(Get-Content -LiteralPath '{path}' -TotalCount 2 -ErrorAction SilentlyContinue).Count -eq 2)"
            for _, _, path in resources.values()
        ]
        ready, result = ps(
            "for ($i = 0; $i -lt 25; $i++) { "
            f"if ({' -and '.join(checks)}) {{ [DateTime]::UtcNow.ToString('o'); exit 0 }}; "
            "Start-Sleep -Seconds 1 }; throw 'Independent sampler streams did not become ready'",
            timeout=35,
        )
        if not ready:
            raise RuntimeError(result)
        clock = parse_utc(result)
        duration = FIRST_BASELINE_MIN if has_workload else HEALTHY_CONTROL_MIN
        start, boundary = responsiveness_grid_bounds(result, duration)
        print(f"  {mode} observation starts {start.isoformat()}", flush=True)
        stop, ssh_samples = threading.Event(), []
        cross_check = threading.Thread(target=record_ssh_probe, args=(stop, ssh_samples))
        cross_check.start()
        try:
            time.sleep((boundary - clock).total_seconds() + 2)
            end = boundary
            if has_workload:
                workload_task = f"{tag}-workload"
                workload_script = rf"C:\CrashLab\{workload_task}.ps1"
                workload_csv = rf"C:\CrashLab\{workload_task}.csv"
                resources["workload"] = (workload_task, workload_script, workload_csv)
                if stepped is not None or burst_seed is not None:
                    workload = launch_protocol_workload(
                        start, resources["powershell"][2], resources["memory"][2],
                        workload_task, workload_script, workload_csv, stepped=stepped, burst_seed=burst_seed,
                    )
                    limit = workload["workload_time_cap_minutes"]
                    print(f"  protocol workload started ({mode})", flush=True)
                elif ramp_rate is not None or high_plateau:
                    workload = launch_ramp(
                        start, resources["powershell"][2], workload_task, workload_script,
                        workload_csv, ramp_rate=ramp_rate or PLATEAU_CHUNK_MIB,
                        plateau_memory_csv=resources["memory"][2] if high_plateau else None,
                    )
                    limit = workload["workload_time_cap_minutes"]
                    if high_plateau:
                        print(f"  guarded plateau started, target {plateau_percent}% commit", flush=True)
                    else:
                        print(f"  supervised {ramp_rate} MiB/s ramp started with a Windows-local stop", flush=True)
                else:
                    workload = launch_capped_plateau(
                        plateau_percent, resources["memory"][2], workload_task, workload_script, workload_csv,
                    )
                    limit = PLATEAU_HOLD_LIMIT_MIN
                    print(f"  bounded plateau started, target {plateau_percent}% commit", flush=True)
                _, end = responsiveness_grid_bounds(workload["inject_time_utc"], limit + 1)
                time.sleep((end - parse_utc(workload["inject_time_utc"])).total_seconds() + 2)
            recovered_boot = None
            def read_recovered_boot():
                nonlocal recovered_boot
                recovered_boot = verified_boot_time()
                return recovered_boot

            if not wait_until(read_recovered_boot, BOOT_WAIT_MIN * 60):
                raise RuntimeError("Surface did not respond after the observation window")
        finally:
            stop.set()
            cross_check.join()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"{error}; remote telemetry prefix C:\\CrashLab\\{tag}") from error
    finally:
        stopped = [remove_workload_task(task) for task, _, _ in resources.values()]

    if not all(stopped):
        raise RuntimeError(f"Could not verify sampler cleanup; remote telemetry prefix C:\\CrashLab\\{tag}")
    changed_boot = recovered_boot != boot
    if protocol_v3:
        boot_delta = datetime.datetime.fromisoformat(recovered_boot) - datetime.datetime.fromisoformat(boot)
        changed_boot = abs(boot_delta.total_seconds()) > 5
    metadata = {
        "run_id": runid, "machine": MACHINE, "fault": f"responsiveness-{mode}",
        "status": "reboot_unverified" if changed_boot else ("healthy" if mode == "control" else mode),
        "died": False,
        "boot_time_local": boot,
        "experiment": "surface-responsiveness-v3" if protocol_v3 else "surface-responsiveness-v2",
        "probe_timeout_ms": timeout_ms, "cmd_probe_timeout_ms": 2000,
        "probe_observation_cap_ms": timeout_ms,
        "probe_failure_threshold_ms": RESPONSIVENESS_FAILURE_MS,
        "observation_start_utc": start.isoformat(), "observation_end_utc": end.isoformat(),
        "intervention_times_utc": [],
        "reboot_times_utc": [datetime.datetime.fromisoformat(recovered_boot).astimezone(datetime.timezone.utc).isoformat()] if changed_boot else [],
        **workload,
    }
    if protocol_v3:
        metadata["protocol_version"] = 3
        metadata["freeze_commit"] = freeze_commit
    if pilot:
        metadata["pilot"] = True
    with tempfile.TemporaryDirectory(prefix=".crashlab-responsiveness-", dir=os.path.dirname(RUNS_DIR)) as staging:
        for kind, (_, _, remote_csv) in resources.items():
            filename = "workload.csv" if kind == "workload" else RESPONSIVENESS_FILES[kind]
            local = os.path.join(staging, filename)
            copy_remote(remote_csv.replace("\\", "/"), local, checked=True)
        with open(os.path.join(staging, "ssh_probe.csv"), "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(PROBE_COLUMNS)
            writer.writerows(ssh_samples)
        collect_event_exports(staging, checked=True)
        if ramp_rate is not None or high_plateau or stepped is not None or burst_seed is not None:
            summarize_ramp(staging, metadata)
        if plateau_percent is not None:
            summarize_plateau(staging, metadata)
        write_run_metadata(staging, metadata)
        destination = os.path.join(RUNS_DIR, runid)
        os.rename(staging, destination)
    for _, script, remote_csv in resources.values():
        ok, detail = ps(f"Remove-Item -LiteralPath '{script}', '{remote_csv}'")
        if not ok:
            print(f"  Surface telemetry cleanup failed: {detail}", flush=True)
    print(f"  saved responsiveness telemetry -> {destination}", flush=True)
    return destination


def summarize_plateau(staging, metadata):
    with open(os.path.join(staging, "workload.csv"), newline="") as stream:
        events = list(csv.DictReader(stream))
        stops = [
            parse_utc(row["timestamp_utc"]) for row in events
            if row["event"] in ("intervention", "workload_exit")
        ]
    if not stops:
        if not metadata["reboot_times_utc"]:
            raise RuntimeError("The capped workload has no verified stop timestamp")
        return

    stop = stops[0]
    injection = parse_utc(metadata["inject_time_utc"])
    metadata["intervention_times_utc"] = [stop.isoformat()]
    topup = next((row for row in events if row["event"] in ("topup_started", "topup_skipped")), None)
    if topup:
        metadata["topup_allocation_chunk_count"] = int(topup["allocation_chunk_count"])
        metadata["topup_reference_commit_bytes"] = int(topup["committed_bytes"])
        if topup["event"] == "topup_started":
            topup_time = parse_utc(topup["timestamp_utc"]).isoformat()
            metadata["topup_time_utc"] = topup_time
    with open(os.path.join(staging, "memory.csv"), newline="") as stream:
        readings = [
            (parse_utc(row["timestamp_utc"]), 100 * int(row["committed_bytes"]) / int(row["commit_limit_bytes"]))
            for row in csv.DictReader(stream)
            if row.get("committed_bytes") and row.get("commit_limit_bytes")
        ]
    workload = [ratio for stamp, ratio in readings if injection <= stamp < stop]
    plateau = sorted(ratio for stamp, ratio in readings if max(injection, stop - datetime.timedelta(minutes=5)) <= stamp < stop)
    metadata["observed_peak_commit_percent"] = max(workload, default=None)
    metadata["observed_final_5_min_median_commit_percent"] = (
        (plateau[(len(plateau) - 1) // 2] + plateau[len(plateau) // 2]) / 2 if plateau else None
    )
    if metadata.get("target_commit_percent") == HIGH_PLATEAU_PERCENT:
        near = [(stamp, ratio) for stamp, ratio in readings if injection <= stamp < stop]
        metadata["observed_seconds_near_target_commit"] = sum(
            (right_time - left_time).total_seconds()
            for (left_time, left_ratio), (right_time, right_ratio) in zip(near, near[1:])
            if all(abs(value - HIGH_PLATEAU_PERCENT) <= 0.5 for value in (left_ratio, right_ratio))
            and 0 < (right_time - left_time).total_seconds() <= 2 * MEMORY_SAMPLE_SECONDS
        )


def harvest_surface_run(rundir, remote_csv, metadata):
    # Keep incomplete telemetry outside the Git worktree.
    with tempfile.TemporaryDirectory(prefix=".crashlab-surface-", dir=os.path.dirname(RUNS_DIR)) as staging:
        memory_csv = os.path.join(staging, "memory.csv")
        copy_remote(remote_csv.replace("\\", "/"), memory_csv, checked=True)
        with open(memory_csv, newline="") as stream:
            samples = csv.DictReader(stream)
            columns = tuple(samples.fieldnames or ())
            if columns not in (MEMORY_COLUMNS, MEMORY_COLUMNS + PROBE_COLUMNS):
                raise RuntimeError("Surface memory CSV has unexpected columns")
            observed = [
                parse_utc(row["timestamp_utc"]) for row in samples
                if row.get("timestamp_utc") and all((row.get(key) or "").isdigit() for key in MEMORY_COLUMNS[1:])
            ]
        start = parse_utc(metadata["observation_start_utc"])
        end_value = metadata["observation_end_utc"]
        # Diagnostic runs end at the last durable sample, even when the OS clock
        # was unreachable at the workload deadline.
        if end_value is None:
            if not observed:
                raise RuntimeError("Surface memory CSV contains no verified samples")
            end_value = observed[-1].isoformat()
            metadata["observation_end_utc"] = end_value
        end = parse_utc(end_value)
        observed = [sample for sample in observed if sample <= end]
        training_run = metadata["status"] in ("healthy", "died")
        if (
            not observed
            or observed[0] != start
            or (training_run and observed[-1] - start < datetime.timedelta(minutes=5))
            or any(a >= b for a, b in zip(observed, observed[1:]))
        ):
            raise RuntimeError("Surface memory CSV contains insufficient verified samples")
        if metadata["status"] == "healthy":
            if end - observed[-1] > datetime.timedelta(seconds=2 * MEMORY_SAMPLE_SECONDS):
                raise RuntimeError("Surface memory sampling ended before the healthy control finished")
        try:
            collect_event_exports(staging, checked=True)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            if training_run:
                raise
            metadata["event_export_error"] = str(error)
            print(f"  saved diagnostic memory trace without Windows event exports: {error}", flush=True)
        write_run_metadata(staging, metadata)
        if os.path.exists(rundir):
            raise RuntimeError(f"Run directory already exists: {rundir}")
        os.rename(staging, rundir)
    ok, detail = ps(f"Remove-Item -LiteralPath '{remote_csv}'")
    if not ok:
        print(f"  Surface memory CSV cleanup failed: {detail}", flush=True)


def open_task_manager():
    ok, result = ps(
        "$user = (Get-CimInstance Win32_ComputerSystem).UserName; "
        "if (-not $user) { 'no Windows desktop is signed in'; exit 0 }; "
        "$action = New-ScheduledTaskAction -Execute (Join-Path $env:SystemRoot 'System32\\Taskmgr.exe'); "
        "$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive; "
        "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero); "
        "Register-ScheduledTask -TaskName 'CrashLab-TaskManager' -Action $action "
        "-Principal $principal -Settings $settings -Force | Out-Null; "
        "Start-ScheduledTask -TaskName 'CrashLab-TaskManager'; 'launch requested'"
    )
    message = result if ok else f"launch failed: {result}"
    print(f"  Surface Task Manager: {message}", flush=True)


def observe_cycle(i, boot0, *, surface_data, healthy_control):
    baseline = FIRST_BASELINE_MIN if i == 1 else random.randint(BASELINE_MIN, BASELINE_MAX)
    if healthy_control:
        print(f"  healthy control {HEALTHY_CONTROL_MIN} min", flush=True)
        time.sleep(HEALTHY_CONTROL_MIN * 60)
        end = surface_utc_now()
        if boot_time(required=True) != boot0:
            raise RuntimeError("Surface rebooted during the healthy control")
        return HEALTHY_CONTROL_MIN, None, boot0, end, False, True

    print(f"  baseline {baseline} min", flush=True)
    time.sleep(baseline * 60)
    task_name = f"CrashLab-{uuid.uuid4().hex}"
    inject = surface_utc_now() if surface_data else datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        pid = start_workload(task_name)
        print(f"  injected {FAULT} (Surface PID {pid})", flush=True)
        if surface_data:
            boot_after, lost_contact = wait_for_surface_workload(boot0)
        else:
            boot_after = "rebooted" if wait_until(lambda: rebooted_since(boot0), MAX_RUN_MIN * 60, poll_s=20) else boot0
            lost_contact = False
    finally:
        cleaned = remove_workload_task(task_name)
    if surface_data and boot_after and not cleaned:
        cleaned = wait_until(lambda: remove_workload_task(task_name), 60)
    return baseline, inject, boot_after, None, lost_contact, cleaned


def wait_for_surface_workload(initial_boot):
    current_boot = initial_boot
    lost_contact = False

    def poll_boot():
        nonlocal current_boot, lost_contact
        current_boot = verified_boot_time()
        if not current_boot:
            lost_contact = True
        return current_boot

    wait_until(lambda: poll_boot() not in (None, initial_boot), MAX_RUN_MIN * 60, poll_s=20)
    if current_boot is None:
        print("  Surface stopped responding; waiting for Windows to recover", flush=True)
        wait_until(poll_boot, BOOT_WAIT_MIN * 60)
    return current_boot, lost_contact


def verified_boot_time():
    current_boot = boot_time(timeout=15)
    try:
        if current_boot and datetime.datetime.fromisoformat(current_boot).tzinfo:
            return current_boot
    except ValueError:
        pass
    return None


def verified_crash_time(marker, previous_marker, inject, boot_after):
    if not marker or marker == previous_marker:
        return None
    try:
        death = datetime.datetime.fromtimestamp(int(marker), datetime.timezone.utc)
        reboot = datetime.datetime.fromisoformat(boot_after)
        if parse_utc(inject) < death <= reboot:
            return death
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    return None


def rebooted_since(initial_boot):
    current_boot = boot_time()
    return current_boot is not None and current_boot != initial_boot


def one_cycle(i, *, surface_data=False, healthy_control=False):
    fault = "healthy-control" if healthy_control else FAULT
    runid  = f"laptop_{datetime.datetime.now():%Y%m%d_%H%M%S}_{fault}"
    rundir = os.path.join(RUNS_DIR, runid)
    print(f"\n=== cycle {i}: {runid} ===", flush=True)

    # 1. machine up?
    if not wait_until(reachable, BOOT_WAIT_MIN * 60):
        if surface_data:
            raise RuntimeError("Surface is not reachable over SSH")
        print("  not reachable, skipping", flush=True)
        return
    boot0  = boot_time(required=True)
    death0 = read_last_death()
    print(f"  up  boot={boot0}", flush=True)

    # Start the sampler before baseline and stop it before any forced reset.
    sampler_task = f"CrashLab-Memory-{uuid.uuid4().hex}"
    remote_csv = rf"C:\CrashLab\{sampler_task}.csv"
    try:
        start = start_memory_sampling(sampler_task, remote_csv) if surface_data else None
        baseline, inject, boot_after, end, lost_contact, workload_cleaned = observe_cycle(
            i, boot0, surface_data=surface_data, healthy_control=healthy_control,
        )
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        if surface_data:
            raise RuntimeError(f"{error}; telemetry retained at {remote_csv}") from error
        raise
    finally:
        if surface_data:
            sampler_cleaned = remove_workload_task(sampler_task)

    if surface_data and boot_after is None:
        raise RuntimeError(f"Surface did not recover; telemetry retained at {remote_csv}")
    did_reboot = boot_after != boot0
    if not surface_data:
        status = "died" if did_reboot else "survived"
        print(f"  {status}", flush=True)
        if not did_reboot:
            ps("Restart-Computer -Force", timeout=20)
            did_reboot = wait_until(lambda: rebooted_since(boot0), BOOT_WAIT_MIN * 60)

    recovered = wait_until(reachable, BOOT_WAIT_MIN * 60)
    if surface_data and not recovered:
        raise RuntimeError(f"Surface did not recover; telemetry retained at {remote_csv}")
    if healthy_control and boot_time(required=True) != boot0:
        raise RuntimeError("Surface rebooted before the healthy control could be harvested")
    death = read_last_death() if not healthy_control else None
    got_death = bool(boot_after != boot0 and death and death != death0)
    if surface_data:
        crash_time = verified_crash_time(death, death0, inject, boot_after) if did_reboot else None
        got_death = crash_time is not None
        if got_death:
            end = crash_time.isoformat()
            status = "died"
        elif healthy_control:
            status = "healthy"
        elif did_reboot:
            status = "reboot_unverified"
        else:
            status = "hang_recovered" if lost_contact else "survived"
        print(f"  {status}", flush=True)

    if did_reboot and recovered:
        open_task_manager()
    # 7. run.json in the exact shape load_run expects
    meta = {
        "run_id": runid,
        "machine": MACHINE,
        "fault": fault,
        "fault_cmd": None if healthy_control else f"{TOOL} {FAULT_ARG}",
        "status": status,
        "baseline_minutes": baseline,
        "boot_time_local": boot0,                # carries the laptop's UTC offset
        "inject_time_utc": inject,
        "died": bool(got_death),
        "death_unix": int(death) if got_death else None,
        "death_time_utc": (datetime.datetime.fromtimestamp(
            int(death), datetime.timezone.utc).isoformat() if got_death else None),
    }
    if surface_data:
        try:
            ttf = (parse_utc(end) - parse_utc(inject)).total_seconds() if got_death else None
            meta.update(observation_start_utc=start, observation_end_utc=end, ttf_seconds=ttf)
            harvest_surface_run(rundir, remote_csv, meta)
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
            raise RuntimeError(f"{error}; run {runid}, Surface telemetry at {remote_csv}") from error
    else:
        os.makedirs(rundir, exist_ok=True)
        collect_event_exports(rundir)
        write_run_metadata(rundir, meta)
    print(f"  harvested -> {runid}  (died={got_death})", flush=True)

    if surface_data:
        if not workload_cleaned:
            raise RuntimeError(f"Saved {runid}, but could not verify cleanup of the Surface workload")
        if not sampler_cleaned:
            raise RuntimeError(f"Saved {runid}, but could not verify cleanup of Surface sampler {sampler_task}")
        if not healthy_control and not did_reboot:
            print("  restarting Surface for the next run", flush=True)
            ps("Restart-Computer -Force", timeout=20)
            if not wait_until(lambda: verified_boot_time() not in (None, boot0), BOOT_WAIT_MIN * 60):
                raise RuntimeError(f"Saved {runid}, but the Surface restart could not be verified")
            if not wait_until(reachable, BOOT_WAIT_MIN * 60):
                raise RuntimeError(f"Saved {runid}, but SSH did not recover after restart")
            open_task_manager()

    if GIT_PUSH:
        if surface_data:
            git_sync(f"{runid} {status}", runid)
        else:
            git_sync(f"{runid} {status}")


def main():
    parser = argparse.ArgumentParser(description="Collect CrashLab runs from the Surface.")
    parser.add_argument("cycles", type=int, nargs="?", default=None)
    parser.add_argument("--surface-data", action="store_true", help="Record Windows memory counters")
    parser.add_argument("--healthy-control", action="store_true", help="Collect one 12-minute healthy run")
    parser.add_argument("--responsiveness", action="store_true", help="Collect independent Surface responsiveness telemetry")
    parser.add_argument("--probe-timeout-ms", type=int, default=None,
                        help="Primary PowerShell observation cap in milliseconds (4000)")
    parser.add_argument("--plateau-percent", type=int, choices=(*range(75, 81), HIGH_PLATEAU_PERCENT),
                        help="Collect one capped plateau at 75–80%% or a supervised 96%%")
    parser.add_argument("--ramp", nargs="?", type=int, const=10, choices=RAMP_LIMIT_MINUTES,
                        metavar="MiB/s", help="Collect one supervised 5, 10 or 15 MiB/s ramp (default 10)")
    parser.add_argument("--stepped", choices=STEPPED_PATTERNS, help="Collect a protocol stepped ramp")
    parser.add_argument("--burst-seed", type=int, help="Seed the realistic healthy workload for protocol v3")
    run_type = parser.add_mutually_exclusive_group()
    run_type.add_argument("--protocol-v3", action="store_true", help="Restart and collect one frozen-protocol replicate")
    run_type.add_argument("--pilot", action="store_true", help="Collect a pilot without a freeze check or restart")
    args = parser.parse_args()
    cycles = args.cycles if args.cycles is not None else (1 if args.healthy_control or args.responsiveness else 10)
    if cycles < 1:
        parser.error("cycles must be positive")
    if args.healthy_control and (not args.surface_data or cycles != 1):
        parser.error("--healthy-control requires --surface-data and exactly one cycle")
    if (args.stepped is not None or args.burst_seed is not None) and not (args.protocol_v3 or args.pilot):
        parser.error("--stepped and --burst-seed require --protocol-v3 or --pilot")
    if args.burst_seed is not None and not args.healthy_control:
        parser.error("--burst-seed requires --healthy-control")
    if args.protocol_v3 and args.healthy_control and args.burst_seed is None:
        parser.error("--protocol-v3 healthy control requires --burst-seed")
    if args.protocol_v3 and args.plateau_percent not in (None, HIGH_PLATEAU_PERCENT):
        parser.error("--protocol-v3 plateau must target 96 percent")
    if args.responsiveness:
        modes = sum((args.healthy_control, args.plateau_percent is not None, args.ramp is not None,
                     args.stepped is not None))
        if not args.surface_data or cycles != 1 or modes != 1:
            parser.error(
                "--responsiveness requires --surface-data, exactly one cycle, and exactly one of "
                "--healthy-control, --plateau-percent, --ramp, or --stepped"
            )
        if args.probe_timeout_ms not in (None, RESPONSIVENESS_TIMEOUT_MS):
            parser.error("--probe-timeout-ms must be 4000 for independent probe runs")
        collect_responsiveness_run(
            timeout_ms=RESPONSIVENESS_TIMEOUT_MS, plateau_percent=args.plateau_percent, ramp_rate=args.ramp,
            stepped=args.stepped, burst_seed=args.burst_seed, protocol_v3=args.protocol_v3, pilot=args.pilot,
        )
        return
    if args.probe_timeout_ms is not None or args.plateau_percent is not None or args.ramp or args.protocol_v3 or args.pilot:
        parser.error("--probe-timeout-ms, --plateau-percent, --ramp, --protocol-v3, and --pilot require --responsiveness")
    print(f"crashloop: {cycles} cycles -> {RUNS_DIR}", flush=True)
    for i in range(1, cycles + 1):
        try:
            one_cycle(i, surface_data=args.surface_data, healthy_control=args.healthy_control)
        except Exception as e:
            if args.surface_data:
                raise SystemExit(f"Surface cycle {i} failed: {e}") from e
            print(f"  cycle error: {e}", flush=True)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
