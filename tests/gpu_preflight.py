from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

COMMAND_TIMEOUT_SECONDS = 5.0
AMD_VENDOR_ID = "0x1002"
_TARGET_ALIASES: dict[str, str] = {
    "all": "all",
    "cpu": "cpu",
    "x86": "cpu",
    "x86-auto": "cpu",
    "x86-avx2": "cpu",
    "x86-avx512": "cpu",
    "nvidia": "nvidia-dgpu",
    "nvptx": "nvidia-dgpu",
    "nvidia-dgpu": "nvidia-dgpu",
    "nvidia_dgpu": "nvidia-dgpu",
    "amd-igpu": "amd-igpu",
    "amd_igpu": "amd-igpu",
    "amdgcn": "amd-igpu",
}
_PHYSICAL_NVIDIA_LINE = re.compile(r"^GPU\s+\d+:\s+(.+?)(?:\s+\(UUID:|\s*$)")
_DMESG_TIMESTAMP = re.compile(r"^\[?(\d{4}-\d{2}-\d{2}T[^\]]+)\]?\s*(.*)$")


@dataclass
class GpuCheckResult:
    ok: bool
    device_name: str = ""
    driver_version: str = ""
    diagnostics: str = ""


@dataclass
class DmesgCheckResult:
    clean: bool
    warnings: list[str] = field(default_factory=list)
    permission_denied: bool = False


@dataclass
class GpuPreflightReport:
    target: str
    overall_ok: bool
    nvidia: GpuCheckResult | None = None
    amd: GpuCheckResult | None = None
    dmesg: DmesgCheckResult | None = None
    summary: str = ""


@dataclass
class _CommandResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    diagnostics: str = ""


def _run_command(command: Sequence[str]) -> _CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return _CommandResult(ok=False, diagnostics=f"command not found: {command[0]}")
    except subprocess.TimeoutExpired:
        return _CommandResult(
            ok=False,
            diagnostics=f"command timed out after {COMMAND_TIMEOUT_SECONDS:.0f}s: {' '.join(command)}",
        )
    except Exception as exc:
        return _CommandResult(
            ok=False,
            diagnostics=f"{type(exc).__name__}: {exc}",
        )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    ok = completed.returncode == 0
    diagnostics = ""
    if not ok:
        details = stderr or stdout or f"exit code {completed.returncode}"
        diagnostics = f"{' '.join(command)} failed: {details}"
    return _CommandResult(
        ok=ok,
        stdout=stdout,
        stderr=stderr,
        returncode=completed.returncode,
        diagnostics=diagnostics,
    )


def _normalize_target(target: str) -> str:
    normalized = _TARGET_ALIASES.get(target.strip().lower())
    return normalized or target.strip().lower()


def _parse_nvidia_listing(output: str) -> tuple[list[str], str]:
    physical_lines = [line.strip() for line in output.splitlines() if line.strip().startswith("GPU ")]
    device_name = ""
    if physical_lines:
        match = _PHYSICAL_NVIDIA_LINE.match(physical_lines[0])
        device_name = match.group(1).strip() if match else physical_lines[0]
    return physical_lines, device_name


def _extract_first_value(output: str) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.split(",", 1)[0].strip()
    return ""


def check_nvidia() -> GpuCheckResult:
    listing = _run_command(("nvidia-smi", "-L"))
    if not listing.ok:
        return GpuCheckResult(ok=False, diagnostics=listing.diagnostics)

    gpu_lines, device_name = _parse_nvidia_listing(listing.stdout)
    if not gpu_lines:
        return GpuCheckResult(
            ok=False,
            diagnostics="nvidia-smi -L returned no physical NVIDIA GPUs",
        )

    telemetry = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader",
        )
    )
    if not telemetry.ok:
        return GpuCheckResult(
            ok=False,
            device_name=device_name,
            diagnostics=telemetry.diagnostics,
        )

    driver = _run_command(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    )
    driver_version = _extract_first_value(driver.stdout) if driver.ok else ""

    first_row = next((line.strip() for line in telemetry.stdout.splitlines() if line.strip()), "")
    diagnostics_parts = [f"detected {len(gpu_lines)} NVIDIA GPU(s)"]
    if first_row:
        diagnostics_parts.append(f"telemetry={first_row}")
    if not driver.ok and driver.diagnostics:
        diagnostics_parts.append(f"driver query failed: {driver.diagnostics}")

    return GpuCheckResult(
        ok=True,
        device_name=device_name,
        driver_version=driver_version,
        diagnostics="; ".join(diagnostics_parts),
    )


def _parse_rocminfo_gpu(output: str) -> tuple[bool, str, str]:
    current: dict[str, str] = {}
    gpu_blocks: list[dict[str, str]] = []

    def flush_current() -> None:
        if current:
            gpu_blocks.append(current.copy())
            current.clear()

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Agent ") and current:
            flush_current()
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    flush_current()

    only_gpu_blocks = [
        block for block in gpu_blocks if block.get("Device Type", "").upper() == "GPU"
    ]
    if not only_gpu_blocks:
        return False, "", "rocminfo found no GPU agent"

    preferred_block = only_gpu_blocks[0]
    for block in only_gpu_blocks:
        memory_properties = block.get("Memory Properties", "")
        device_name = block.get("Marketing Name") or block.get("Name", "")
        if any(
            token in f"{memory_properties} {device_name}".upper()
            for token in ("APU", "IGPU", "INTEGRATED")
        ):
            preferred_block = block
            break

    memory_properties = preferred_block.get("Memory Properties", "")
    device_name = preferred_block.get("Marketing Name") or preferred_block.get("Name", "")
    looks_integrated = any(
        token in f"{memory_properties} {device_name}".upper()
        for token in ("APU", "IGPU", "INTEGRATED")
    )
    diagnostics = f"rocminfo GPU agent: {device_name or 'unnamed GPU'}"
    if memory_properties:
        diagnostics = f"{diagnostics}; memory={memory_properties}"
    if not looks_integrated:
        diagnostics = (
            f"{diagnostics}; scanned {len(only_gpu_blocks)} GPU agent(s); "
            "no explicit integrated-GPU marker found"
        )
    return looks_integrated, device_name, diagnostics


def _parse_rocm_smi_gpu(output: str) -> tuple[bool, str, str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return False, "", "rocm-smi returned no output"
    joined = " ".join(lines)
    looks_amd = any(token in joined.lower() for token in ("amdgpu", "radeon", "gfx"))
    looks_integrated = any(token in joined.lower() for token in ("apu", "igpu", "integrated"))
    device_name = ""
    for line in lines:
        if "card series" in line.lower() or "market name" in line.lower():
            device_name = line.split(":", 1)[-1].strip()
            break
    diagnostics = f"rocm-smi detected AMD GPU output ({len(lines)} line(s))"
    if not looks_integrated:
        diagnostics = f"{diagnostics}; no explicit integrated-GPU marker found"
    return looks_amd and looks_integrated, device_name, diagnostics


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _check_amd_sysfs() -> GpuCheckResult:
    drm_root = pathlib.Path("/sys/class/drm")
    if not drm_root.is_dir():
        return GpuCheckResult(ok=False, diagnostics="/sys/class/drm is not available")

    for vendor_path in sorted(drm_root.glob("card*/device/vendor")):
        vendor = _read_text(vendor_path).lower()
        if vendor != AMD_VENDOR_ID:
            continue

        device_root = vendor_path.parent
        pci_class = _read_text(device_root / "class").lower()
        device_id = _read_text(device_root / "device")
        uevent = _read_text(device_root / "uevent")
        looks_integrated = pci_class.startswith("0x038") or "APU" in uevent.upper()
        diagnostics = [
            f"sysfs detected AMD GPU at {device_root.parent.name}",
            f"vendor={vendor}",
        ]
        if pci_class:
            diagnostics.append(f"class={pci_class}")
        if device_id:
            diagnostics.append(f"device={device_id}")
        if not looks_integrated:
            diagnostics.append("no clear integrated-GPU marker in sysfs")

        return GpuCheckResult(
            ok=looks_integrated,
            device_name=device_root.parent.name,
            diagnostics="; ".join(diagnostics),
        )

    return GpuCheckResult(ok=False, diagnostics="no AMD GPU vendor ID found in /sys/class/drm")


def check_amd_igpu() -> GpuCheckResult:
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is not None:
        rocm_result = _run_command((rocm_smi,))
        if rocm_result.ok:
            ok, device_name, diagnostics = _parse_rocm_smi_gpu(rocm_result.stdout)
            if ok:
                return GpuCheckResult(ok=True, device_name=device_name, diagnostics=diagnostics)

    rocminfo = shutil.which("rocminfo")
    if rocminfo is not None:
        rocminfo_result = _run_command((rocminfo,))
        if rocminfo_result.ok:
            ok, device_name, diagnostics = _parse_rocminfo_gpu(rocminfo_result.stdout)
            if ok:
                return GpuCheckResult(ok=True, device_name=device_name, diagnostics=diagnostics)

    sysfs_result = _check_amd_sysfs()
    if sysfs_result.ok:
        return sysfs_result

    diagnostics_parts: list[str] = []
    if rocm_smi is not None:
        diagnostics_parts.append("rocm-smi did not confirm an integrated AMD GPU")
    else:
        diagnostics_parts.append("rocm-smi not found")
    if rocminfo is not None:
        diagnostics_parts.append("rocminfo did not confirm an integrated AMD GPU")
    else:
        diagnostics_parts.append("rocminfo not found")
    diagnostics_parts.append(sysfs_result.diagnostics)
    return GpuCheckResult(
        ok=False,
        device_name=sysfs_result.device_name,
        diagnostics="; ".join(diagnostics_parts),
    )


def _looks_like_permission_denied(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "permission denied",
            "operation not permitted",
            "not permitted",
            "access denied",
            "insufficient permissions",
            "kernel buffer failed",
            "kernel buffer is inaccessible",
            "izin verilmedi",
        )
    )


def _kernel_warning_from_line(line: str) -> str | None:
    lowered = line.lower()
    if "xid" in lowered:
        return line.strip()
    if re.search(r"amdgpu.*error", lowered):
        return line.strip()
    if "gpu fault" in lowered:
        return line.strip()
    if "page fault" in lowered and any(
        token in lowered for token in ("gpu", "amdgpu", "nvrm", "nvidia", "radeon", "gfx")
    ):
        return line.strip()
    return None


def _parse_dmesg_timestamp(raw_timestamp: str) -> datetime | None:
    cleaned = raw_timestamp.strip()
    if cleaned.count(",") == 1 and "." not in cleaned.split(",", 1)[0]:
        cleaned = cleaned.replace(",", ".", 1)
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _filter_dmesg_lines(output: str, seconds_back: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds_back)
    filtered: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _DMESG_TIMESTAMP.match(line)
        if match is None:
            continue
        timestamp = _parse_dmesg_timestamp(match.group(1))
        if timestamp is None or timestamp < cutoff:
            continue
        filtered.append(match.group(2).strip())
    return filtered


def _scan_kernel_output(lines: Sequence[str]) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for line in lines:
        warning = _kernel_warning_from_line(line)
        if warning is None or warning in seen:
            continue
        warnings.append(warning)
        seen.add(warning)
    return warnings


def check_dmesg_health(seconds_back: int = 60) -> DmesgCheckResult:
    journal_result = _run_command(
        (
            "journalctl",
            "-k",
            "--since",
            f"{seconds_back} seconds ago",
            "--no-pager",
            "-q",
        )
    )
    if journal_result.ok:
        warnings = _scan_kernel_output(journal_result.stdout.splitlines())
        return DmesgCheckResult(clean=not warnings, warnings=warnings)

    permission_denied = _looks_like_permission_denied(
        f"{journal_result.stdout}\n{journal_result.stderr}\n{journal_result.diagnostics}"
    )
    dmesg_result = _run_command(("dmesg", "--time-format", "iso"))
    if dmesg_result.ok:
        recent_lines = _filter_dmesg_lines(dmesg_result.stdout, seconds_back)
        warnings = _scan_kernel_output(recent_lines)
        return DmesgCheckResult(
            clean=not warnings,
            warnings=warnings,
            permission_denied=permission_denied,
        )

    permission_denied = permission_denied or _looks_like_permission_denied(
        f"{dmesg_result.stdout}\n{dmesg_result.stderr}\n{dmesg_result.diagnostics}"
    )
    if permission_denied:
        return DmesgCheckResult(clean=True, warnings=[], permission_denied=True)

    failure_notes = [note for note in (journal_result.diagnostics, dmesg_result.diagnostics) if note]
    return DmesgCheckResult(
        clean=False,
        warnings=[
            "kernel log health check failed: "
            + ("; ".join(failure_notes) if failure_notes else "unable to read journalctl or dmesg")
        ],
        permission_denied=False,
    )


def run_full_preflight(target: str) -> GpuPreflightReport:
    normalized_target = _normalize_target(target)
    if normalized_target == "cpu":
        return GpuPreflightReport(
            target="cpu",
            overall_ok=True,
            summary="CPU target selected: GPU preflight skipped.",
        )

    dmesg_result: DmesgCheckResult | None = None
    nvidia_result: GpuCheckResult | None = None
    amd_result: GpuCheckResult | None = None
    required_checks: list[bool] = []

    if normalized_target in {"nvidia-dgpu", "all"}:
        nvidia_result = check_nvidia()
        required_checks.append(nvidia_result.ok)
    if normalized_target in {"amd-igpu", "all"}:
        amd_result = check_amd_igpu()
        required_checks.append(amd_result.ok)
    if normalized_target in {"nvidia-dgpu", "amd-igpu", "all"}:
        dmesg_result = check_dmesg_health()

    overall_ok = all(required_checks) if required_checks else False
    advisory = (
        dmesg_result is not None
        and (not dmesg_result.clean or dmesg_result.permission_denied)
    )
    if overall_ok and not advisory:
        summary = f"{normalized_target} preflight passed."
    elif overall_ok:
        if dmesg_result is not None and dmesg_result.permission_denied and dmesg_result.clean:
            summary = f"{normalized_target} preflight passed with advisory-only kernel log access limits."
        else:
            summary = f"{normalized_target} preflight passed with advisory kernel warnings."
    else:
        failed_targets: list[str] = []
        if nvidia_result is not None and not nvidia_result.ok:
            failed_targets.append("nvidia")
        if amd_result is not None and not amd_result.ok:
            failed_targets.append("amd-igpu")
        summary = f"{normalized_target} preflight failed: {', '.join(failed_targets) or 'no checks completed'}."

    return GpuPreflightReport(
        target=normalized_target,
        overall_ok=overall_ok,
        nvidia=nvidia_result,
        amd=amd_result,
        dmesg=dmesg_result,
        summary=summary,
    )


def _result_status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _format_gpu_section(label: str, result: GpuCheckResult | None) -> list[str]:
    if result is None:
        return []
    lines = [f"{label}: {_result_status(result.ok)}"]
    if result.device_name:
        lines.append(f"  device: {result.device_name}")
    if result.driver_version:
        lines.append(f"  driver: {result.driver_version}")
    if result.diagnostics:
        lines.append(f"  diagnostics: {result.diagnostics}")
    return lines


def _format_dmesg_section(result: DmesgCheckResult | None) -> list[str]:
    if result is None:
        return []
    if result.permission_denied and result.clean:
        status = "ADVISORY"
    else:
        status = "PASS" if result.clean else "ADVISORY"
    lines = [f"kernel log health: {status}"]
    if result.permission_denied:
        lines.append("  diagnostics: kernel log access was denied; results are advisory only")
    if result.warnings:
        lines.append("  warnings:")
        for warning in result.warnings:
            lines.append(f"    - {warning}")
    return lines


def _compute_exit_code(report: GpuPreflightReport) -> int:
    advisory = report.dmesg is not None and (
        not report.dmesg.clean or report.dmesg.permission_denied
    )
    if not report.overall_ok:
        return 1
    if advisory:
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU benchmark pre-flight health checker")
    parser.add_argument(
        "--target",
        default="all",
        help=(
            "Target to validate: nvidia, nvidia-dgpu, amd-igpu, amdgcn, cpu, or all "
            "(default: all)"
        ),
    )
    args = parser.parse_args(argv)

    normalized_target = _normalize_target(args.target)
    if normalized_target not in {"nvidia-dgpu", "amd-igpu", "cpu", "all"}:
        print(
            f"Unsupported target '{args.target}'. Expected one of: nvidia, amd-igpu, cpu, all.",
            file=sys.stderr,
        )
        return 1

    report = run_full_preflight(args.target)
    lines = [
        f"GPU preflight target: {report.target}",
        f"summary: {report.summary}",
    ]
    lines.extend(_format_gpu_section("nvidia", report.nvidia))
    lines.extend(_format_gpu_section("amd-igpu", report.amd))
    lines.extend(_format_dmesg_section(report.dmesg))
    print("\n".join(lines))
    return _compute_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
