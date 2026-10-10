#!/usr/bin/env bash
# CMP 170HX 64GB（解锁后）验收脚本。
# 发给卖家，在他的 Ubuntu 上运行。只依赖 NVIDIA 驱动；有 PyTorch 时会额外做 Tensor Core 矩阵乘。
#
# 用法:
#   bash test-170hx-64g.sh
#   bash test-170hx-64g.sh --gpu 0
#   bash test-170hx-64g.sh --stress 600
#   bash test-170hx-64g.sh --quick
#   TORCH_PYTHON=~/ComfyUI/venv/bin/python bash test-170hx-64g.sh
#
# 通过条件:
#   PCI 10de:20c2（8GB 规格）解锁后 nvidia-smi 约 65536 MiB，
#   并且至少 52 GiB 显存能正确读写。计算、带宽、ECC、温度一并记录。
#
# 退出码: 0 通过（可以有警告）  1 失败  2 没测完  130 被中断
# 请把生成的日志原样发回，不要只发最后一行。

if [ -z "${BASH_VERSION:-}" ]; then
  echo "请用 bash 运行: bash test-170hx-64g.sh"
  exit 1
fi

VERSION="2026-10-10"
GPU="all"
STRESS=""
QUICK=0
FORCE=0
SELFTEST=0

usage() {
  cat <<'EOF'
用法: bash test-170hx-64g.sh [选项]

选项:
  --gpu N        只测第 N 张卡（nvidia-smi 的编号，默认自动找 170HX / 64GB）
  --stress SEC   计算加压秒数，默认 180。设为 0 则跳过加压
  --quick        加压缩短到 20 秒，显存全量校验仍会做
  --force        显卡上已有程序时也继续测（可能把对方任务挤爆）
  -h, --help     显示本说明

环境变量:
  TORCH_PYTHON   指定已安装 CUDA 版 PyTorch 的 python，例如 ComfyUI 的 venv

说明:
  测试会占满被测 GPU，大约 4 到 8 分钟。请先停掉挖矿、ComfyUI 和其他占卡程序。
  CMP 170HX 通常没有自带风扇，靠机箱风道散热，风扇读数为 N/A 是正常的。
  日志写在当前目录，文件名 170hx-64g-时间.log
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)
      GPU="${2:-}"
      shift 2
      ;;
    --stress)
      STRESS="${2:-}"
      shift 2
      ;;
    --quick)
      QUICK=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --self-test)
      SELFTEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

if [ "$GPU" != "all" ]; then
  case "$GPU" in
    ''|*[!0-9]*)
      echo "错误: --gpu 需要数字，例如 --gpu 0"
      exit 1
      ;;
  esac
fi

if [ -n "$STRESS" ]; then
  case "$STRESS" in
    ''|*[!0-9]*)
      echo "错误: --stress 需要非负整数秒"
      exit 1
      ;;
  esac
else
  if [ "$QUICK" -eq 1 ]; then
    STRESS=20
  else
    STRESS=180
  fi
fi

pick_python() {
  cand=""
  if [ -n "${TORCH_PYTHON:-}" ]; then
    if [ ! -x "$TORCH_PYTHON" ]; then
      echo "TORCH_PYTHON 不可执行: $TORCH_PYTHON" >&2
      return 1
    fi
    printf '%s\n' "$TORCH_PYTHON"
    return 0
  fi
  _cands=""
  if [ -x "$HOME/ComfyUI/venv/bin/python" ]; then
    _cands="$HOME/ComfyUI/venv/bin/python"
  fi
  # /home 在 macOS 上可能是自动挂载点，直接展开会卡住。只在它是普通目录时查找。
  if [ ! -L /home ] && [ -d /home ]; then
    for cand in /home/*/ComfyUI/venv/bin/python; do
      if [ -x "$cand" ]; then
        _cands="$_cands $cand"
      fi
    done
  fi
  if command -v python3 >/dev/null 2>&1; then
    _cands="$_cands $(command -v python3)"
  fi
  for cand in $_cands; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then
      if "$cand" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
        printf '%s\n' "$cand"
        return 0
      fi
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

run_py() {
  "$PY" -u - <<'PY'
from __future__ import annotations

import ctypes
import gc
import math
import os
import re
import shutil
import struct
import subprocess
import time
import traceback
MEM_64_MIN_MIB = 60000
MEM_64_MAX_MIB = 70000
MEM_TEST_MIN_BYTES = 52 * (1024 ** 3)
CHUNK_BYTES = 1 * (1024 ** 3)
RESERVE_BYTES = int(1.5 * (1024 ** 3))
GIB = 1024 ** 3
SLICE_ELEMS = 1 << 26  # 256 MiB of int32
BW_WARN_GBS = 800.0
BW_FAIL_GBS = 400.0
NA = {"", "N/A", "[N/A]", "[NOT SUPPORTED]", "NOT SUPPORTED", "UNKNOWN"}


class Report:
    def __init__(self, gpu: int):
        self.gpu = gpu
        self.fails: list[str] = []
        self.warns: list[str] = []
        self.stats: dict[str, str] = {}

    def fail(self, msg: str) -> None:
        self.fails.append(msg)
        print(f"[失败] {msg}", flush=True)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)
        print(f"[警告] {msg}", flush=True)

    def ok(self, msg: str) -> None:
        print(f"[通过] {msg}", flush=True)

    def info(self, msg: str) -> None:
        print(f"[信息] {msg}", flush=True)

    @property
    def status(self) -> str:
        if self.fails:
            return "失败"
        if self.warns:
            return "通过（有警告）"
        return "通过"

    def finish(self) -> None:
        print(f"\n===== GPU {self.gpu} 结论: {self.status} =====", flush=True)
        if self.fails:
            print("失败项:", flush=True)
            for msg in self.fails:
                print(f"  - {msg}", flush=True)
        if self.warns:
            print("警告项:", flush=True)
            for msg in self.warns:
                print(f"  - {msg}", flush=True)
        if self.stats:
            bits = " ".join(f"{k}={v}" for k, v in self.stats.items())
            print(f"##RESULT gpu={self.gpu} status={self.status} {bits} fails={len(self.fails)} warns={len(self.warns)}", flush=True)


def section(title: str) -> None:
    print(f"\n----- {title} -----", flush=True)


def run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)


def num(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text.upper() in NA:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_active(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip().lower()
    if "not active" in text:
        return False
    return "active" in text


def norm_id(value: str | None) -> str:
    if value is None:
        return ""
    text = value.strip().upper()
    if text in NA:
        return ""
    return text.replace(" ", "")


def salt_bits(index: int) -> int:
    return (0xA5A50000 ^ ((index * 0x9E3779B1) & 0xFFFFFFFF)) & 0xFFFFFFFF


def as_i32(bits: int) -> int:
    bits &= 0xFFFFFFFF
    if bits >= 0x80000000:
        return bits - 0x100000000
    return bits


def split_csv(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


def judge_identity(name: str, device_id: str, memory_mib: float | None) -> tuple[list[str], list[str], list[str]]:
    fails: list[str] = []
    warns: list[str] = []
    infos: list[str] = []
    did = norm_id(device_id)
    is_20c2 = "20C2" in did
    is_2082 = "2082" in did and not is_20c2

    if memory_mib is None:
        fails.append("读不到显存容量")
    else:
        mem = memory_mib
        shown = f"{mem:.0f} MiB（{mem / 1024:.1f} GiB）"
        if is_2082 and 36000 <= mem <= 48000:
            fails.append(f"这是 10GB 规格（PCI 2082），解锁后是 {shown}，属于 40GB 档，不是 64GB")
        elif is_20c2 and mem < 16000:
            fails.append(f"设备是 8GB 规格 20c2，但显存仍是 {shown}，没有解锁到 64GB")
        elif is_20c2 and 36000 <= mem <= 48000:
            fails.append(f"设备是 20c2，但容量只有 {shown}，属于 40GB 档，不是 64GB")
        elif mem < 16000:
            fails.append(f"显存只有 {shown}，仍是出厂小容量，不是解锁后的 64GB")
        elif 16000 <= mem < 36000 or 48000 <= mem < MEM_64_MIN_MIB:
            fails.append(f"显存 {shown}，不到 64GB 档（期望约 65536 MiB）")
        elif mem > MEM_64_MAX_MIB:
            fails.append(f"显存 {shown}，超过 64GB 档，不像是解锁后的 170HX 64GB")
        elif is_20c2:
            infos.append(f"容量 {shown}，设备 ID 含 20c2，符合解锁后的 64GB 170HX")
        elif not did:
            warns.append(f"容量 {shown} 在 64GB 档，但没读到 PCI 设备 ID，请核对下面的 lspci")
        else:
            warns.append(f"容量 {shown} 在 64GB 档，但设备 ID 不是 20c2（读到 {did}）")

    upper = (name or "").upper()
    if "A100" in upper:
        warns.append(f"产品名是「{name}」，不是 CMP 170HX 的常见名称")
    elif upper and "170" not in upper and "CMP" not in upper and not fails:
        warns.append(f"产品名是「{name}」，不像 CMP 170HX，请对照设备 ID")
    return fails, warns, infos


def min_test_bytes(memory_mib: float | None, identity_ok: bool) -> int:
    if identity_ok:
        return MEM_TEST_MIN_BYTES
    if memory_mib is None:
        return 256 * 1024 * 1024
    return max(256 * 1024 * 1024, int(memory_mib * 1024 * 1024 * 0.80))


def smi_multi(gpu: int, fields: list[str]) -> dict[str, str] | None:
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu),
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = run(cmd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[信息] nvidia-smi 查询失败: {exc}", flush=True)
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()
    if not line:
        return None
    parts = split_csv(line[0])
    if len(parts) != len(fields):
        return None
    return dict(zip(fields, parts))


def smi_one(gpu: int, field: str) -> str | None:
    got = smi_multi(gpu, [field])
    if not got:
        return None
    value = got.get(field, "").strip()
    if value.upper() in NA:
        return None
    return value


def lspci_line(bus_id: str) -> str:
    if not bus_id or shutil.which("lspci") is None:
        return ""
    parts = bus_id.split(":")
    if len(parts) < 3:
        return ""
    slot = parts[-2] + ":" + parts[-1]
    try:
        proc = run(["lspci", "-nn", "-s", slot])
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip()


def parse_lspci_id(line: str) -> str:
    match = re.search(r"\[(10de):([0-9a-fA-F]{4})\]", line, re.I)
    if not match:
        return ""
    return "10DE:" + match.group(2).upper()


def read_ecc(gpu: int) -> dict[str, int | None]:
    keys = {
        "vol_c": ["ecc.errors.corrected.volatile.dram", "ecc.errors.corrected.volatile.total"],
        "vol_u": ["ecc.errors.uncorrected.volatile.dram", "ecc.errors.uncorrected.volatile.total"],
        "agg_c": ["ecc.errors.corrected.aggregate.dram", "ecc.errors.corrected.aggregate.total"],
        "agg_u": ["ecc.errors.uncorrected.aggregate.dram", "ecc.errors.uncorrected.aggregate.total"],
    }
    out: dict[str, int | None] = {}
    for key, fields in keys.items():
        out[key] = None
        for field in fields:
            value = smi_one(gpu, field)
            parsed = num(value)
            if parsed is not None:
                out[key] = int(parsed)
                break
    return out


def read_remap(gpu: int) -> dict[str, str]:
    try:
        proc = run(["nvidia-smi", "-i", str(gpu), "-q", "-d", "ROW_REMAPPER,ECC"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"raw": str(exc)}
    text = proc.stdout or proc.stderr or ""
    found: dict[str, str] = {"raw": text}

    def grab(label: str) -> str:
        for line in text.splitlines():
            if label in line and ":" in line:
                return line.split(":", 1)[1].strip()
        return ""

    # Row remapper block uses these labels. ECC block also contains "Correctable",
    # so prefer the remapper-specific failure/pending keys and the first remap counts
    # only when the remapper header exists.
    if "Remapped Rows" in text or "Remapping Failure" in text:
        in_remap = False
        for line in text.splitlines():
            if "Remapped Rows" in line:
                in_remap = True
                continue
            if in_remap and line and not line.startswith(" ") and not line.startswith("\t"):
                break
            if not in_remap or ":" not in line:
                continue
            label, value = line.split(":", 1)
            label = label.strip()
            value = value.strip()
            if label == "Correctable Error":
                found["correctable"] = value
            elif label == "Uncorrectable Error":
                found["uncorrectable"] = value
            elif label == "Pending":
                found["pending"] = value
            elif "Remapping Failure" in label:
                found["failure"] = value
    else:
        found["pending"] = grab("Pending")
        found["failure"] = grab("Remapping Failure Occurred")
    return found


def read_xids() -> list[str] | None:
    commands = [
        ["dmesg", "-T"],
        ["journalctl", "-k", "-n", "400", "--no-pager", "-o", "short"],
        ["sudo", "-n", "dmesg", "-T"],
    ]
    for cmd in commands:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            proc = run(cmd, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0 or not proc.stdout:
            continue
        return [line for line in proc.stdout.splitlines() if "Xid" in line]
    return None


def list_compute_apps(uuid: str) -> list[str]:
    try:
        proc = run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = split_csv(line)
        if len(parts) < 4:
            continue
        if uuid and parts[0] != uuid:
            continue
        rows.append(f"pid={parts[1]} mem={parts[3]} MiB {parts[2]}")
    return rows


def sample_gpu(gpu: int) -> dict[str, float | None]:
    fields = [
        "temperature.gpu",
        "temperature.memory",
        "power.draw",
        "utilization.gpu",
        "clocks.current.graphics",
        "clocks.current.memory",
    ]
    got = smi_multi(gpu, fields) or {}
    return {field: num(got.get(field)) for field in fields}


def fmt_sample(sample: dict[str, float | None]) -> str:
    def show(key: str, suffix: str) -> str:
        value = sample.get(key)
        if value is None:
            return "N/A"
        if suffix == "W" or suffix == "C":
            return f"{value:.0f}{suffix}"
        return f"{value:.0f}{suffix}"

    return (
        f"温度 {show('temperature.gpu', 'C')}  "
        f"显存温度 {show('temperature.memory', 'C')}  "
        f"功耗 {show('power.draw', 'W')}  "
        f"利用率 {show('utilization.gpu', '%')}  "
        f"核心 {show('clocks.current.graphics', 'MHz')}  "
        f"显存频率 {show('clocks.current.memory', 'MHz')}"
    )


def judge_samples(report: Report, samples: list[dict[str, float | None]]) -> None:
    temps = [s["temperature.gpu"] for s in samples if s.get("temperature.gpu") is not None]
    mem_temps = [s["temperature.memory"] for s in samples if s.get("temperature.memory") is not None]
    powers = [s["power.draw"] for s in samples if s.get("power.draw") is not None]
    utils = [s["utilization.gpu"] for s in samples if s.get("utilization.gpu") is not None]
    clocks = [s["clocks.current.graphics"] for s in samples if s.get("clocks.current.graphics") is not None]
    if temps:
        report.stats["max_temp_c"] = f"{max(temps):.0f}"
        peak = max(temps)
        if peak >= 95:
            report.fail(f"核心温度达到 {peak:.0f}°C，这次不能算稳定通过")
        elif peak >= 85:
            report.warn(f"核心温度最高 {peak:.0f}°C，散热偏紧")
        else:
            report.ok(f"核心温度最高 {peak:.0f}°C")
    if mem_temps:
        report.stats["max_mem_temp_c"] = f"{max(mem_temps):.0f}"
        peak = max(mem_temps)
        if peak >= 100:
            report.fail(f"显存温度达到 {peak:.0f}°C")
        elif peak >= 90:
            report.warn(f"显存温度最高 {peak:.0f}°C")
    if powers:
        report.stats["max_power_w"] = f"{max(powers):.0f}"
        report.info(f"加压期间最高功耗 {max(powers):.0f} W")
    if utils and clocks and max(utils) >= 80 and max(clocks) < 500:
        report.warn(f"利用率上去了，但核心频率最高只有 {max(clocks):.0f} MHz，可能在降频")
    limit = num(smi_one(report.gpu, "power.limit"))
    if powers and utils and limit is not None and limit >= 200 and max(utils) >= 90 and max(powers) < 80:
        report.warn(f"满载功耗只有 {max(powers):.0f} W，功耗墙是 {limit:.0f} W，偏低")


def judge_throttle(report: Report, gpu: int) -> None:
    fields = [
        "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.hw_slowdown",
        "clocks_event_reasons.sw_power_cap",
        "clocks_event_reasons.hw_power_brake_slowdown",
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.hw_slowdown",
        "clocks_throttle_reasons.sw_power_cap",
        "clocks_throttle_reasons.hw_power_brake_slowdown",
    ]
    seen = set()
    for field in fields:
        value = smi_one(gpu, field)
        if value is None or not is_active(value):
            continue
        if field in seen:
            continue
        seen.add(field)
        if "thermal" in field:
            report.warn("加压时出现温度降频。可能是机箱风道不够，不一定是卡本身坏，但这次温度余量不足")
        elif "power_brake" in field:
            report.warn("出现 Power Brake。先检查电源功率和供电线，这个信号经常是供电不足，不是核心损坏")
        elif "sw_power_cap" in field:
            report.info("碰到了软件功耗墙，这是正常限电")
        elif "hw_slowdown" in field:
            report.warn("出现硬件降频（hw_slowdown）")


def judge_ecc_delta(report: Report, before: dict[str, int | None], after: dict[str, int | None]) -> None:
    labels = {
        "vol_c": "本次可纠正",
        "vol_u": "本次不可纠正",
        "agg_c": "历史可纠正",
        "agg_u": "历史不可纠正",
    }
    readable = [value for value in before.values() if value is not None]
    if not readable and not any(value is not None for value in after.values()):
        report.warn("读不到 ECC 计数。GA100 应该有 ECC，请把日志后部的 nvidia-smi 原文一起发回")
        return
    for key, label in labels.items():
        left = before.get(key)
        right = after.get(key)
        if left is not None:
            report.info(f"ECC {label}: 开始 {left}" + (f"，结束 {right}" if right is not None else ""))
    if (before.get("vol_u") or 0) > 0:
        report.fail(f"测试开始前就有不可纠正 ECC: {before.get('vol_u')}")
    if before.get("vol_u") is not None and after.get("vol_u") is not None and after["vol_u"] > before["vol_u"]:
        report.fail(f"测试期间新增不可纠正 ECC: {before['vol_u']} -> {after['vol_u']}")
    elif (after.get("vol_u") or 0) > 0 and before.get("vol_u") is None:
        report.fail(f"测试结束时存在不可纠正 ECC: {after.get('vol_u')}")
    if before.get("vol_c") is not None and after.get("vol_c") is not None and after["vol_c"] > before["vol_c"]:
        report.warn(f"测试期间新增可纠正 ECC: {before['vol_c']} -> {after['vol_c']}")
    if (before.get("agg_u") or 0) > 0:
        report.warn(f"这张卡历史上有不可纠正 ECC {before.get('agg_u')} 条。测试期间没有新增的话，说明现在这次没出错，但卡曾经出过错误")
    if (before.get("agg_c") or 0) > 0:
        report.warn(f"历史上有可纠正 ECC {before.get('agg_c')} 条")


def judge_remap(report: Report, remap: dict[str, str]) -> None:
    pending = (remap.get("pending") or "").strip().lower()
    failure = (remap.get("failure") or "").strip().lower()
    if pending in {"yes", "true", "1"}:
        report.fail("行重映射还挂着（Pending=Yes），显存不稳定，需要重置且不应该当好卡")
    if failure in {"yes", "true", "1"}:
        report.fail("发生过行重映射失败（Remapping Failure），显存有坏区没补上")
    for key, label in (("uncorrectable", "不可纠正行重映射"), ("correctable", "可纠正行重映射")):
        raw = remap.get(key)
        if raw is None or raw.upper() in NA:
            continue
        try:
            count = int(float(raw))
        except ValueError:
            continue
        if count > 0:
            report.warn(f"{label} {count} 行。卡还能用，但说明显存曾经替换过坏行")
        else:
            report.info(f"{label} 0")


def is_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def fill_pattern_torch(buf, salt_i32: int) -> None:
    import torch

    n = buf.numel()
    for start in range(0, n, SLICE_ELEMS):
        end = min(start + SLICE_ELEMS, n)
        ref = torch.arange(start, end, device=buf.device, dtype=torch.int32)
        ref.bitwise_xor_(salt_i32)
        buf[start:end].copy_(ref)
        del ref


def check_pattern_torch(buf, salt_i32: int) -> tuple[int, int | None]:
    import torch

    n = buf.numel()
    bad = 0
    first = None
    for start in range(0, n, SLICE_ELEMS):
        end = min(start + SLICE_ELEMS, n)
        got = buf[start:end].bitwise_xor(salt_i32)
        ref = torch.arange(start, end, device=buf.device, dtype=torch.int32)
        mismatch = got != ref
        count = int(mismatch.sum().item())
        if count and first is None:
            first = start + int(mismatch.nonzero()[0].item())
        bad += count
        del got, ref, mismatch
        if bad > 1000:
            break
    return bad, first


def spot_torch(buf, salt: int) -> tuple[bool, int | None, int | None, int | None]:
    n = buf.numel()
    indexes = [0, 1, 17, min(1000, n - 1), n // 2, n - 1]
    seen = set()
    for index in indexes:
        if index in seen or index < 0 or index >= n:
            continue
        seen.add(index)
        got = int(buf[index].item())
        exp = as_i32((index ^ salt) & 0xFFFFFFFF)
        if got != exp:
            return False, index, got, exp
    return True, None, None, None


def pytorch_deep(report: Report, torch, cuda_index: int, min_bytes: int, stress_s: int) -> None:
    device = torch.device(f"cuda:{cuda_index}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory
    major, minor = torch.cuda.get_device_capability(device)
    report.info(
        f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}  "
        f"设备 {torch.cuda.get_device_name(device)}  计算能力 {major}.{minor}  "
        f"CUDA 可见显存 {total / GIB:.2f} GiB"
    )
    report.stats["torch"] = str(torch.__version__)
    report.stats["sm"] = f"{major}.{minor}"
    if (major, minor) != (8, 0):
        report.fail(f"计算能力是 {major}.{minor}，CMP 170HX / GA100 应为 8.0")

    section("计算正确性")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    cpu_a = torch.randn(256, 256, generator=generator)
    cpu_b = torch.randn(256, 256, generator=generator)
    reference = cpu_a @ cpu_b
    got32 = (cpu_a.to(device) @ cpu_b.to(device)).cpu()
    err32 = (got32 - reference).abs().max().item()
    if (not math.isfinite(err32)) or err32 > 1e-2:
        report.fail(f"FP32 矩阵乘和 CPU 对不上，最大误差 {err32}")
    else:
        report.ok(f"FP32 256 矩阵乘正确，最大误差 {err32:.3e}")
    got16 = (cpu_a.half().to(device) @ cpu_b.half().to(device)).float().cpu()
    rel16 = ((got16 - reference).abs() / reference.abs().clamp(min=1)).max().item()
    if (not math.isfinite(rel16)) or rel16 > 0.1:
        report.fail(f"FP16 矩阵乘和 FP32 参考差太多，相对误差 {rel16}")
    else:
        report.ok(f"FP16 256 矩阵乘正确，相对误差 {rel16:.3e}")
    rep_a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    rep_b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    rep_1 = rep_a @ rep_b
    rep_2 = rep_a @ rep_b
    torch.cuda.synchronize(device)
    rep_diff = (rep_1 - rep_2).abs().max().item()
    if (not math.isfinite(rep_diff)) or rep_diff > 0.1:
        report.fail(f"同一套 FP16 矩阵乘连做两次结果不一致，最大差 {rep_diff}")
    else:
        report.ok(f"FP16 矩阵乘重复计算一致，最大差 {rep_diff:.3e}")
    del cpu_a, cpu_b, reference, got32, got16, rep_a, rep_b, rep_1, rep_2
    torch.cuda.empty_cache()

    section("显存容量和数据")
    allocated = 0
    chunks = []
    pattern_ok = True
    try:
        while allocated + CHUNK_BYTES <= 62 * GIB:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            if free_bytes < CHUNK_BYTES + RESERVE_BYTES:
                break
            try:
                buf = torch.empty(CHUNK_BYTES // 4, dtype=torch.int32, device=device)
            except Exception as exc:
                if is_oom(exc):
                    torch.cuda.empty_cache()
                    break
                raise
            salt = salt_bits(len(chunks))
            fill_pattern_torch(buf, as_i32(salt))
            good, index, got, exp = spot_torch(buf, salt)
            chunks.append((buf, salt))
            allocated += CHUNK_BYTES
            if not good:
                pattern_ok = False
                report.fail(f"显存抽查失败: 块 {len(chunks) - 1} 下标 {index} 读到 {got}，期望 {exp}")
                break
            if len(chunks) % 4 == 0:
                torch.cuda.synchronize(device)
                report.info(f"已写入并抽查 {allocated / GIB:.0f} GiB")
        torch.cuda.synchronize(device)
        if not chunks:
            pattern_ok = False
            report.fail("一块显存都没分配出去")
        elif pattern_ok:
            report.info(f"开始全量回读，共 {allocated / GIB:.1f} GiB")
            total_bad = 0
            for index, (buf, salt) in enumerate(chunks):
                bad, first = check_pattern_torch(buf, as_i32(salt))
                total_bad += bad
                if bad:
                    pattern_ok = False
                    report.fail(f"第 {index} 块回读有 {bad} 个错误，第一个下标 {first}")
                    if total_bad > 1000:
                        break
            if pattern_ok:
                for buf, _salt in chunks:
                    buf.bitwise_xor_(as_i32(0xFFFFFFFF))
                torch.cuda.synchronize(device)
                for index, (buf, salt) in enumerate(chunks):
                    bad, first = check_pattern_torch(buf, as_i32(salt ^ 0xFFFFFFFF))
                    total_bad += bad
                    if bad:
                        pattern_ok = False
                        report.fail(f"取反后第 {index} 块有 {bad} 个错误，第一个下标 {first}。有的位写不进去")
                        if total_bad > 1000:
                            break
            if pattern_ok:
                report.ok(f"显存图案两遍校验通过，共 {allocated / GIB:.1f} GiB，错误 0")
    finally:
        del chunks
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    report.stats["tested_gib"] = f"{allocated / GIB:.1f}"
    if allocated < min_bytes:
        report.fail(
            f"只校验了 {allocated / GIB:.2f} GiB，这张卡至少要校验 {min_bytes / GIB:.0f} GiB 才算 64GB 解锁可用"
        )
    elif pattern_ok:
        report.ok(f"可用显存达到 {allocated / GIB:.1f} GiB，覆盖了解锁后的 64GB 容量")

    section("显存带宽")
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    buf_bytes = 8 * GIB if free_bytes > 20 * GIB else 1 * GIB
    try:
        elements = buf_bytes // 2
        left = torch.empty(elements, dtype=torch.float16, device=device)
        right = torch.empty(elements, dtype=torch.float16, device=device)
        left.fill_(1)
        for _ in range(3):
            right.copy_(left)
        torch.cuda.synchronize(device)
        iters = 30
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            right.copy_(left)
        end.record()
        torch.cuda.synchronize(device)
        seconds = start.elapsed_time(end) / 1000.0
        moved = iters * buf_bytes * 2
        gbs = moved / seconds / 1e9 if seconds > 0 else 0.0
        report.stats["bandwidth_gbs"] = f"{gbs:.0f}"
        report.info(f"设备内拷贝 {gbs:.0f} GB/s（缓冲区 {buf_bytes / GIB:.0f} GiB x {iters}，标称峰值约 1498 GB/s）")
        if gbs > 2200:
            report.warn(f"带宽读数 {gbs:.0f} GB/s 高得不合理，计时可能有问题")
        elif gbs < BW_FAIL_GBS:
            report.fail(f"显存拷贝只有 {gbs:.0f} GB/s，HBM 明显不正常")
        elif gbs < BW_WARN_GBS:
            report.warn(f"显存拷贝 {gbs:.0f} GB/s，低于这张卡应有的水平")
        else:
            report.ok(f"显存带宽正常，{gbs:.0f} GB/s")
        del left, right
    except Exception as exc:
        if is_oom(exc):
            report.fail(f"带宽测试显存不够: {exc}")
        else:
            raise
    finally:
        torch.cuda.empty_cache()

    if stress_s <= 0:
        report.info("已跳过加压")
        return

    section(f"计算加压 {stress_s} 秒")
    hold = None
    try:
        hold_elems = (2 * GIB) // 4
        hold = torch.empty(hold_elems, dtype=torch.int32, device=device)
        hold_salt = salt_bits(999)
        fill_pattern_torch(hold, as_i32(hold_salt))
        side = 8192
        mat_a = torch.randn(side, side, device=device, dtype=torch.float16)
        mat_b = torch.randn(side, side, device=device, dtype=torch.float16)
        for _ in range(2):
            mat_c = mat_a @ mat_b
        torch.cuda.synchronize(device)
        samples = [sample_gpu(report.gpu)]
        started = time.time()
        iters = 0
        last_log = started
        while True:
            for _ in range(3):
                mat_c = mat_a @ mat_b
                iters += 1
            torch.cuda.synchronize(device)
            now = time.time()
            if now - last_log >= 10:
                sample = sample_gpu(report.gpu)
                samples.append(sample)
                report.info(f"加压 {now - started:.0f}s  迭代 {iters}  {fmt_sample(sample)}")
                last_log = now
            if now - started >= stress_s:
                break
        samples.append(sample_gpu(report.gpu))
        elapsed = time.time() - started
        flops = 2.0 * side * side * side * iters
        tflops = flops / elapsed / 1e12 if elapsed > 0 else 0.0
        report.stats["fp16_tflops"] = f"{tflops:.1f}"
        report.info(f"FP16 GEMM {side} 约 {tflops:.1f} TFLOPS，迭代 {iters}，用时 {elapsed:.1f}s")
        mean = float(mat_c.float().mean().item())
        if not math.isfinite(mean):
            report.fail("加压矩阵乘出现 NaN/Inf")
        elif tflops < 5:
            report.fail(f"FP16 算力只有 {tflops:.1f} TFLOPS，卡基本没在算")
        elif tflops < 20:
            report.warn(f"FP16 算力 {tflops:.1f} TFLOPS，偏低。健康的解锁 170HX 一般明显高于这个数")
        else:
            report.ok(f"FP16 算力 {tflops:.1f} TFLOPS")
        again = mat_a @ mat_b
        torch.cuda.synchronize(device)
        drift = (mat_c - again).abs().max().item()
        if (not math.isfinite(drift)) or drift > 0.1:
            report.fail(f"加压结束后重复矩阵乘不一致，最大差 {drift}")
        else:
            report.ok("加压结束后重复矩阵乘仍然一致")
        bad, first = check_pattern_torch(hold, as_i32(hold_salt))
        if bad:
            report.fail(f"加压时旁边的 2 GiB 数据被改坏了，错误 {bad}，第一个下标 {first}")
        else:
            report.ok("加压期间旁边的 2 GiB 数据没有被改坏")
        judge_samples(report, samples)
        judge_throttle(report, report.gpu)
        del mat_a, mat_b, mat_c, again
    finally:
        del hold
        gc.collect()
        torch.cuda.empty_cache()


PTX = r"""
.version 7.4
.target sm_80
.address_size 64

.visible .entry fill_i32(.param .u64 buf, .param .u32 n, .param .u32 salt)
{
    .reg .pred %p;
    .reg .b32 %n, %salt, %idx, %stride, %val, %bdim, %bid, %tx, %grid;
    .reg .b64 %buf, %addr, %off;
    ld.param.u64 %buf, [buf];
    ld.param.u32 %n, [n];
    ld.param.u32 %salt, [salt];
    mov.u32 %tx, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tx;
    mov.u32 %grid, %nctaid.x;
    mul.lo.u32 %stride, %grid, %bdim;
fill_loop:
    setp.ge.u32 %p, %idx, %n;
    @%p bra fill_done;
    xor.b32 %val, %idx, %salt;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %buf, %off;
    st.global.u32 [%addr], %val;
    add.u32 %idx, %idx, %stride;
    bra fill_loop;
fill_done:
    ret;
}

.visible .entry check_i32(.param .u64 buf, .param .u32 n, .param .u32 salt, .param .u64 bad, .param .u64 first)
{
    .reg .pred %p;
    .reg .b32 %n, %salt, %idx, %stride, %got, %exp, %seen, %firstl, %bdim, %bid, %tx, %grid, %ign;
    .reg .b64 %buf, %addr, %off, %bad, %firstp;
    ld.param.u64 %buf, [buf];
    ld.param.u32 %n, [n];
    ld.param.u32 %salt, [salt];
    ld.param.u64 %bad, [bad];
    ld.param.u64 %firstp, [first];
    mov.u32 %tx, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tx;
    mov.u32 %grid, %nctaid.x;
    mul.lo.u32 %stride, %grid, %bdim;
    mov.u32 %seen, 0;
    mov.b32 %firstl, 0xFFFFFFFF;
check_loop:
    setp.ge.u32 %p, %idx, %n;
    @%p bra check_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %buf, %off;
    ld.global.u32 %got, [%addr];
    xor.b32 %exp, %idx, %salt;
    setp.ne.u32 %p, %got, %exp;
    @%p mov.u32 %seen, 1;
    @%p min.u32 %firstl, %firstl, %idx;
    add.u32 %idx, %idx, %stride;
    bra check_loop;
check_done:
    setp.eq.u32 %p, %seen, 1;
    @%p atom.global.add.u32 %ign, [%bad], 1;
    @%p atom.global.min.u32 %ign, [%firstp], %firstl;
    ret;
}

.visible .entry xor_i32(.param .u64 buf, .param .u32 n, .param .u32 mask)
{
    .reg .pred %p;
    .reg .b32 %n, %mask, %idx, %stride, %val, %bdim, %bid, %tx, %grid;
    .reg .b64 %buf, %addr, %off;
    ld.param.u64 %buf, [buf];
    ld.param.u32 %n, [n];
    ld.param.u32 %mask, [mask];
    mov.u32 %tx, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tx;
    mov.u32 %grid, %nctaid.x;
    mul.lo.u32 %stride, %grid, %bdim;
xor_loop:
    setp.ge.u32 %p, %idx, %n;
    @%p bra xor_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %buf, %off;
    ld.global.u32 %val, [%addr];
    xor.b32 %val, %val, %mask;
    st.global.u32 [%addr], %val;
    add.u32 %idx, %idx, %stride;
    bra xor_loop;
xor_done:
    ret;
}

.visible .entry copy_v4(.param .u64 dst, .param .u64 src, .param .u32 nvec)
{
    .reg .pred %p;
    .reg .b32 %n, %idx, %stride, %tx, %bdim, %bid, %grid;
    .reg .b32 %a, %b, %c, %d;
    .reg .b64 %dst, %src, %off, %saddr, %daddr;
    ld.param.u64 %dst, [dst];
    ld.param.u64 %src, [src];
    ld.param.u32 %n, [nvec];
    mov.u32 %tx, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tx;
    mov.u32 %grid, %nctaid.x;
    mul.lo.u32 %stride, %grid, %bdim;
copy_loop:
    setp.ge.u32 %p, %idx, %n;
    @%p bra copy_done;
    mul.wide.u32 %off, %idx, 16;
    add.u64 %saddr, %src, %off;
    add.u64 %daddr, %dst, %off;
    ld.global.v4.u32 {%a, %b, %c, %d}, [%saddr];
    st.global.v4.u32 [%daddr], {%a, %b, %c, %d};
    add.u32 %idx, %idx, %stride;
    bra copy_loop;
copy_done:
    ret;
}

.visible .entry fma_bounded(.param .u64 out, .param .u32 loops)
{
    .reg .pred %p;
    .reg .f32 %a, %b, %c;
    .reg .u32 %i, %loops, %idx, %tx, %bdim, %bid;
    .reg .b64 %out, %addr, %off;
    ld.param.u64 %out, [out];
    ld.param.u32 %loops, [loops];
    mov.u32 %tx, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tx;
    mov.f32 %a, 0f3F7FBE77;
    mov.f32 %b, 0f3A83126F;
    mov.f32 %c, 0f00000000;
    mov.u32 %i, 0;
fma_loop:
    setp.ge.u32 %p, %i, %loops;
    @%p bra fma_done;
    fma.rn.f32 %c, %c, %a, %b;
    add.u32 %i, %i, 1;
    bra fma_loop;
fma_done:
    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %c;
    ret;
}

.visible .entry scale_f32(.param .u64 out, .param .u64 inp, .param .u32 n)
{
    .reg .pred %p;
    .reg .f32 %v, %scale, %bias;
    .reg .b32 %n, %idx, %stride, %tx, %bdim, %bid, %grid;
    .reg .b64 %out, %inp, %off, %oaddr, %iaddr;
    ld.param.u64 %out, [out];
    ld.param.u64 %inp, [inp];
    ld.param.u32 %n, [n];
    mov.f32 %scale, 0f3FC00000;
    mov.f32 %bias, 0f3E800000;
    mov.u32 %tx, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tx;
    mov.u32 %grid, %nctaid.x;
    mul.lo.u32 %stride, %grid, %bdim;
scale_loop:
    setp.ge.u32 %p, %idx, %n;
    @%p bra scale_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %iaddr, %inp, %off;
    add.u64 %oaddr, %out, %off;
    ld.global.f32 %v, [%iaddr];
    fma.rn.f32 %v, %v, %scale, %bias;
    st.global.f32 [%oaddr], %v;
    add.u32 %idx, %idx, %stride;
    bra scale_loop;
scale_done:
    ret;
}
"""


class CudaError(RuntimeError):
    pass


def ptx_load_hint(rc: int) -> str:
    common = "显存读写测试还没开始，不能据此判断卡是坏的。上面的容量和 UUID 仍然有效。"
    if rc == 221:
        return (
            "驱动找不到 PTX 编译器 libnvidia-ptxjitcompiler，多半是没装完整的 NVIDIA 计算库。"
            + common
            + " 请改用带 CUDA 的 PyTorch 再跑: TORCH_PYTHON=~/ComfyUI/venv/bin/python bash test-170hx-64g.sh"
        )
    if rc == 218:
        return "驱动拒绝编译内置测试程序。" + common
    if rc == 209:
        return "驱动没法给这块 GPU 生成可执行代码。" + common
    return common


class OutOfMemory(CudaError):
    pass


class Cuda:
    def __init__(self, ordinal: int):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.ctx = ctypes.c_void_p()
        self.ordinal = ordinal
        self._bind()
        self._check(self.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self._check(self.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
        self.dev = dev
        name = ctypes.create_string_buffer(128)
        self._check(self.cuDeviceGetName(name, 128, dev), "cuDeviceGetName")
        self.name = name.value.decode(errors="replace")
        major = self.attr(75)
        minor = self.attr(76)
        self.major = major
        self.minor = minor
        self._check(self.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev), "cuCtxCreate")
        free_b = ctypes.c_size_t()
        total_b = ctypes.c_size_t()
        self._check(self.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b)), "cuMemGetInfo")
        self.free_bytes = int(free_b.value)
        self.total_bytes = int(total_b.value)
        self.mod = ctypes.c_void_p()
        self._load_module()
        self.fill = self._func(b"fill_i32")
        self.check = self._func(b"check_i32")
        self.xor = self._func(b"xor_i32")
        self.copy = self._func(b"copy_v4")
        self.fma = self._func(b"fma_bounded")
        self.scale = self._func(b"scale_f32")

    def _sym(self, name: str):
        fn = getattr(self.lib, name, None)
        if fn is None and not name.endswith("_v2"):
            fn = getattr(self.lib, name + "_v2", None)
        if fn is None:
            raise CudaError(f"libcuda 缺少 {name}")
        return fn

    def _bind(self) -> None:
        CUdeviceptr = ctypes.c_uint64
        CUstream = ctypes.c_void_p
        self.cuInit = self._sym("cuInit")
        self.cuInit.argtypes = [ctypes.c_uint]
        self.cuDeviceGet = self._sym("cuDeviceGet")
        self.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.cuDeviceGetName = self._sym("cuDeviceGetName")
        self.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.cuDeviceGetAttribute = self._sym("cuDeviceGetAttribute")
        self.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self.cuCtxCreate_v2 = self._sym("cuCtxCreate_v2")
        self.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        self.cuCtxDestroy_v2 = self._sym("cuCtxDestroy_v2")
        self.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self.cuCtxSynchronize = self._sym("cuCtxSynchronize")
        self.cuCtxSynchronize.argtypes = []
        self.cuMemAlloc_v2 = self._sym("cuMemAlloc_v2")
        self.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
        self.cuMemFree_v2 = self._sym("cuMemFree_v2")
        self.cuMemFree_v2.argtypes = [CUdeviceptr]
        self.cuMemGetInfo_v2 = self._sym("cuMemGetInfo_v2")
        self.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.cuModuleLoadData = self._sym("cuModuleLoadData")
        self.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.cuModuleGetFunction = self._sym("cuModuleGetFunction")
        self.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.cuLaunchKernel = self._sym("cuLaunchKernel")
        self.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, CUstream,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.cuMemcpyDtoH_v2 = self._sym("cuMemcpyDtoH_v2")
        self.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]
        self.cuMemcpyHtoD_v2 = self._sym("cuMemcpyHtoD_v2")
        self.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t]
        self.cuEventCreate = self._sym("cuEventCreate")
        self.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.cuEventRecord = self._sym("cuEventRecord")
        self.cuEventRecord.argtypes = [ctypes.c_void_p, CUstream]
        self.cuEventSynchronize = self._sym("cuEventSynchronize")
        self.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        self.cuEventElapsedTime = self._sym("cuEventElapsedTime")
        self.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        self.cuEventDestroy = self._sym("cuEventDestroy_v2") if hasattr(self.lib, "cuEventDestroy_v2") else self._sym("cuEventDestroy")
        self.cuEventDestroy.argtypes = [ctypes.c_void_p]
        self.cuGetErrorString = self._sym("cuGetErrorString")
        self.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        for fn in (
            self.cuInit, self.cuDeviceGet, self.cuDeviceGetName, self.cuDeviceGetAttribute,
            self.cuCtxCreate_v2, self.cuCtxDestroy_v2, self.cuCtxSynchronize,
            self.cuMemAlloc_v2, self.cuMemFree_v2, self.cuMemGetInfo_v2,
            self.cuModuleLoadData, self.cuModuleGetFunction, self.cuLaunchKernel,
            self.cuMemcpyDtoH_v2, self.cuMemcpyHtoD_v2,
            self.cuEventCreate, self.cuEventRecord, self.cuEventSynchronize,
            self.cuEventElapsedTime, self.cuEventDestroy, self.cuGetErrorString,
        ):
            fn.restype = ctypes.c_int

    def _check(self, rc: int, what: str) -> None:
        if rc == 0:
            return
        if rc == 2:
            raise OutOfMemory(what)
        err = ctypes.c_char_p()
        text = ""
        if self.cuGetErrorString(rc, ctypes.byref(err)) == 0 and err.value:
            text = err.value.decode(errors="replace")
        raise CudaError(f"{what} 失败，CUDA 错误 {rc} {text}")

    def attr(self, code: int) -> int:
        value = ctypes.c_int()
        self._check(self.cuDeviceGetAttribute(ctypes.byref(value), code, self.dev), f"attr {code}")
        return int(value.value)

    def _load_module(self) -> None:
        image = ctypes.c_char_p(PTX.encode())
        rc = self.cuModuleLoadData(ctypes.byref(self.mod), image)
        if rc == 0:
            return
        err = ctypes.c_char_p()
        text = ""
        if self.cuGetErrorString(rc, ctypes.byref(err)) == 0 and err.value:
            text = err.value.decode(errors="replace")
        raise CudaError(f"加载测试内核失败 ({rc} {text})。{ptx_load_hint(rc)}")

    def _func(self, name: bytes):
        fn = ctypes.c_void_p()
        self._check(self.cuModuleGetFunction(ctypes.byref(fn), self.mod, name), name.decode())
        return fn

    def alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_uint64()
        rc = self.cuMemAlloc_v2(ctypes.byref(ptr), nbytes)
        if rc == 2:
            raise OutOfMemory(f"分配 {nbytes} 字节失败")
        self._check(rc, "cuMemAlloc")
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        self.cuMemFree_v2(ctypes.c_uint64(ptr))

    def sync(self) -> None:
        self._check(self.cuCtxSynchronize(), "cuCtxSynchronize")

    def launch(self, fn, blocks: int, threads: int, args: list[ctypes._SimpleCData]) -> None:
        ptrs = (ctypes.c_void_p * len(args))()
        for i, arg in enumerate(args):
            ptrs[i] = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p)
        self._check(
            self.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, None, ptrs, None),
            "cuLaunchKernel",
        )

    def dtoh(self, ptr: int, nbytes: int):
        host = ctypes.create_string_buffer(nbytes)
        self._check(self.cuMemcpyDtoH_v2(ctypes.cast(host, ctypes.c_void_p), ctypes.c_uint64(ptr), nbytes), "DtoH")
        return host

    def htod(self, ptr: int, host) -> None:
        raw = ctypes.cast(host, ctypes.c_void_p)
        nbytes = ctypes.sizeof(host)
        self._check(self.cuMemcpyHtoD_v2(ctypes.c_uint64(ptr), raw, nbytes), "HtoD")

    def close(self) -> None:
        if self.ctx:
            self.cuCtxDestroy_v2(self.ctx)
            self.ctx = ctypes.c_void_p()


def ptx_spot(cuda: Cuda, ptr: int, n_elems: int, salt: int) -> tuple[bool, int | None, int | None, int | None]:
    indexes = [0, 1, 17, min(1000, n_elems - 1), n_elems // 2, n_elems - 1]
    seen = set()
    for index in indexes:
        if index in seen or index < 0 or index >= n_elems:
            continue
        seen.add(index)
        host = cuda.dtoh(ptr + index * 4, 4)
        got = int.from_bytes(host.raw[:4], "little")
        exp = (index ^ salt) & 0xFFFFFFFF
        if got != exp:
            return False, index, got, exp
    return True, None, None, None


def ptx_deep(report: Report, cuda_index: int, min_bytes: int, stress_s: int) -> None:
    report.info("没找到带 CUDA 的 PyTorch，改用驱动自带接口做显存和 FMA 测试。矩阵乘算力这项会缺。")
    report.info("如果这台机器有 ComfyUI，请用它的 Python 再跑一次: TORCH_PYTHON=~/ComfyUI/venv/bin/python bash test-170hx-64g.sh")
    cuda = Cuda(cuda_index)
    ptrs: list[int] = []
    try:
        report.info(
            f"驱动 API 设备 {cuda.name}  计算能力 {cuda.major}.{cuda.minor}  "
            f"CUDA 可见显存 {cuda.total_bytes / GIB:.2f} GiB"
        )
        report.stats["sm"] = f"{cuda.major}.{cuda.minor}"
        report.stats["backend"] = "driver"
        if (cuda.major, cuda.minor) != (8, 0):
            report.fail(f"计算能力是 {cuda.major}.{cuda.minor}，CMP 170HX / GA100 应为 8.0")

        section("计算正确性")
        count = 4096
        host_in = (ctypes.c_float * count)()
        for i in range(count):
            host_in[i] = (i % 17) * 0.125 - 1.0
        inp = cuda.alloc(count * 4)
        out = cuda.alloc(count * 4)
        ptrs.extend([inp, out])
        cuda.htod(inp, host_in)
        cuda.launch(
            cuda.scale,
            (count + 255) // 256,
            256,
            [ctypes.c_uint64(out), ctypes.c_uint64(inp), ctypes.c_uint32(count)],
        )
        cuda.sync()
        raw = cuda.dtoh(out, count * 4)
        got = struct.unpack_from("<" + str(count) + "f", raw.raw)
        worst = 0.0
        bad_at = None
        for i in range(count):
            exp = host_in[i] * 1.5 + 0.25
            diff = abs(got[i] - exp)
            if diff > worst:
                worst = diff
                bad_at = i
        if worst > 1e-4:
            report.fail(f"FP32 内核结果不对，下标 {bad_at} 误差 {worst}")
        else:
            report.ok(f"FP32 缩放内核正确，最大误差 {worst:.3e}")
        cuda.free(inp)
        cuda.free(out)
        ptrs.remove(inp)
        ptrs.remove(out)

        section("显存容量和数据")
        allocated = 0
        chunks: list[tuple[int, int, int]] = []
        pattern_ok = True
        threads = 256
        blocks = 2048
        ctr = cuda.alloc(8)
        ptrs.append(ctr)
        try:
            while allocated + CHUNK_BYTES <= 62 * GIB:
                free_b = ctypes.c_size_t()
                total_b = ctypes.c_size_t()
                cuda._check(cuda.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b)), "cuMemGetInfo")
                if int(free_b.value) < CHUNK_BYTES + RESERVE_BYTES:
                    break
                try:
                    ptr = cuda.alloc(CHUNK_BYTES)
                except OutOfMemory:
                    break
                elems = CHUNK_BYTES // 4
                salt = salt_bits(len(chunks))
                cuda.launch(
                    cuda.fill,
                    blocks,
                    threads,
                    [ctypes.c_uint64(ptr), ctypes.c_uint32(elems), ctypes.c_uint32(salt)],
                )
                cuda.sync()
                good, index, got, exp = ptx_spot(cuda, ptr, elems, salt)
                chunks.append((ptr, elems, salt))
                ptrs.append(ptr)
                allocated += CHUNK_BYTES
                if not good:
                    pattern_ok = False
                    report.fail(f"显存抽查失败: 块 {len(chunks) - 1} 下标 {index} 读到 {got}，期望 {exp}")
                    break
                if len(chunks) % 4 == 0:
                    report.info(f"已写入并抽查 {allocated / GIB:.0f} GiB")
            if not chunks:
                pattern_ok = False
                report.fail("一块显存都没分配出去")
            elif pattern_ok:
                report.info(f"开始全量回读，共 {allocated / GIB:.1f} GiB")
                total_bad = 0
                for index, (ptr, elems, salt) in enumerate(chunks):
                    marker = (ctypes.c_uint32 * 2)(0, 0xFFFFFFFF)
                    cuda.htod(ctr, marker)
                    cuda.launch(
                        cuda.check,
                        blocks,
                        threads,
                        [
                            ctypes.c_uint64(ptr),
                            ctypes.c_uint32(elems),
                            ctypes.c_uint32(salt),
                            ctypes.c_uint64(ctr),
                            ctypes.c_uint64(ctr + 4),
                        ],
                    )
                    cuda.sync()
                    back = cuda.dtoh(ctr, 8)
                    bad = int.from_bytes(back.raw[:4], "little")
                    first = int.from_bytes(back.raw[4:8], "little")
                    total_bad += bad
                    if bad:
                        pattern_ok = False
                        shown = "无" if first == 0xFFFFFFFF else str(first)
                        report.fail(f"第 {index} 块回读有 {bad} 个线程看到错误，最小下标 {shown}")
                        if total_bad > 1000:
                            break
                if pattern_ok:
                    for ptr, elems, _salt in chunks:
                        cuda.launch(
                            cuda.xor,
                            blocks,
                            threads,
                            [ctypes.c_uint64(ptr), ctypes.c_uint32(elems), ctypes.c_uint32(0xFFFFFFFF)],
                        )
                    cuda.sync()
                    for index, (ptr, elems, salt) in enumerate(chunks):
                        flipped = (salt ^ 0xFFFFFFFF) & 0xFFFFFFFF
                        marker = (ctypes.c_uint32 * 2)(0, 0xFFFFFFFF)
                        cuda.htod(ctr, marker)
                        cuda.launch(
                            cuda.check,
                            blocks,
                            threads,
                            [
                                ctypes.c_uint64(ptr),
                                ctypes.c_uint32(elems),
                                ctypes.c_uint32(flipped),
                                ctypes.c_uint64(ctr),
                                ctypes.c_uint64(ctr + 4),
                            ],
                        )
                        cuda.sync()
                        back = cuda.dtoh(ctr, 8)
                        bad = int.from_bytes(back.raw[:4], "little")
                        first = int.from_bytes(back.raw[4:8], "little")
                        total_bad += bad
                        if bad:
                            pattern_ok = False
                            shown = "无" if first == 0xFFFFFFFF else str(first)
                            report.fail(f"取反后第 {index} 块有 {bad} 个线程看到错误，最小下标 {shown}")
                            if total_bad > 1000:
                                break
                if pattern_ok:
                    report.ok(f"显存图案两遍校验通过，共 {allocated / GIB:.1f} GiB，错误 0")
        finally:
            for ptr, _elems, _salt in chunks:
                cuda.free(ptr)
                if ptr in ptrs:
                    ptrs.remove(ptr)
            chunks.clear()
            if ctr in ptrs:
                cuda.free(ctr)
                ptrs.remove(ctr)

        report.stats["tested_gib"] = f"{allocated / GIB:.1f}"
        if allocated < min_bytes:
            report.fail(
                f"只校验了 {allocated / GIB:.2f} GiB，这张卡至少要校验 {min_bytes / GIB:.0f} GiB 才算 64GB 解锁可用"
            )
        elif pattern_ok:
            report.ok(f"可用显存达到 {allocated / GIB:.1f} GiB，覆盖了解锁后的 64GB 容量")

        section("显存带宽")
        free_b = ctypes.c_size_t()
        total_b = ctypes.c_size_t()
        cuda._check(cuda.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b)), "cuMemGetInfo")
        buf_bytes = 8 * GIB if int(free_b.value) > 20 * GIB else 1 * GIB
        nvec = buf_bytes // 16
        src = cuda.alloc(buf_bytes)
        dst = cuda.alloc(buf_bytes)
        ptrs.extend([src, dst])
        cuda.launch(cuda.fill, 1024, 256, [ctypes.c_uint64(src), ctypes.c_uint32(buf_bytes // 4), ctypes.c_uint32(0x12345678)])
        cuda.sync()
        warmup = 3
        for _ in range(warmup):
            cuda.launch(cuda.copy, 2048, 256, [ctypes.c_uint64(dst), ctypes.c_uint64(src), ctypes.c_uint32(nvec)])
        cuda.sync()
        iters = 20
        ev0 = ctypes.c_void_p()
        ev1 = ctypes.c_void_p()
        cuda._check(cuda.cuEventCreate(ctypes.byref(ev0), 0), "event")
        cuda._check(cuda.cuEventCreate(ctypes.byref(ev1), 0), "event")
        cuda._check(cuda.cuEventRecord(ev0, None), "record")
        for _ in range(iters):
            cuda.launch(cuda.copy, 2048, 256, [ctypes.c_uint64(dst), ctypes.c_uint64(src), ctypes.c_uint32(nvec)])
        cuda._check(cuda.cuEventRecord(ev1, None), "record")
        cuda._check(cuda.cuEventSynchronize(ev1), "sync event")
        millis = ctypes.c_float()
        cuda._check(cuda.cuEventElapsedTime(ctypes.byref(millis), ev0, ev1), "elapsed")
        seconds = float(millis.value) / 1000.0
        gbs = (iters * buf_bytes * 2) / seconds / 1e9 if seconds > 0 else 0.0
        report.stats["bandwidth_gbs"] = f"{gbs:.0f}"
        report.info(f"设备内拷贝 {gbs:.0f} GB/s（缓冲区 {buf_bytes / GIB:.0f} GiB x {iters}，标称峰值约 1498 GB/s）")
        if gbs > 2200:
            report.warn(f"带宽读数 {gbs:.0f} GB/s 高得不合理，计时可能有问题")
        elif gbs < BW_FAIL_GBS:
            report.fail(f"显存拷贝只有 {gbs:.0f} GB/s，HBM 明显不正常")
        elif gbs < BW_WARN_GBS:
            report.warn(f"显存拷贝 {gbs:.0f} GB/s，低于这张卡应有的水平")
        else:
            report.ok(f"显存带宽正常，{gbs:.0f} GB/s")
        cuda.free(src)
        cuda.free(dst)
        ptrs.remove(src)
        ptrs.remove(dst)
        cuda.cuEventDestroy(ev0)
        cuda.cuEventDestroy(ev1)

        if stress_s <= 0:
            report.info("已跳过加压")
            return

        section(f"计算加压 {stress_s} 秒")
        report.info("这条路径是标量 FMA，不是 Tensor Core 矩阵乘，算力数字会低于 PyTorch，只用来确认计算单元没算错、没过热。")
        fma_blocks = 2048
        fma_threads = 256
        fma_loops = 100000
        fma_out = cuda.alloc(fma_blocks * fma_threads * 4)
        hold = cuda.alloc(GIB)
        ptrs.extend([fma_out, hold])
        hold_elems = GIB // 4
        hold_salt = salt_bits(999)
        cuda.launch(
            cuda.fill,
            blocks,
            threads,
            [ctypes.c_uint64(hold), ctypes.c_uint32(hold_elems), ctypes.c_uint32(hold_salt)],
        )
        cuda.sync()
        samples = [sample_gpu(report.gpu)]
        started = time.time()
        launches = 0
        last_log = started
        while True:
            cuda.launch(
                cuda.fma,
                fma_blocks,
                fma_threads,
                [ctypes.c_uint64(fma_out), ctypes.c_uint32(fma_loops)],
            )
            launches += 1
            cuda.sync()
            now = time.time()
            if now - last_log >= 10 or launches == 1:
                sample = sample_gpu(report.gpu)
                samples.append(sample)
                report.info(f"加压 {now - started:.0f}s  轮次 {launches}  {fmt_sample(sample)}")
                last_log = now
            if now - started >= stress_s:
                break
        samples.append(sample_gpu(report.gpu))
        elapsed = max(time.time() - started, 1e-6)
        flops = 2.0 * fma_loops * fma_blocks * fma_threads * launches
        tflops = flops / elapsed / 1e12
        report.stats["fma_tflops"] = f"{tflops:.2f}"
        raw = cuda.dtoh(fma_out, 32 * 4)
        vals = struct.unpack_from("<32f", raw.raw)
        finite_bad = False
        for i in range(32):
            if not math.isfinite(vals[i]) or abs(vals[i] - 1.0) > 1e-2:
                finite_bad = True
                report.fail(f"FMA 结果异常: 下标 {i} 值 {vals[i]}")
                break
        if not finite_bad:
            report.ok("FMA 结果落在预期范围内")
        report.info(f"标量 FMA 约 {tflops:.2f} TFLOPS，轮次 {launches}，用时 {elapsed:.1f}s")
        if tflops < 0.2:
            report.fail(f"FMA 吞吐只有 {tflops:.2f} TFLOPS，计算单元基本没在工作")
        elif tflops < 1.0:
            report.warn(f"FMA 吞吐 {tflops:.2f} TFLOPS，偏低")
        ctr = cuda.alloc(8)
        ptrs.append(ctr)
        marker = (ctypes.c_uint32 * 2)(0, 0xFFFFFFFF)
        cuda.htod(ctr, marker)
        cuda.launch(
            cuda.check,
            blocks,
            threads,
            [
                ctypes.c_uint64(hold),
                ctypes.c_uint32(hold_elems),
                ctypes.c_uint32(hold_salt),
                ctypes.c_uint64(ctr),
                ctypes.c_uint64(ctr + 4),
            ],
        )
        cuda.sync()
        back = cuda.dtoh(ctr, 8)
        bad = int.from_bytes(back.raw[:4], "little")
        if bad:
            report.fail(f"加压时旁边的 1 GiB 数据被改坏了，出错线程 {bad}")
        else:
            report.ok("加压期间旁边的 1 GiB 数据没有被改坏")
        judge_samples(report, samples)
        judge_throttle(report, report.gpu)
    finally:
        for ptr in list(ptrs):
            try:
                cuda.free(ptr)
            except Exception:
                pass
        cuda.close()


def load_torch():
    try:
        import torch
    except Exception as exc:
        print(f"[信息] 没装 PyTorch（{exc}），将使用驱动接口测试", flush=True)
        return None
    try:
        ok = bool(torch.cuda.is_available())
    except Exception as exc:
        print(f"[信息] PyTorch 看不了 CUDA（{exc}），将使用驱动接口测试", flush=True)
        return None
    if not ok:
        print(f"[信息] 当前 Python 的 PyTorch 没有 CUDA（{torch.__version__}），将使用驱动接口测试", flush=True)
        return None
    return torch


def list_smi_indices() -> list[int]:
    proc = run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi 无法列出 GPU")
    indices = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            indices.append(int(line))
    return indices


def collect_gpu(gpu: int) -> dict[str, str]:
    fields = [
        "name", "uuid", "serial", "vbios_version", "driver_version",
        "pci.bus_id", "pci.device_id", "memory.total", "memory.used",
        "temperature.gpu", "temperature.memory", "power.draw", "power.limit",
        "clocks.current.graphics", "clocks.current.memory",
        "pcie.link.gen.current", "pcie.link.gen.max",
        "pcie.link.width.current", "pcie.link.width.max",
        "fan.speed", "compute_mode", "persistence_mode",
    ]
    got = smi_multi(gpu, fields)
    if got is None:
        data = {}
        for field in fields:
            value = smi_one(gpu, field)
            if value is not None:
                data[field] = value
        got = data
    return got or {}


def is_candidate(info: dict[str, str]) -> bool:
    did = norm_id(info.get("pci.device_id", ""))
    name = info.get("name", "").upper()
    mem = num(info.get("memory.total"))
    if "20C2" in did or "170" in name:
        return True
    return mem is not None and MEM_64_MIN_MIB <= mem <= MEM_64_MAX_MIB


def test_gpu(smi_index: int, cuda_index: int, torch_mod, stress_s: int, force: bool) -> Report:
    report = Report(smi_index)
    section(f"GPU {smi_index}")
    info = collect_gpu(smi_index)
    name = info.get("name", "未知")
    mem = num(info.get("memory.total"))
    used = num(info.get("memory.used"))
    device_id = info.get("pci.device_id", "")
    bus = info.get("pci.bus_id", "")
    pci_text = lspci_line(bus)
    lspci_id = parse_lspci_id(pci_text)
    if not norm_id(device_id) and lspci_id:
        device_id = lspci_id
    report.info(f"名称: {name}")
    report.info(f"UUID: {info.get('uuid', 'N/A')}  序列号: {info.get('serial', 'N/A')}")
    report.info(f"VBIOS: {info.get('vbios_version', 'N/A')}  驱动: {info.get('driver_version', 'N/A')}")
    report.info(f"PCI: {bus or 'N/A'}  设备 ID: {device_id or 'N/A'}")
    if pci_text:
        report.info(f"lspci: {pci_text}")
    if mem is not None:
        report.stats["memory_mib"] = f"{mem:.0f}"
        report.info(f"显存: {mem:.0f} MiB 总量，当前已用 {used if used is not None else 0:.0f} MiB")
    else:
        report.info("显存: 读不到")
    report.info(
        "空闲状态 "
        + fmt_sample(
            {
                "temperature.gpu": num(info.get("temperature.gpu")),
                "temperature.memory": num(info.get("temperature.memory")),
                "power.draw": num(info.get("power.draw")),
                "utilization.gpu": None,
                "clocks.current.graphics": num(info.get("clocks.current.graphics")),
                "clocks.current.memory": num(info.get("clocks.current.memory")),
            }
        )
    )
    report.info(
        f"功耗墙 {info.get('power.limit', 'N/A')} W  "
        f"PCIe 当前 Gen{info.get('pcie.link.gen.current', 'N/A')} x{info.get('pcie.link.width.current', 'N/A')}  "
        f"最大 Gen{info.get('pcie.link.gen.max', 'N/A')} x{info.get('pcie.link.width.max', 'N/A')}"
    )
    fan = info.get("fan.speed")
    if fan is None or fan.upper() in NA:
        report.info("风扇读数 N/A。CMP 170HX 通常是被动散热，靠机箱风道，这不是故障")
    else:
        report.info(f"风扇 {fan}%")

    fails, warns, infos = judge_identity(name, device_id, mem)
    for msg in infos:
        report.ok(msg)
    for msg in warns:
        report.warn(msg)
    for msg in fails:
        report.fail(msg)
    if lspci_id and norm_id(info.get("pci.device_id", "")) and lspci_id.split(":")[-1] not in norm_id(info.get("pci.device_id", "")):
        report.warn(f"nvidia-smi 设备 ID {info.get('pci.device_id')} 和 lspci {lspci_id} 不一致")

    gen = num(info.get("pcie.link.gen.current"))
    width = num(info.get("pcie.link.width.current"))
    if gen is not None and gen < 2:
        report.warn(f"当前 PCIe 是 Gen{gen:.0f}。解锁后的 170HX 冷启动通常至少到 Gen2，Gen1 会让整机通信很慢。这和主板、转接、是否冷启动有关，先记下来")
    if width is not None and width < 8:
        report.warn(f"当前 PCIe 通道是 x{width:.0f}。通道窄不一定是卡坏，可能是插槽或转接卡")

    idle_temp = num(info.get("temperature.gpu"))
    if idle_temp is not None and idle_temp >= 80:
        report.warn(f"还没加压核心就有 {idle_temp:.0f}°C，散热环境不好")

    section("ECC 和行重映射")
    ecc_before = read_ecc(smi_index)
    remap = read_remap(smi_index)
    judge_remap(report, remap)
    xids_before = read_xids()
    if xids_before is None:
        report.warn("读不到内核日志，Xid 没查到。可以用 sudo bash test-170hx-64g.sh 再跑一次补上这项")
    elif xids_before:
        report.info(f"测试前内核里已有 {len(xids_before)} 条 Xid，结束后只看有没有新增")

    apps = list_compute_apps(info.get("uuid", ""))
    if apps:
        report.info("当前占卡进程:")
        for row in apps:
            report.info("  " + row)
    busy = used is not None and used > 2048
    if busy and not force:
        report.fail(
            f"显卡已被占用（已用 {used:.0f} MiB）。请先停掉挖矿、ComfyUI 或其他程序再测。"
            "例如: sudo systemctl stop comfyui 。确认可以打断当前任务时再加 --force"
        )
        report.finish()
        return report
    if busy and force:
        report.warn(f"显卡已被占用（已用 {used:.0f} MiB），按 --force 继续，容量测试可能因此失败")

    identity_ok = not fails
    floor = min_test_bytes(mem, identity_ok)
    try:
        if torch_mod is not None:
            pytorch_deep(report, torch_mod, cuda_index, floor, stress_s)
        else:
            ptx_deep(report, cuda_index, floor, stress_s)
    except Exception as exc:
        if "加载测试内核失败" not in str(exc):
            traceback.print_exc()
        report.fail(f"深度测试中断: {type(exc).__name__}: {exc}")

    section("测试后的 ECC / Xid")
    ecc_after = read_ecc(smi_index)
    judge_ecc_delta(report, ecc_before, ecc_after)
    xids_after = read_xids()
    if xids_before is not None and xids_after is not None:
        known = set(xids_before)
        fresh = [line for line in xids_after if line not in known]
        if fresh:
            report.fail("测试期间内核出现新的 Xid:")
            for line in fresh[:12]:
                print("  " + line, flush=True)
        else:
            report.ok("测试期间没有新的 Xid")

    section("诊断原文")
    try:
        proc = run(["nvidia-smi", "-i", str(smi_index), "-q", "-d", "ECC,ROW_REMAPPER,PCIE,POWER,CLOCK,PERFORMANCE"], timeout=40)
        text = (proc.stdout or proc.stderr or "").strip()
        if text:
            print(text, flush=True)
    except Exception as exc:
        report.info(f"读 nvidia-smi 详情失败: {exc}")
    report.finish()
    return report


def select_gpus() -> list[tuple[int, int]]:
    mode = os.environ.get("TEST_GPU", "all")
    if mode != "all":
        return [(int(mode), 0)]
    indices = list_smi_indices()
    infos = {index: collect_gpu(index) for index in indices}
    chosen = [index for index in indices if is_candidate(infos[index])]
    if not chosen:
        print("[信息] 没有自动认出 170HX / 64GB 卡，改为把每张卡都测一遍", flush=True)
        chosen = indices
    else:
        skipped = [index for index in indices if index not in chosen]
        if skipped:
            print(f"[信息] 跳过不像 170HX 64GB 的卡: {skipped}", flush=True)
    return [(index, index) for index in chosen]


def selftest() -> int:
    assert salt_bits(0) == 0xA5A50000
    assert as_i32(0xFFFFFFFF) == -1
    assert as_i32(salt_bits(0)) < 0
    data = [(index ^ salt_bits(3)) & 0xFFFFFFFF for index in range(128)]
    assert all(((value ^ salt_bits(3)) & 0xFFFFFFFF) == index for index, value in enumerate(data))
    flipped = [value ^ 0xFFFFFFFF for value in data]
    salt2 = salt_bits(3) ^ 0xFFFFFFFF
    assert all(((value ^ salt2) & 0xFFFFFFFF) == index for index, value in enumerate(flipped))
    data[10] ^= 1
    bad = [index for index, value in enumerate(data) if ((value ^ salt_bits(3)) & 0xFFFFFFFF) != index]
    assert bad == [10]
    assert is_active("Active")
    assert not is_active("Not Active")
    assert num("[N/A]") is None
    assert num("65536") == 65536

    def levels(name, did, mem):
        fails, warns, _infos = judge_identity(name, did, mem)
        return fails, warns

    fails, warns = levels("NVIDIA CMP 170HX", "0x20C210DE", 65536)
    assert not fails and not warns
    fails, _warns = levels("NVIDIA CMP 170HX", "0x20C210DE", 8192)
    assert fails
    fails, _warns = levels("NVIDIA CMP 170HX", "0x208210DE", 40960)
    assert fails and any("40GB" in msg for msg in fails)
    fails, _warns = levels("NVIDIA CMP 170HX", "0x20C210DE", 50000)
    assert fails
    fails, _warns = levels("NVIDIA A100", "0x20B210DE", 81920)
    assert fails
    fails, warns = levels("NVIDIA CMP 170HX", "", 65536)
    assert not fails and warns
    assert min_test_bytes(65536, True) == 52 * 1024 ** 3
    assert min_test_bytes(8192, False) < 8 * 1024 ** 3
    print("SELFTEST PASS", flush=True)
    return 0


def main() -> int:
    if os.environ.get("TEST_SELFTEST") == "1":
        return selftest()
    print("判定标准: 8GB 规格 CMP 170HX（PCI 10de:20c2）解锁后应约为 65536 MiB，并且至少 52 GiB 能正确读写。", flush=True)
    print("这是验收测试，不是连续几天的老化。默认计算加压 180 秒。想多烤一会儿: bash test-170hx-64g.sh --stress 600", flush=True)
    if shutil.which("nvidia-smi") is None:
        print("[失败] 没有 nvidia-smi。这台机器还没装好 NVIDIA 驱动。", flush=True)
        return 2
    try:
        probe = run(["nvidia-smi", "-L"])
    except subprocess.TimeoutExpired:
        print("[失败] nvidia-smi 30 秒没有响应，驱动或卡已经卡死。", flush=True)
        return 1
    if probe.returncode != 0:
        print("[失败] nvidia-smi 无法和驱动通信。卡没被识别，或驱动没加载。", flush=True)
        if probe.stderr:
            print(probe.stderr, flush=True)
        if shutil.which("lsmod"):
            try:
                mods = run(["lsmod"]).stdout.splitlines()
            except Exception:
                mods = []
            for line in mods:
                if "nvidia" in line or "nouveau" in line:
                    print(line, flush=True)
            if any(line.startswith("nouveau") for line in mods):
                print("[失败] 正在使用 nouveau，NVIDIA 驱动没有接管这张卡。", flush=True)
        return 1
    try:
        version = open("/proc/driver/nvidia/version", encoding="utf-8", errors="replace").readline().strip()
        if version:
            print("[信息] " + version, flush=True)
    except OSError:
        pass

    try:
        selected = select_gpus()
    except Exception as exc:
        print(f"[失败] 列举 GPU 失败: {exc}", flush=True)
        return 1
    if not selected:
        print("[失败] 没有找到 NVIDIA GPU。", flush=True)
        return 1
    print(f"[信息] 将测试: {', '.join('GPU ' + str(gpu) for gpu, _cuda in selected)}", flush=True)
    torch_mod = load_torch()
    stress_s = int(os.environ.get("TEST_STRESS_SECONDS", "180"))
    force = os.environ.get("TEST_FORCE", "0") == "1"
    reports = []
    for smi_index, cuda_index in selected:
        reports.append(test_gpu(smi_index, cuda_index, torch_mod, stress_s, force))

    print("\n========== 最终结论 ==========", flush=True)
    worst = 0
    for report in reports:
        print(f"GPU {report.gpu}: {report.status}", flush=True)
        if report.fails:
            worst = 1
    if worst:
        print("最终结论: 失败", flush=True)
        print("把整个日志发回。失败项上面已经列出来了。", flush=True)
        return 1
    if any(report.warns for report in reports):
        print("最终结论: 通过（有警告）", flush=True)
        print("卡这次的容量、数据和计算是过的，但警告里的 ECC / 温度 / 带宽要一起看，不要只看最后这一句。", flush=True)
    else:
        print("最终结论: 通过", flush=True)
    print("请把这个日志原样发回。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[失败] 测试被中断", flush=True)
        raise SystemExit(130)
PY
}

if [ "$SELFTEST" -eq 1 ]; then
  PY="$(command -v python3 || true)"
  if [ -z "$PY" ]; then
    echo "没有 python3"
    exit 1
  fi
  export TEST_SELFTEST=1
  run_py
  exit $?
fi

PY="$(pick_python)" || {
  echo "没有找到 python3"
  exit 1
}

LOG="$(pwd)/170hx-64g-$(date +%Y%m%d-%H%M%S).log"
export TEST_GPU="$GPU"
export TEST_STRESS_SECONDS="$STRESS"
export TEST_FORCE="$FORCE"
export TEST_LOG_PATH="$LOG"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
if [ "$GPU" != "all" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
else
  unset CUDA_VISIBLE_DEVICES
fi

echo "日志会写到: $LOG"
echo "开始测试。这张卡会被占满，大约需要几分钟。中间如果超过两分钟完全没有新输出，说明驱动或卡已经挂死，可以 Ctrl+C，并把已有日志发回。"

set +e
{
  echo "170HX 64GB 验收  版本 ${VERSION}"
  echo "时间: $(date '+%F %T %Z')"
  echo "主机: $(hostname 2>/dev/null || echo unknown)"
  echo "内核: $(uname -r)"
  if [ -r /proc/meminfo ]; then
    awk '/MemTotal/ {printf "主机内存: %.0f GiB\n", $2/1024/1024}' /proc/meminfo
  fi
  echo "Python: $PY"
  echo "加压秒数: $STRESS"
  echo "日志: $LOG"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  fi
  echo
  run_py
} 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "日志文件: $LOG"
exit "$rc"
