"""Текущая нагрузка системы: CPU, память, видеокарта — без внешних зависимостей.

Нужна виджету в интерфейсе, чтобы человек видел, что происходит с машиной во
время рендера. Всё считается локально и дёшево: никаких сетевых вызовов.

macOS: CPU — счётчики ядра через mach host_statistics, память — vm_stat,
видеокарта — ioreg (Device Utilization %).
Windows: CPU — GetSystemTimes, память — GlobalMemoryStatusEx, GPU — nvidia-smi,
если он есть (иначе «н/д»).
Linux: /proc/stat и /proc/meminfo, GPU — nvidia-smi.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
import sys
import time
import sys as _sys

from typing import Any, Dict, List, Optional, Tuple

_CACHE: Dict[str, Any] = {}


def _platform() -> str:
    return sys.platform
_GPU_CACHE: Dict[str, Any] = {"value": None, "ts": 0.0}
_GPU_FAIL_TS: List[float] = [0.0]
_GPU_SOURCE: List[str] = [""]
_MEM_CACHE: Dict[str, Any] = {"value": None, "ts": 0.0}


def _spawn() -> Dict[str, Any]:
    """Флаги запуска без консольного окна (Windows)."""
    try:
        from . import resources

        return resources.spawn_kwargs()
    except Exception:  # noqa: BLE001
        return {"creationflags": 0x08000000} if _sys.platform == "win32" else {}
_LAST_CPU: List[Tuple[float, float]] = []
_LAST_VALUE: List[float] = [0.0]
_LAST_TS: List[float] = [0.0]


# --- CPU ---------------------------------------------------------------------

def _cpu_macos() -> Optional[Tuple[int, int]]:
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
    except OSError:
        return None
    mach_host_self = libc.mach_host_self
    mach_host_self.restype = ctypes.c_uint32
    host_statistics = libc.host_statistics
    host_statistics.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]

    class CpuLoadInfo(ctypes.Structure):
        _fields_ = [("user", ctypes.c_uint32), ("system", ctypes.c_uint32),
                    ("idle", ctypes.c_uint32), ("nice", ctypes.c_uint32)]

    info = CpuLoadInfo()
    count = ctypes.c_uint32(4)
    ret = host_statistics(mach_host_self(), 3,  # HOST_CPU_LOAD_INFO
                          ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_uint32)),
                          ctypes.byref(count))
    if ret != 0:
        return None
    busy = info.user + info.system + info.nice
    return busy, busy + info.idle


def _cpu_windows() -> Optional[Tuple[int, int]]:
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        class FILETIME(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

        idle, kern, user = FILETIME(), FILETIME(), FILETIME()
        if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
            return None

        def to_int(ft: "FILETIME") -> int:
            return (ft.high << 32) | ft.low

        idle_t = to_int(idle)
        busy = to_int(kern) + to_int(user) - idle_t
        return busy, busy + idle_t
    except Exception:
        return None


def _cpu_linux() -> Optional[Tuple[int, int]]:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as fh:
            parts = fh.readline().split()[1:]
        vals = [int(x) for x in parts]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        return total - idle, total
    except Exception:
        return None


def _cpu_ticks() -> Optional[Tuple[int, int]]:
    if _platform() == "darwin":
        return _cpu_macos()
    if sys.platform == "win32":
        return _cpu_windows()
    return _cpu_linux()


def cpu_percent(force: bool = False) -> Optional[float]:
    """Загрузка CPU между двумя вызовами (как в top).

    Частые вызовы (< 250 мс) отдают закешированное значение: за такой интервал
    разница счётчиков может быть нулевой, и виджет мигал бы «н/д».
    """
    now = time.monotonic()
    if not force and _LAST_VALUE[0] and now - _LAST_TS[0] < 0.25:
        return _LAST_VALUE[0]
    ticks = _cpu_ticks()
    if ticks is None:
        return None
    busy, total = ticks
    if not _LAST_CPU:
        _LAST_CPU.append((busy, total))
        time.sleep(0.12)
        ticks = _cpu_ticks()
        if ticks is None:
            return None
        busy, total = ticks
    prev_busy, prev_total = _LAST_CPU[0]
    _LAST_CPU[0] = (busy, total)
    d_total = total - prev_total
    d_busy = busy - prev_busy
    if d_total <= 0:
        # дельта нулевая — оцениваем по средней нагрузке на ядро
        try:
            value = round(min(100.0, os.getloadavg()[0] / max(1, os.cpu_count() or 1) * 100), 1)
        except Exception:
            value = _LAST_VALUE[0] or 0.0
        _LAST_TS[0] = now
        return value
    value = round(max(0.0, min(100.0, d_busy / d_total * 100)), 1)
    _LAST_VALUE[0] = value
    _LAST_TS[0] = now
    return value


# --- Память ------------------------------------------------------------------

def memory(force: bool = False) -> Dict[str, Any]:
    now = time.monotonic()
    if not force and _MEM_CACHE.get("value") and now - _MEM_CACHE.get("ts", 0.0) < 1.0:
        return dict(_MEM_CACHE["value"])
    result = _memory_uncached()
    _MEM_CACHE["value"] = result
    _MEM_CACHE["ts"] = now
    return dict(result)


def _memory_uncached() -> Dict[str, Any]:
    total = 0
    used = 0
    if _platform() == "darwin":
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                       text=True, timeout=3, **_spawn()).stdout.strip() or 0)
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3, **_spawn()).stdout
            page = 16384
            m = re.search(r"page size of (\d+) bytes", out)
            if m:
                page = int(m.group(1))
            vals = {}
            for line in out.splitlines():
                m2 = re.match(r"([^:]+):\s+(\d+)", line.strip())
                if m2:
                    vals[m2.group(1).strip()] = int(m2.group(2))
            used_pages = (vals.get("Pages active", 0) + vals.get("Pages wired down", 0)
                          + vals.get("Pages occupied by compressor", 0))
            used = used_pages * page
        except Exception:
            pass
    elif _platform() == "win32":
        try:
            class MEMSTATUS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            st = MEMSTATUS()
            st.dwLength = ctypes.sizeof(MEMSTATUS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                total = int(st.ullTotalPhys)
                used = total - int(st.ullAvailPhys)
        except Exception:
            pass
    else:
        try:
            info = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    k, v = line.split(":", 1)
                    info[k] = int(v.strip().split()[0]) * 1024
            total = info.get("MemTotal", 0)
            used = total - info.get("MemAvailable", info.get("MemFree", 0))
        except Exception:
            pass

    total_mb = round(total / 1024 / 1024)
    used_mb = round(used / 1024 / 1024)
    percent = round(used / total * 100, 1) if total else 0.0
    return {"used_mb": used_mb, "total_mb": total_mb, "percent": percent}


# --- Видеокарта ---------------------------------------------------------------

def _gpu_windows_pdh() -> Dict[str, Any]:
    """Загрузка GPU через счётчики PDH — без запуска nvidia-smi и окон консоли.

    Windows отдаёт производительность видеокарты в счётчиках «GPU Engine».
    Читаем их напрямую через pdh.dll: ни одного дочернего процесса.
    """
    try:
        import ctypes
        from ctypes import wintypes as wt
    except Exception:  # noqa: BLE001
        return {"percent": None, "name": ""}

    try:
        pdh = ctypes.WinDLL("pdh.dll")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return {"percent": None, "name": ""}

    query = wt.HANDLE()
    if pdh.PdhOpenQueryW(None, ctypes.c_size_t(0), ctypes.byref(query)) != 0:
        return {"percent": None, "name": ""}

    path = "\\GPU Engine(*)\\Utilization Percentage"
    size = wt.DWORD(0)
    pdh.PdhExpandWildCardPathW(None, path, None, ctypes.byref(size), wt.DWORD(0))
    if size.value <= 0:
        pdh.PdhCloseQuery(query)
        return {"percent": None, "name": ""}

    buf = ctypes.create_unicode_buffer(size.value + 2)
    if pdh.PdhExpandWildCardPathW(None, path, buf, ctypes.byref(size), wt.DWORD(0)) != 0:
        pdh.PdhCloseQuery(query)
        return {"percent": None, "name": ""}

    counters = []
    for counter_path in buf[:size.value].split("\0"):
        if not counter_path or "engtype_3d" not in counter_path.lower():
            continue
        handle = wt.HANDLE()
        if pdh.PdhAddCounterW(query, counter_path, ctypes.c_size_t(0), ctypes.byref(handle)) == 0:
            counters.append(handle)
    if not counters:
        pdh.PdhCloseQuery(query)
        return {"percent": None, "name": ""}

    pdh.PdhCollectQueryData(query)
    time.sleep(0.08)
    pdh.PdhCollectQueryData(query)

    total = 0.0
    PDH_FMT_DOUBLE = 0x00000200
    for handle in counters:
        value = ctypes.c_double(0.0)
        counter_type = wt.DWORD(0)
        if pdh.PdhGetFormattedCounterValue(handle, PDH_FMT_DOUBLE, ctypes.byref(counter_type),
                                           ctypes.byref(value)) == 0:
            total += max(0.0, value.value)
    pdh.PdhCloseQuery(query)
    percent = int(round(min(100.0, total)))
    return {"percent": percent, "name": "GPU"}


def _gpu_windows_registry_name() -> str:
    """Имя видеокарты из реестра — тоже без запуска процессов."""
    try:
        import winreg  # type: ignore[import-not-found]

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\WinSAT")
        value, _ = winreg.QueryValueEx(key, "PrimaryAdapterString")
        return str(value).strip() or "GPU"
    except Exception:  # noqa: BLE001
        return "GPU"


def _gpu_windows() -> Dict[str, Any]:
    """Сначала PDH; nvidia-smi — только как редкий запасной вариант (30 с кэш)."""
    now = time.monotonic()
    if _GPU_CACHE.get("value") and now - _GPU_CACHE.get("ts", 0.0) < 2.0:
        return dict(_GPU_CACHE["value"])

    result = _gpu_windows_pdh()
    if result.get("percent") is not None:
        _GPU_SOURCE[0] = "pdh"
        result["name"] = _gpu_windows_registry_name()
        _GPU_CACHE["value"] = result
        _GPU_CACHE["ts"] = now
        return dict(result)

    # PDH не отдал данные (старая Windows/счётчики отключены) — nvidia-smi,
    # но не чаще раза в 30 секунд и всегда без окна консоли
    legacy = _GPU_CACHE.get("legacy") or {"value": None, "ts": 0.0}
    if legacy["value"] and now - legacy["ts"] < 30.0:
        return dict(legacy["value"])
    # неудачу тоже запоминаем, чтобы не спавнить nvidia-smi каждые 2 секунды
    if now - _GPU_FAIL_TS[0] < 30.0:
        return {"percent": None, "name": _gpu_windows_registry_name()}
    smi = _gpu_nvidia()
    if smi.get("percent") is not None:
        _GPU_SOURCE[0] = "nvidia-smi"
        _GPU_CACHE["legacy"] = {"value": smi, "ts": now}
        _GPU_CACHE["value"] = smi
        _GPU_CACHE["ts"] = now
        return dict(smi)
    _GPU_FAIL_TS[0] = now
    return {"percent": None, "name": _gpu_windows_registry_name()}


def _gpu_macos() -> Dict[str, Any]:
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
                             capture_output=True, text=True, timeout=4, **_spawn()).stdout
        if not out:
            out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                                 capture_output=True, text=True, timeout=4, **_spawn()).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        if not m:
            return {"percent": None, "name": "Apple GPU"}
        model = ""
        mm = re.search(r'"model"\s*=\s*"([^"]+)"', out)
        if mm:
            model = mm.group(1)
        return {"percent": int(m.group(1)), "name": model or "Apple GPU"}
    except Exception:
        return {"percent": None, "name": "Apple GPU"}


def _gpu_nvidia() -> Dict[str, Any]:
    exe = "nvidia-smi"
    try:
        out = subprocess.run([exe, "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=4, **_spawn())
        if out.returncode != 0 or not out.stdout.strip():
            return {"percent": None, "name": ""}
        first = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        return {"percent": int(float(parts[0])), "name": parts[3] if len(parts) > 3 else "GPU"}
    except Exception:
        return {"percent": None, "name": ""}


def gpu(force: bool = False) -> Dict[str, Any]:
    """Видеокарта с кэшем на 1.5 с: на Windows каждый вызов — это nvidia-smi."""
    now = time.monotonic()
    if not force and _GPU_CACHE.get("value") and now - _GPU_CACHE.get("ts", 0.0) < 1.5:
        return dict(_GPU_CACHE["value"])
    plat = _platform()
    if plat == "darwin":
        result = _gpu_macos()
    elif plat == "win32":
        return _gpu_windows()
    elif plat == "linux":
        result = _gpu_nvidia()
    else:
        result = {"percent": None, "name": ""}
    _GPU_CACHE["value"] = result
    _GPU_CACHE["ts"] = now
    return dict(result)


def active_job(projects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Что приложение делает прямо сейчас — для строки процесса в виджете."""
    for p in projects:
        status = str(p.get("status") or "")
        if status in ("running", "preparing", "queued"):
            return {
                "id": p.get("id"),
                "title": p.get("title") or p.get("url") or "Проект",
                "status": status,
                "stage": p.get("stage") or "",
                "progress": round(float(p.get("progress") or 0), 1),
            }
    return None


def snapshot(app_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Всё сразу: CPU, память, GPU и (опционально) наши лимиты/энкодер."""
    data = {
        "cpu": cpu_percent(),
        "monitor": {"gpu_source": _GPU_SOURCE[0] or ("ioreg" if _platform() == "darwin" else "none")},
        "ram": memory(),
        "gpu": gpu(),
        "ts": round(time.time(), 3),
    }
    if app_stats:
        data["app"] = app_stats
    return data
