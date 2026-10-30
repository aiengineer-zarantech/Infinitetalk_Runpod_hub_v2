"""Container RAM check before swapping transformer blocks to the CPU.

The wrapper moves swapped blocks with block.to(cpu) and does not pin them (no pin_memory in the
installed ComfyUI-WanVideoWrapper), so they need ordinary RAM. ComfyUI's own pinned-memory limit
(logged at startup as "Enabled pinned memory N") covers ComfyUI-managed models only.
Reads the cgroup (v2, else v1) limit and usage, minus reclaimable page cache, and /proc/meminfo.
"""
import logging

logger = logging.getLogger("worker_guard")

GB = 1024 ** 3
RAM_MARGIN_GB = 2.0
# cgroup v1 reports "no limit" as a huge number
_NO_LIMIT = 1 << 60

CGROUP_V2 = "/sys/fs/cgroup"
CGROUP_V1 = "/sys/fs/cgroup/memory"


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _stat(text, key):
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == key:
            return _int(parts[1])
    return None


def _meminfo(text, key):
    for line in (text or "").splitlines():
        if line.startswith(key + ":"):
            value = _int(line.split()[1])
            return None if value is None else value * 1024
    return None


def read_ram(read=_read):
    """{"limit": bytes or None, "used": bytes or None, "available": bytes or None, "source": str}."""
    limit = used = None
    source = "meminfo"
    max_v2 = read(f"{CGROUP_V2}/memory.max")
    if max_v2 is not None:
        source = "cgroup v2"
        limit = None if max_v2 == "max" else _int(max_v2)
        current = _int(read(f"{CGROUP_V2}/memory.current"))
        cache = _stat(read(f"{CGROUP_V2}/memory.stat"), "inactive_file") or 0
        used = None if current is None else max(0, current - cache)
    else:
        limit_v1 = _int(read(f"{CGROUP_V1}/memory.limit_in_bytes"))
        if limit_v1 is not None:
            source = "cgroup v1"
            limit = None if limit_v1 >= _NO_LIMIT else limit_v1
            usage = _int(read(f"{CGROUP_V1}/memory.usage_in_bytes"))
            cache = _stat(read(f"{CGROUP_V1}/memory.stat"), "total_inactive_file") or 0
            used = None if usage is None else max(0, usage - cache)
    meminfo = read("/proc/meminfo")
    host_available = _meminfo(meminfo, "MemAvailable")
    if limit is None:
        limit = _meminfo(meminfo, "MemTotal")
    candidates = [v for v in (host_available, None if limit is None or used is None else limit - used) if v is not None]
    return {"limit": limit, "used": used, "available": min(candidates) if candidates else None, "source": source}


def _gb(value):
    return "n/a" if value is None else f"{value / GB:.1f} GB"


def check_swap_ram(blocks_to_swap, block_gb, block_source, ram=None, what="job"):
    """Logs RAM headroom for swapping blocks_to_swap blocks and for 40 (the ladder's last steps).
    Returns the RAM dict. Only logs; it never changes the job."""
    from worker_guard.offload import MAX_BLOCKS
    ram = read_ram() if ram is None else ram
    need = blocks_to_swap * block_gb
    need_max = MAX_BLOCKS * block_gb
    available = ram.get("available")
    logger.info(
        f"[OFFLOAD] RAM ({what}): limit {_gb(ram.get('limit'))}, used {_gb(ram.get('used'))}, available "
        f"{_gb(available)} ({ram.get('source')}); blocks_to_swap {blocks_to_swap} needs ~{need:.1f} GB, "
        f"{MAX_BLOCKS} needs ~{need_max:.1f} GB (block {block_gb:.2f} GB, {block_source}; unpinned RAM)"
    )
    if available is None:
        logger.warning("[OFFLOAD] RAM: available memory unknown; cannot check room for swapped blocks")
        return ram
    available_gb = available / GB
    if blocks_to_swap > 0 and available_gb < need + RAM_MARGIN_GB:
        logger.error(
            f"[OFFLOAD] RAM too low for blocks_to_swap {blocks_to_swap}: needs ~{need:.1f} GB + {RAM_MARGIN_GB:.0f} GB "
            f"margin, only {available_gb:.1f} GB available. The worker may be killed for running out of RAM"
        )
    elif available_gb < need_max + RAM_MARGIN_GB:
        fits = max(0, min(MAX_BLOCKS, int((available_gb - RAM_MARGIN_GB) / block_gb)))
        logger.warning(
            f"[OFFLOAD] RAM: OOM ladder steps above {fits} blocks would not fit ({MAX_BLOCKS} blocks need "
            f"~{need_max:.1f} GB + {RAM_MARGIN_GB:.0f} GB margin, {available_gb:.1f} GB available)"
        )
    return ram
