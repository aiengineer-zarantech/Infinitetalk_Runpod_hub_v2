# ==============================================================================
# worker_guard/sysmem.py: container RAM (cgroup v2 / v1 / meminfo) and the swap RAM log lines.
# ==============================================================================
import logging

import pytest

from worker_guard import sysmem
from worker_guard.sysmem import read_ram as real_read_ram  # conftest replaces sysmem.read_ram

GB = 1024 ** 3
MEMINFO = "MemTotal:       131072000 kB\nMemFree:         1000 kB\nMemAvailable:   100000000 kB\n"


def files(mapping):
    return lambda path: mapping.get(path)


def test_cgroup_v2_with_limit_subtracts_page_cache():
    ram = real_read_ram(files({
        "/sys/fs/cgroup/memory.max": str(64 * GB),
        "/sys/fs/cgroup/memory.current": str(20 * GB),
        "/sys/fs/cgroup/memory.stat": f"anon 1\ninactive_file {4 * GB}\nactive_file 5\n",
        "/proc/meminfo": MEMINFO,
    }))
    assert ram == {"limit": 64 * GB, "used": 16 * GB, "available": 48 * GB, "source": "cgroup v2"}


def test_cgroup_v2_without_limit_uses_meminfo():
    ram = real_read_ram(files({
        "/sys/fs/cgroup/memory.max": "max",
        "/sys/fs/cgroup/memory.current": str(10 * GB),
        "/sys/fs/cgroup/memory.stat": "",
        "/proc/meminfo": MEMINFO,
    }))
    assert ram["limit"] == 131072000 * 1024
    assert ram["available"] == min(100000000 * 1024, 131072000 * 1024 - 10 * GB)


def test_cgroup_v1():
    ram = real_read_ram(files({
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(32 * GB),
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": str(12 * GB),
        "/sys/fs/cgroup/memory/memory.stat": f"total_inactive_file {2 * GB}\n",
        "/proc/meminfo": MEMINFO,
    }))
    assert ram == {"limit": 32 * GB, "used": 10 * GB, "available": 22 * GB, "source": "cgroup v1"}


def test_nothing_readable():
    assert real_read_ram(files({})) == {"limit": None, "used": None, "available": None, "source": "meminfo"}


def ram(available_gb):
    return {"limit": 64 * GB, "used": (64 - available_gb) * GB, "available": available_gb * GB, "source": "cgroup v2"}


def test_log_line_includes_need_for_chosen_and_40(caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    sysmem.check_swap_ram(25, 0.44, "estimate", ram(56))
    assert caplog.messages == [
        "[OFFLOAD] RAM (job): limit 64.0 GB, used 8.0 GB, available 56.0 GB (cgroup v2); blocks_to_swap 25 needs "
        "~11.0 GB, 40 needs ~17.6 GB (block 0.44 GB, estimate; unpinned RAM)"
    ]


def test_too_little_ram_for_the_chosen_swap_is_an_error(caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    sysmem.check_swap_ram(25, 0.44, "estimate", ram(12))
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == ["[OFFLOAD] RAM too low for blocks_to_swap 25: needs ~11.0 GB + 2 GB margin, only 12.0 GB "
                      "available. The worker may be killed for running out of RAM"]


def test_room_for_the_job_but_not_for_the_last_ladder_steps(caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    sysmem.check_swap_ram(0, 0.44, "estimate", ram(10))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["[OFFLOAD] RAM: OOM ladder steps above 18 blocks would not fit (40 blocks need ~17.6 GB "
                        "+ 2 GB margin, 10.0 GB available)"]


def test_unknown_ram_only_warns(caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    sysmem.check_swap_ram(10, 0.44, "estimate", {"limit": None, "used": None, "available": None, "source": "meminfo"})
    assert any("available memory unknown" in m for m in caplog.messages)
