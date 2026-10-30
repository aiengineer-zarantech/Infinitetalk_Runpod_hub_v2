# ==============================================================================
# v26: the upscale always runs in time segments sized from the container RAM. Every frame is kept exactly
# once, in order; each segment loads up to CONTEXT_FRAMES extra frames on both sides.
# ==============================================================================
import pytest

from worker_guard import upscale

GB = 1024 ** 3


@pytest.mark.parametrize("available_gb, per_frame_mb, expected", [
    (56, 150, 337),     # (56 - 4) GB / 150 MB = 354 frames - 16 context -> 337 (4n+1)
    (24, 150, 117),     # 136 - 16 = 120 -> 117
    (200, 150, 401),    # capped
    (6, 150, 33),       # floor (logged by run())
    (56, 300, 161),     # a heavier per-frame estimate halves the segment
])
def test_segment_size_from_ram(available_gb, per_frame_mb, expected):
    size = upscale.segment_frames(available_gb * GB, per_frame_mb)
    assert size == expected
    assert (size - 1) % 4 == 0


def test_segment_size_unknown_ram_and_override():
    assert upscale.segment_frames(None, 150) == upscale.SEGMENT_UNKNOWN_RAM
    assert upscale.segment_frames(56 * GB, 150, override=24) == 21
    assert upscale.segment_frames(56 * GB, 150, override=2) == 5
    assert upscale.segment_frames(56 * GB, 150, override=10_000) == upscale.SEGMENT_MAX


def test_ram_per_frame_env(monkeypatch):
    assert upscale.ram_per_frame_mb() == upscale.DEFAULT_RAM_PER_FRAME_MB
    monkeypatch.setenv("UPSCALE_RAM_PER_FRAME_MB", "90")
    assert upscale.ram_per_frame_mb() == 90
    monkeypatch.setenv("UPSCALE_RAM_PER_FRAME_MB", "lots")
    assert upscale.ram_per_frame_mb() == upscale.DEFAULT_RAM_PER_FRAME_MB


@pytest.mark.parametrize("total, size", [(1, 33), (33, 33), (34, 33), (375, 97), (1500, 337), (50, 21)])
def test_segments_cover_every_frame_once(total, size):
    segments = upscale.plan_segments(total, size)
    kept = [f for s in segments for f in range(s.start, s.end)]
    assert kept == list(range(total))
    for s in segments:
        assert s.load_start == max(0, s.start - upscale.CONTEXT_FRAMES)
        assert s.load_end == min(total, s.end + upscale.CONTEXT_FRAMES)
        assert s.end - s.start <= size
    assert [s.index for s in segments] == list(range(len(segments)))


def test_single_segment_has_no_context():
    (seg,) = upscale.plan_segments(300, 337)
    assert (seg.start, seg.end, seg.load_start, seg.load_end) == (0, 300, 0, 300)


@pytest.mark.parametrize("size, expected", [((848, 480), (1280, 720)), ((480, 848), (720, 1280)),
                                            ((854, 480), (1280, 720)), ((1280, 720), (1280, 720))])
def test_target_size(size, expected):
    assert upscale.target_size(*size) == expected


@pytest.mark.parametrize("size", [(640, 480), (480, 480), (720, 1080)])
def test_target_size_rejects_other_aspects(size):
    with pytest.raises(upscale.UpscaleError, match="16:9 and 9:16 only"):
        upscale.target_size(*size)


def test_center_crop_matches_the_target_aspect():
    assert upscale.center_crop(848, 480, (1280, 720)) == (848, 477)
    assert upscale.center_crop(480, 848, (720, 1280)) == (477, 848)
    assert upscale.center_crop(1280, 720, (1280, 720)) == (1280, 720)
