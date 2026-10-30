# ==============================================================================
# worker_guard/modes.py: "mode": "fast480" (I2V single, 848x480 or 480x848), old jobs untouched.
# ==============================================================================
import pytest

from worker_guard import modes


@pytest.mark.parametrize("job_input", [
    {},
    {"width": 720, "height": 480, "blocks_to_swap": 0},
    {"mode": "standard", "width": 1280, "height": 720},
])
def test_jobs_without_fast480_are_returned_unchanged(job_input):
    out, error = modes.prepare_job_input(job_input)
    assert error is None and out is job_input
    assert modes.job_mode(job_input) == "standard"


@pytest.mark.parametrize("extra, size", [
    ({}, (848, 480)),
    ({"aspect": "16:9"}, (848, 480)),
    ({"aspect": "9:16"}, (480, 848)),
    ({"aspect": " 9:16 ", "width": 1280, "height": 720}, (480, 848)),  # client size replaced
    ({"mode": "FAST480"}, (848, 480)),
])
def test_fast480_sets_the_size(extra, size):
    job_input = {"mode": "fast480", "image_base64": "x", **extra}
    out, error = modes.prepare_job_input(job_input)
    assert error is None
    assert (out["width"], out["height"]) == size
    assert out is not job_input  # the caller's dict is not modified
    assert modes.job_mode(out) == "fast480"
    again, _ = modes.prepare_job_input(out)  # idempotent (the guard and the handler both call it)
    assert again == out


@pytest.mark.parametrize("job_input, message", [
    ({"mode": "fast480", "input_type": "video"}, "fast480 supports input_type 'image' with person_count 'single' only"),
    ({"mode": "fast480", "person_count": "multi"}, "fast480 supports input_type 'image' with person_count 'single'"),
    ({"mode": "fast480", "aspect": "4:3"}, "fast480 aspect must be one of ['16:9', '9:16']"),
    ({"mode": "turbo"}, "unknown mode 'turbo'"),
])
def test_invalid_inputs_are_rejected(job_input, message):
    out, error = modes.prepare_job_input(job_input)
    assert error.startswith("[MODE] ") and message in error
    assert out is job_input


def test_fast480_is_480p_for_offloading():
    from worker_guard import offload
    for width, height in modes.FAST480_SIZES.values():
        assert offload.resolution_row(width, height) == "480p"


# ---------------------------------------------------------------- v26: upscale inputs and "upscale_only"

@pytest.mark.parametrize("job_input, expected", [
    ({"mode": "fast480"}, {"model": "3b"}),                                  # 720p is the fast480 default
    ({"mode": "fast480", "upscale": "720p", "upscale_model": "7B"}, {"model": "7b"}),
    ({"mode": "fast480", "upscale": "none"}, None),
    ({"mode": "upscale_only", "video_url": "https://x/v.mp4"}, {"model": "3b"}),
    ({}, None),
    ({"mode": "standard"}, None),
    ({"upscale": "none"}, None),
])
def test_upscale_settings(job_input, expected):
    out, error = modes.prepare_job_input(job_input)
    assert error is None
    assert modes.upscale_settings(out) == expected


def test_old_jobs_with_upscale_none_are_unchanged():
    job_input = {"upscale": "none", "width": 720, "height": 480}
    out, error = modes.prepare_job_input(job_input)
    assert error is None and out is job_input


@pytest.mark.parametrize("job_input, message", [
    ({"upscale": "720p"}, "upscale is only available with mode 'fast480' or 'upscale_only'"),
    ({"mode": "standard", "upscale": "720p"}, "upscale is only available"),
    ({"mode": "fast480", "upscale": "4k"}, "upscale must be '720p' or 'none'"),
    ({"mode": "fast480", "upscale_model": "1b"}, "upscale_model must be one of ['3b', '7b']"),
    ({"mode": "upscale_only"}, "upscale_only needs one of ['video_base64', 'video_url', 'video_path']"),
    ({"mode": "upscale_only", "video_path": "/x.mp4", "upscale": "none"}, "nothing to do"),
])
def test_invalid_upscale_inputs_are_rejected(job_input, message):
    out, error = modes.prepare_job_input(job_input)
    assert error.startswith("[MODE] ") and message in error


def test_upscale_only_mode():
    job_input = {"mode": "upscale_only", "video_base64": "eA=="}
    out, error = modes.prepare_job_input(job_input)
    assert error is None and out is job_input  # no size to set
    assert modes.job_mode(out) == "upscale_only"
