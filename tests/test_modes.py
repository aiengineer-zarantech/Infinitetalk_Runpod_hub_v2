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
