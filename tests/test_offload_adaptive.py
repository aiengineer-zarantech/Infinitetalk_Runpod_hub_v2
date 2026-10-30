# ==============================================================================
# Adaptive offload (worker_guard/offload.py): calibration format, prediction, smallest blocks_to_swap
# within 85% of the VRAM, the 480p rule, "not calibrated" = tier values, the adaptive log line.
# ==============================================================================
import json
import logging

import pytest

from worker_guard import offload

GB = 1024 ** 3
# Example numbers only (the real ones come from the A.4 runs)
CAL = {"block_gb": 0.4,
       "sampler": {"base_gb": 18.0, "gb_per_ktoken": 0.12},
       "vae": {"base_gb": 17.0, "gb_per_ktoken": 0.2}}


@pytest.fixture
def calibrated(monkeypatch):
    monkeypatch.setenv("OFFLOAD_CALIBRATION", json.dumps(CAL))


def test_latent_tokens():
    assert offload.latent_tokens(1280, 720) == 80 * 45 * 21
    assert offload.latent_tokens(848, 480) == 53 * 30 * 21
    assert offload.latent_tokens(854, 480) == 53 * 30 * 21  # the workflow resizes to a multiple of 16


def test_prediction_is_the_larger_stage_minus_swapped_blocks():
    k = offload.latent_tokens(1280, 720) / 1000  # 75.6
    expected0 = max(18.0 + 0.12 * k, 17.0 + 0.2 * k)  # vae: 32.12
    assert offload.predicted_peak_gb(CAL, 1280, 720, 0) == pytest.approx(expected0)
    assert offload.predicted_peak_gb(CAL, 1280, 720, 10) == pytest.approx(expected0 - 4.0)


def test_adaptive_picks_the_smallest_that_fits():
    budget = 0.85 * 31.4  # 26.69
    blocks, peak, fits = offload.adaptive_blocks(CAL, 1280, 720, budget)
    assert fits and peak <= budget
    assert offload.predicted_peak_gb(CAL, 1280, 720, blocks - 1) > budget  # one block less would not fit
    assert blocks == 14


def test_nothing_fits_uses_40():
    blocks, peak, fits = offload.adaptive_blocks(CAL, 1280, 720, 5.0)
    assert (blocks, fits) == (40, False)


def test_not_calibrated_keeps_the_tier_values():
    d, _ = offload.decide({"width": 1280, "height": 720}, int(31.4 * GB), vram_used_bytes=int(0.5 * GB))
    assert (d.blocks_to_swap, d.force_offload) == (25, True)
    assert d.prediction.calibrated is False and d.prediction.note == "not calibrated, tier value used"


def test_calibrated_720p_is_adaptive(calibrated):
    d, _ = offload.decide({"width": 1280, "height": 720}, int(31.4 * GB), vram_used_bytes=0)
    assert (d.blocks_to_swap, d.force_offload) == (14, False)
    assert d.prediction.fits and d.prediction.budget_gb == pytest.approx(0.85 * 31.4)


def test_calibrated_480p_stays_on_the_gpu(calibrated):
    d, _ = offload.decide({"width": 848, "height": 480}, int(31.4 * GB), vram_used_bytes=0)
    assert (d.blocks_to_swap, d.force_offload) == (0, False)
    assert d.prediction.note == "480p rule: always 0"


def test_vram_not_ours_is_taken_off_the_budget(calibrated):
    # 18 GB used at start, 15 GB of it held by our own resident model: 3 GB belongs to something else
    d, _ = offload.decide({"width": 1280, "height": 720}, int(31.4 * GB), vram_used_bytes=18 * GB,
                          resident_bytes=15 * GB)
    assert d.prediction.foreign_gb == pytest.approx(3.0)
    assert d.prediction.budget_gb == pytest.approx(0.85 * 31.4 - 3.0)
    assert d.blocks_to_swap == 22


def test_target_fraction_env(monkeypatch, calibrated):
    monkeypatch.setenv("OFFLOAD_TARGET_FRACTION", "0.9")
    d, _ = offload.decide({"width": 1280, "height": 720}, int(31.4 * GB), vram_used_bytes=0)
    assert d.prediction.budget_gb == pytest.approx(0.9 * 31.4)
    monkeypatch.setenv("OFFLOAD_TARGET_FRACTION", "0.99")
    assert offload.target_fraction() == 0.85


@pytest.mark.parametrize("raw", ["not json", "[]", json.dumps({**CAL, "block_gb": 0}),
                                 json.dumps({"block_gb": 0.4, "sampler": CAL["sampler"]})])
def test_invalid_calibration_means_not_calibrated(monkeypatch, caplog, raw):
    monkeypatch.setenv("OFFLOAD_CALIBRATION", raw)
    assert offload.load_calibration() is None
    assert any("invalid OFFLOAD_CALIBRATION" in m for m in caplog.messages)


def test_block_gb_estimate_until_calibrated(monkeypatch):
    assert offload.block_gb() == (offload.BLOCK_GB_ESTIMATE, "estimate")
    monkeypatch.setenv("OFFLOAD_CALIBRATION", json.dumps(CAL))
    assert offload.block_gb() == (0.4, "calibrated")


def test_adaptive_line_formats():
    d, _ = offload.decide({"width": 1280, "height": 720}, int(31.4 * GB), vram_used_bytes=0)
    assert offload.adaptive_line(d, int(20.1 * GB), int(31.4 * GB)) == (
        "[OFFLOAD] adaptive: predicted peak n/a (not calibrated, tier value used), chosen blocks_to_swap 25, "
        "actual peak 20.1 GB (64%), budget 26.7 GB")
    manual = offload.Decision("manual", 5, False, 31.4, 1280, 720)
    assert offload.adaptive_line(manual, None, int(31.4 * GB)) == (
        "[OFFLOAD] adaptive: predicted peak n/a (manual mode), chosen blocks_to_swap 5, actual peak n/a")


def test_adaptive_line_calibrated(calibrated):
    d, _ = offload.decide({"width": 1280, "height": 720}, int(31.4 * GB), vram_used_bytes=0)
    line = offload.adaptive_line(d, int(25.0 * GB), int(31.4 * GB))
    assert line.startswith("[OFFLOAD] adaptive: predicted peak 26.5 GB (84%) [smallest that fits], "
                           "chosen blocks_to_swap 14, actual peak 25.0 GB (80%)")


def test_apply_sets_vae_tiling_only_for_the_last_ladder_step():
    d, _ = offload.decide({"width": 1280, "height": 720}, int(31.4 * GB))
    assert "vae_tiling" not in offload.apply({"width": 1280}, d)
    d.vae_tiling = True
    assert offload.apply({"width": 1280}, d)["vae_tiling"] is True
