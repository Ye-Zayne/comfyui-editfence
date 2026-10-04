import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("editfence_nodes", Path(__file__).parents[1] / "nodes.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_allowed_edit_pass_and_protected_logo_fails():
    original = torch.zeros(1, 16, 16, 3)
    edited = original.clone(); edited[:, 4:12, 4:12] = 1
    allowed = torch.zeros(1, 16, 16); allowed[:, 4:12, 4:12] = 1
    assert mod.EditFenceInspect().inspect(original, edited, allowed)[-1] == 2
    protected = torch.zeros_like(allowed); protected[:, 6, 6] = 1
    result = mod.EditFenceInspect().inspect(original, edited, allowed, protected_mask=protected)
    assert result[-1] == 0
    assert result[1][0, 6, 6] == 1


def test_boundary_review_and_far_change_fail():
    original = torch.zeros(1, 16, 16, 3)
    edited = original.clone(); edited[:, 3, 6] = 1
    allowed = torch.zeros(1, 16, 16); allowed[:, 4:12, 4:12] = 1
    assert mod.EditFenceInspect().inspect(original, edited, allowed, boundary_radius=1)[-1] == 1
    edited[:, 0, 0] = 1
    assert mod.EditFenceInspect().inspect(original, edited, allowed, boundary_radius=1)[-1] == 0


def test_restore_preserves_outside_exactly_and_does_not_mutate_inputs():
    original = torch.rand(2, 16, 16, 3); edited = torch.rand_like(original)
    before = edited.clone()
    mask = torch.zeros(1, 16, 16); mask[:, 4:12, 4:12] = 1
    restored, _ = mod.EditFenceRestore().restore(original, edited, mask)
    assert torch.equal(restored[:, :4], original[:, :4])
    assert torch.equal(restored[:, 4:12, 4:12], edited[:, 4:12, 4:12])
    assert torch.equal(edited, before)


def test_tolerance_batch_aggregation_and_json_finite():
    original = torch.zeros(2, 8, 8, 3); edited = original.clone(); edited[1, 0, 0] = .01
    mask = torch.zeros(1, 8, 8)
    result = mod.EditFenceInspect().inspect(original, edited, mask, pixel_tolerance=.02)
    assert result[-1] == 2
    report = json.loads(result[3]); assert len(report["images"]) == 2
    assert report["images"][0]["allowed"]["ssim"] is None


def test_bad_shapes_and_nonfinite_never_pass():
    original = torch.zeros(1, 8, 8, 3)
    with pytest.raises(ValueError, match="identical"):
        mod.EditFenceInspect().inspect(original, torch.zeros(1, 9, 8, 3), torch.zeros(1, 8, 8))
    with pytest.raises(ValueError, match="matching"):
        mod.EditFenceInspect().inspect(original, original, torch.zeros(1, 7, 8))
    invalid = original.clone(); invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        mod.EditFenceInspect().inspect(original, invalid, torch.zeros(1, 8, 8))


def test_full_permission_empty_permission_and_tiny_image():
    original = torch.zeros(1, 1, 1, 3); edited = torch.ones_like(original)
    assert mod.EditFenceInspect().inspect(original, edited, torch.ones(1, 1, 1))[-1] == 2
    assert mod.EditFenceInspect().inspect(original, edited, torch.zeros(1, 1, 1))[-1] == 0


def test_soft_protection_is_complete_and_overrides_soft_permission():
    original = torch.zeros(1, 4, 4, 3)
    edited = torch.ones_like(original)
    allowed = torch.full((1, 4, 4), .5)
    protected = torch.zeros_like(allowed); protected[0, 1, 1] = .25
    restored, restored_mask = mod.EditFenceRestore().restore(original, edited, allowed, protected)
    assert torch.equal(restored[0, 1, 1], original[0, 1, 1])
    assert restored_mask[0, 1, 1] == 1
    assert torch.equal(restored[0, 0, 0], torch.full((3,), .5))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_restore_preserves_original_dtype_and_exact_protected_values(dtype):
    original = torch.rand(1, 4, 4, 3, dtype=dtype)
    edited = torch.zeros_like(original)
    restored, _ = mod.EditFenceRestore().restore(original, edited, torch.zeros(1, 4, 4))
    assert restored.dtype == dtype
    assert torch.equal(restored, original)


def test_out_of_range_input_is_rejected_before_ssim_or_json():
    image = torch.full((1, 4, 4, 3), 1e300, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        mod.EditFenceInspect().inspect(image, image, torch.zeros(1, 4, 4))


def test_zero_tolerance_detects_float64_change_below_float32_precision():
    original = torch.full((1, 4, 4, 3), .5, dtype=torch.float64)
    edited = original.clone(); edited[0, 0, 0, 0] += 1e-10
    result = mod.EditFenceInspect().inspect(original, edited, torch.zeros(1, 4, 4), pixel_tolerance=0)
    assert result[-1] == 0 and result[1][0, 0, 0] == 1
