"""Permission-based comparison for aligned ComfyUI IMAGE tensors."""

import json
import math

import torch
import torch.nn.functional as F


def validate_pair(original, edited, promote=True):
    if not isinstance(original, torch.Tensor) or not isinstance(edited, torch.Tensor):
        raise ValueError("Expected IMAGE tensors.")
    if original.ndim != 4 or original.shape[-1] not in (3, 4):
        raise ValueError("Expected IMAGE [batch, height, width, 3 or 4].")
    if original.shape != edited.shape:
        raise ValueError(
            "EditFence requires identical batch, size and channels; align images first."
        )
    if original.device != edited.device:
        edited = edited.to(original.device)
    if not bool(torch.isfinite(original).all() and torch.isfinite(edited).all()):
        raise ValueError(
            "Images contain NaN or infinity; no trustworthy verdict is possible."
        )
    if min(original.shape[1:3]) < 1 or original.shape[0] < 1:
        raise ValueError("Image batch and dimensions must be nonempty.")
    if not original.is_floating_point() or not edited.is_floating_point():
        raise ValueError("Images must be floating-point RGB/RGBA within [0, 1].")
    if bool(
        ((original < 0) | (original > 1)).any() or ((edited < 0) | (edited > 1)).any()
    ):
        raise ValueError(
            "Images must be within [0, 1]; normalize HDR inputs explicitly."
        )
    return (original.float(), edited.float()) if promote else (original, edited)


def prepare_mask(mask, image, name):
    if mask is None:
        return torch.zeros(image.shape[:3], device=image.device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or tuple(mask.shape[1:]) != tuple(image.shape[1:3]):
        raise ValueError(
            f"{name} must have matching height and width; masks are never resized silently."
        )
    if mask.shape[0] == 1 and image.shape[0] != 1:
        mask = mask.expand(image.shape[0], -1, -1)
    if mask.shape[0] != image.shape[0]:
        raise ValueError(f"{name} batch must be 1 or match the image batch.")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    if bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError(f"{name} values must be in [0, 1].")
    return mask.to(device=image.device, dtype=torch.float32)


def dilate(mask, radius):
    if radius == 0:
        return mask
    return (
        F.max_pool2d(
            mask.float().unsqueeze(1), radius * 2 + 1, stride=1, padding=radius
        )
        .squeeze(1)
        .bool()
    )


def spatial_ssim(original, edited):
    """Local 7x7 SSIM map, averaged over channels, uniform window."""
    a, b = original.permute(0, 3, 1, 2), edited.permute(0, 3, 1, 2)
    k = min(7, a.shape[-2], a.shape[-1])
    k = k if k % 2 else k - 1
    pool = lambda x: F.avg_pool2d(
        x, k, stride=1, padding=k // 2, count_include_pad=False
    )
    ma, mb = pool(a), pool(b)
    va, vb = (pool(a * a) - ma * ma).clamp_min(0), (pool(b * b) - mb * mb).clamp_min(0)
    cov = pool(a * b) - ma * mb
    result = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return result.mean(dim=1).clamp(-1, 1)


def region_metrics(error, ssim, region, changed):
    count = int(region.sum().item())
    if count == 0:
        return {
            "pixels": 0,
            "changed_pixels": 0,
            "changed_fraction": 0.0,
            "mean_absolute_error": None,
            "max_channel_error": None,
            "ssim": None,
        }
    selected = error[region]
    count_changed = int((changed & region).sum().item())
    return {
        "pixels": count,
        "changed_pixels": count_changed,
        "changed_fraction": count_changed / count,
        "mean_absolute_error": float(selected.mean().item()),
        "max_channel_error": float(selected.max().item()),
        "ssim": float(ssim[region].mean().item()),
    }


class EditFenceInspect:
    CATEGORY = "EditFence"
    FUNCTION = "inspect"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "INT")
    RETURN_NAMES = (
        "overlay",
        "violation_mask",
        "boundary_mask",
        "report_json",
        "verdict",
    )
    DESCRIPTION = "Check changes against an allowed edit mask. 0 FAIL / 1 REVIEW / 2 PASS. Aligned images only."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "allowed_mask": ("MASK",),
                "pixel_tolerance": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "max_outside_change_fraction": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "boundary_radius": ("INT", {"default": 3, "min": 0, "max": 64}),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {"protected_mask": ("MASK",)},
        }

    def inspect(
        self,
        original,
        edited,
        allowed_mask,
        pixel_tolerance=0.02,
        max_outside_change_fraction=0.0,
        boundary_radius=3,
        mask_threshold=0.01,
        protected_mask=None,
    ):
        for label, value in [
            ("pixel_tolerance", pixel_tolerance),
            ("max_outside_change_fraction", max_outside_change_fraction),
            ("mask_threshold", mask_threshold),
        ]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} must be finite and in [0, 1].")
        if not isinstance(boundary_radius, int) or not 0 <= boundary_radius <= 64:
            raise ValueError("boundary_radius must be an integer in [0, 64].")
        original, edited = validate_pair(original, edited, promote=False)
        allowed = prepare_mask(allowed_mask, original, "allowed_mask") > mask_threshold
        protected = (
            prepare_mask(protected_mask, original, "protected_mask") > mask_threshold
        )
        allowed = allowed & ~protected
        boundary = dilate(allowed, boundary_radius) & ~allowed & ~protected
        outside = ~allowed & ~boundary
        error = (edited - original).abs()
        changed = error.amax(-1) > pixel_tolerance
        violations = changed & outside
        ssim = spatial_ssim(original.float(), edited.float())
        overlay = edited[..., :3].float().clone().clamp(0, 1)
        red = torch.tensor([1.0, 0.05, 0.05], device=overlay.device)
        yellow = torch.tensor([1.0, 0.8, 0.05], device=overlay.device)
        overlay[violations] = overlay[violations] * 0.35 + red * 0.65
        boundary_changed = boundary & changed
        overlay[boundary_changed] = overlay[boundary_changed] * 0.35 + yellow * 0.65
        frames = []
        for i in range(original.shape[0]):
            outside_m = region_metrics(error[i], ssim[i], outside[i], changed[i])
            protected_m = region_metrics(error[i], ssim[i], protected[i], changed[i])
            # Explicit protected pixels are strict: the outside area budget cannot hide a damaged logo.
            fail = (
                outside_m["changed_fraction"] > max_outside_change_fraction
                or protected_m["changed_pixels"] > 0
            )
            review = bool(boundary_changed[i].any())
            verdict = 0 if fail else 1 if review else 2
            frames.append(
                {
                    "image_index": i,
                    "verdict": ["FAIL", "REVIEW", "PASS"][verdict],
                    "verdict_code": verdict,
                    "allowed": region_metrics(
                        error[i], ssim[i], allowed[i], changed[i]
                    ),
                    "outside": outside_m,
                    "protected": protected_m,
                    "boundary": region_metrics(
                        error[i], ssim[i], boundary[i], changed[i]
                    ),
                }
            )
        verdict = min(x["verdict_code"] for x in frames)
        report = {
            "schema": "editfence/1",
            "verdict": ["FAIL", "REVIEW", "PASS"][verdict],
            "verdict_code": verdict,
            "images": frames,
            "settings": {
                "pixel_tolerance": pixel_tolerance,
                "max_outside_change_fraction": max_outside_change_fraction,
                "boundary_radius": boundary_radius,
                "mask_threshold": mask_threshold,
            },
            "notes": [
                "No automatic alignment or semantic identity judgment.",
                "SSIM windows can overlap neighboring regions; pixel evidence determines the verdict.",
                "Yellow boundary changes require review; explicit protection overrides edit permission.",
            ],
        }
        return (
            overlay,
            violations.float(),
            boundary.float(),
            json.dumps(report, ensure_ascii=False, allow_nan=False),
            verdict,
        )


class EditFenceRestore:
    CATEGORY = "EditFence"
    FUNCTION = "restore"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("restored_image", "restored_region_mask")
    DESCRIPTION = "Restore original pixels outside the allowed mask. Soft masks blend; protection always wins."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "allowed_mask": ("MASK",),
            },
            "optional": {"protected_mask": ("MASK",)},
        }

    def restore(self, original, edited, allowed_mask, protected_mask=None):
        original, edited = validate_pair(original, edited, promote=False)
        alpha = prepare_mask(allowed_mask, original, "allowed_mask")
        protected = prepare_mask(protected_mask, original, "protected_mask")
        # Permission may feather, but protection must not become partial permission.
        alpha = alpha.masked_fill(protected > 0, 0)
        calculation_dtype = (
            torch.float64 if original.dtype == torch.float64 else torch.float32
        )
        weight = alpha.unsqueeze(-1).to(calculation_dtype)
        blended = (
            original.to(calculation_dtype) * (1 - weight)
            + edited.to(calculation_dtype) * weight
        )
        image = blended.to(original.dtype)
        image = torch.where(weight == 0, original, image)
        image = torch.where(weight == 1, edited.to(original.dtype), image)
        return image, 1 - alpha
