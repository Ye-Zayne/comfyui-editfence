# EditFence · 编辑越界验收

Check whether a local image edit changed pixels outside the region you permitted. Inspect a face, logo or product detail with an explicit protection mask, then restore protected pixels when needed.

检查局部编辑是否越界：输入原图、编辑结果及允许修改的蒙版，输出违规区域、对照标注和验收报告。可选保护蒙版优先于修改权限，适合保护脸、Logo 和商品细节。

## Install / 安装

Install **EditFence** with ComfyUI Manager after Registry publication, or clone this repository into `ComfyUI/custom_nodes/comfyui-editfence` and restart ComfyUI. Uses ComfyUI's existing torch, NumPy and Pillow environment; no model download or paid service.

## Nodes / 节点

| Node | Inputs | Outputs |
| --- | --- | --- |
| `EditFenceInspect` | original, edited, allowed_mask; optional protected_mask | overlay, violation_mask, boundary_mask, report_json, verdict |
| `EditFenceRestore` | original, edited, allowed_mask; optional protected_mask | restored_image, restored_region_mask |

White (`1`) in `allowed_mask` means modification is permitted. White in `protected_mask` means the original must be protected. Masks may have batch size 1 to broadcast across images. Both images must have identical batch, dimensions and channels. No silent resize or automatic registration.

允许修改蒙版白色表示可改；保护蒙版白色表示保护。原图、结果必须已经对齐且尺寸、批次、通道一致。蒙版不自动缩放。

## Verdict / 判定

- `2 = PASS`: changes stay within permission and tolerance.
- `1 = REVIEW`: changes occur only in the configured boundary ring.
- `0 = FAIL`: outside change fraction exceeds the configured budget, or any explicitly protected pixel changes beyond tolerance.

`pixel_tolerance` is a channel difference on the 0–1 tensor scale. Use 0 for exact comparison; use a small tolerance for compression or accepted noise. `boundary_radius=0` makes all outside pixels strict. A protected area is always strict regardless of the outside change budget. Reports include per-image changed fractions, absolute errors and local uniform-window 7×7 SSIM. SSIM is descriptive; pixel evidence determines the verdict.

Return masks identify measured differences, not semantic damage. Camera movement, perspective changes and deliberate global relighting require alignment or a different acceptance policy first. Images must be floating-point RGB/RGBA within `[0, 1]`; normalize HDR inputs explicitly. Soft **allowed** masks blend original and edited images. In Restore, any nonzero **protected** value restores original pixels completely; protection is not a blending weight. Restore preserves the original dtype and retains original tensor values wherever effective permission is zero. Saved JPEG files do not preserve exact pixels.

恢复节点中，允许蒙版可用灰度做柔和混合；保护蒙版只要大于 0 就完整保留原像素，不会只保护一部分。恢复结果保留原图 dtype。

## Example / 示例

Copy `examples/original.png`, `edited.png` and `allowed.png` into ComfyUI's input directory. `allowed.png` stores edit permission in alpha according to ComfyUI's `LoadImage` MASK convention (`1-alpha`). `examples/inspect_api.json` is an **API-format prompt**, submitted with `POST /prompt` as `{"prompt": ...}`; it is not a canvas workflow JSON. Wire Inspect's overlay into Preview Image to see red violations and yellow boundary changes. The generated example contains one permitted edit and one unwanted corner change.

## Development / 开发

```sh
python -m pip install pytest torch numpy Pillow
python -m pytest tests --rootdir=.. --import-mode=importlib -q
```

MIT license. Version 0.1.0.
