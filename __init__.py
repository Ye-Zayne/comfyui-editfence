from .nodes import EditFenceInspect, EditFenceRestore

NODE_CLASS_MAPPINGS = {
    "EditFenceInspect": EditFenceInspect,
    "EditFenceRestore": EditFenceRestore,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EditFenceInspect": "EditFence · Inspect Edit Boundary / 编辑越界验收",
    "EditFenceRestore": "EditFence · Restore Protected Pixels / 恢复保护区域",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
