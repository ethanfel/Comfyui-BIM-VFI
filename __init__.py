from .nodes import LoadBIMVFIModel, BIMVFIInterpolate

NODE_CLASS_MAPPINGS = {
    "LoadBIMVFIModel": LoadBIMVFIModel,
    "BIMVFIInterpolate": BIMVFIInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBIMVFIModel": "Load BIM-VFI Model",
    "BIMVFIInterpolate": "BIM-VFI Interpolate",
}
