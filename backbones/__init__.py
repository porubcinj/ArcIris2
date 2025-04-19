from .convnext import ArcIrisConvNeXt
from .iresnet import arcirisresnet18, arcirisresnet34, arcirisresnet50, arcirisresnet100, arcirisresnet200

def get_model(name: str, **kwargs):
    name = name.lower()

    # ConvNeXt
    if name.startswith("convnext"):
        return ArcIrisConvNeXt(name, **kwargs)

    # ResNet
    if name == "r18":
        return arcirisresnet18(False, **kwargs)
    elif name == "r34":
        return arcirisresnet34(False, **kwargs)
    elif name == "r50":
        return arcirisresnet50(False, **kwargs)
    elif name == "r100":
        return arcirisresnet100(False, **kwargs)
    elif name == "r200":
        return arcirisresnet200(False, **kwargs)

    raise ValueError()
