from .cb_maintainer import CBClassificationMaintainerBase
from .energy_compress import EnergyBasedMaintainer, EnergyCompress
from .cnnr import CNNR
from .inamori import InamoriISelSingleStep

# for retrocompatibility
CBClassificationMaintainer = EnergyBasedMaintainer