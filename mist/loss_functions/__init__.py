"""Import loss functions to trigger their registration."""
from .losses.dice import DiceLoss
from .losses.dice_cross_entropy import DiceCELoss
from .losses.cl_dice import CLDice
from .losses.generalized_surface import GenSurfLoss
from .losses.boundary import BoundaryLoss
from .losses.hausdorff_one_sided import HDOneSidedLoss