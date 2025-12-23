"""Import loss functions to trigger their registration."""
from mist.loss_functions.losses.dice import DiceLoss
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss
from mist.loss_functions.losses.cl_dice import CLDice
from mist.loss_functions.losses.generalized_surface import GenSurfLoss
from mist.loss_functions.losses.boundary import BoundaryLoss
from mist.loss_functions.losses.hausdorff_one_sided import HDOneSidedLoss
