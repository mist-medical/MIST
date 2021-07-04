import numpy as np
import panas as pd
import SimpleITK as sitk
import ants

from mist.metrics import dice
from mist.metrics import hausdorff

def evaluate_brats_batch(csv):
    # Input columns - id, truth, pred

    # Output columns - id, Dice, Hausdorff
    #...
    #Mean, median, std


    ### output results.csv ###

def evaluate_brats_single(truth, pred):

    results_dict = {'Dice_ET:', 0,
                    'Dice_WT': 0,
                    'Dice_TC': 0,
                    'Hausdorff_ET': 0,
                    'Hausdorff_WT': 0,
                    'Hausdorff_TC': 0}
    return results_dict

def create_brats_labels(pred, output_type):

    # Create three images:
    # Pred_WT, Pred_TC, pred_ET

    # output type is either ants image or sitk image
