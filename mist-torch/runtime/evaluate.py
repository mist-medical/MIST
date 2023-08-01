import os
import json
import ants
import pandas as pd

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn
)

from runtime.utils import init_results_df, evaluate_prediction, compute_results_stats, convert_dict_to_df


def evaluate(data_json, paths, source_dir, output_csv):
    with open(data_json, 'r') as file:
        data = json.load(file)

    if not isinstance(paths, pd.DataFrame):
        # Convert input to pandas dataframe
        if '.csv' in paths:
            paths = pd.read_csv(paths)
        elif type(paths) is dict:
            paths = convert_dict_to_df(paths)
        elif '.json' in paths:
            with open(paths, 'r') as file:
                paths = json.load(file)
            paths = convert_dict_to_df(paths)
        else:
            raise ValueError("Invalid format for paths!")

    results_df = init_results_df(data)
    pred_temp_filename = os.path.join(source_dir, 'pred_temp.nii.gz')
    mask_temp_filename = os.path.join(source_dir, 'mask_temp.nii.gz')

    predictions = os.listdir(source_dir)

    # Set up rich progress bar
    eval_progress = Progress(TextColumn("Evaluating"),
                             BarColumn(),
                             MofNCompleteColumn(),
                             TextColumn("•"),
                             TimeElapsedColumn())

    with eval_progress as pb:
        for i in pb.track(range(len(predictions))):
            # Get true mask and original_prediction
            patient_id = predictions[i].split('.')[0]
            pred = ants.image_read(os.path.join(source_dir, predictions[i]))
            original_mask = ants.image_read(paths.loc[paths['id'].astype(str) == patient_id].iloc[0]['mask'])

            eval_results = evaluate_prediction(pred,
                                               original_mask,
                                               patient_id,
                                               data,
                                               pred_temp_filename,
                                               mask_temp_filename,
                                               results_df.columns)
            results_df = pd.concat([results_df, pd.DataFrame(eval_results, index=[0])], ignore_index=True)

    os.remove(pred_temp_filename)
    os.remove(mask_temp_filename)

    results_df = compute_results_stats(results_df)
    results_df.to_csv(output_csv, index=False)
