import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# My libraries
import sys
sys.path.append('../') 

import utils_multimodal as utils


def load_params_from_json(filepath: str) -> dict:
    """
    Load parameters from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file containing parameters.

    Returns
    -------
    dict
        Dictionary of parameters loaded from the JSON file.
    """
    with open(filepath, 'r') as file:
        params = json.load(file)
    return params


if __name__ == "__main__":

    ##Load parameters from JSON file
    params_filepath = "./load_config.json"
    params = load_params_from_json(params_filepath)

    # Extract fixed parameters from the JSON file
    folders = params["folders"]
    way_to_build_graph = params["way_to_build_graph"]
    device_id = params["device_id"]
    device = torch.device(f'cuda:{device_id}')
    print(f"Seleccionando la GPU {device_id}:", torch.cuda.get_device_name(device))

    best_result_by_split = utils.loadBestHyperparameters(f"../hyperparameters/{way_to_build_graph}/GNN_multimodal.json")

    tester = utils.Tester()
    results, importance_nodes, fc_classifiers, gnn_models = tester.test(
        best_result_by_split, params, device
    )

    keys = list(results.keys())

    # Print results for each split (folder)
    for c in range(len(folders)):
        print(f"================= SPLIT {folders[c]} ===================")
        for key in keys:
            print(f"{key}: {np.round(results[key][c] * 100, 2)}")

    print()

    # Compute the average for s1, s2, s3 (exclude s0)
    filtered_results = {
        key: [results[key][i] for i in range(1, len(folders))]  # Exclude s0, only s1, s2, s3
        for key in keys
    }

    formatted_results = {"Model": "./Results_MultGCNN"}
    
    for key in keys:
        average = np.mean(filtered_results[key])
        std = np.std(filtered_results[key])
        formatted_results[key] = f"{np.round(average * 100, 2)} ± {np.round(std * 100, 2)}"
    
    # Convert to DataFrame
    df = pd.DataFrame([formatted_results])
    
    # Save to CSV
    output_path = "../../../Results_5MTS/multimodal_results.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")

    torch.cuda.empty_cache()
