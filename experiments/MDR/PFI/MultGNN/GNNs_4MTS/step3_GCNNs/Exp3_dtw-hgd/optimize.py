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

    # Load parameters from JSON file
    params_filepath = "./load_config.json"
    params = load_params_from_json(params_filepath)

    # Extract fixed parameters from the JSON file
    folders = params["folders"]
    way_to_build_graph = params["way_to_build_graph"]
    device_id = params["device_id"]
    device = torch.device(f'cuda:{device_id}')
    print(f"Seleccionando la GPU {device_id}:", torch.cuda.get_device_name(device))

    best_result_by_split = {}

    for folder in range(len(folders)):
        torch.cuda.empty_cache()
        
        # Load temporal and static data
        X_train_temporal, X_val_temporal, X_test_temporal, \
        X_train_static, X_val_static, X_test_static, \
        y_train, y_val, y_test = utils.load_data(device, folder)
        
        # Load temporal and static adjacency matrices
        S_temporal = pd.read_csv(f"../../step2_graphRepresentation/{way_to_build_graph}/s{str(int(folders[folder])+1)}/graph_Xtr_th_0.25.csv")
        S_static = pd.read_csv(f"../../step2_graphRepresentation/{way_to_build_graph}/s{str(int(folders[folder])+1)}/static_graph_Xtr_th_0.25.csv")
        
        S_temporal = torch.tensor(S_temporal.values, dtype=torch.float32).to(device)
        S_static = torch.tensor(S_static.values, dtype=torch.float32).to(device)
        
        print("S_temporal shape:", S_temporal.shape)
        print("S_static shape:", S_static.shape)
        print("X_train_temporal shape:", X_train_temporal.shape)
        print("X_train_static shape:", X_train_static.shape)
        print("===========> TRAIN-VAL PHASE ==================")
        
        # Train and validate to find the best hyperparameters
        optimizer = utils.Optimizer()
        best_hyperparameters = optimizer.optimize(
            S_temporal, S_static, 
            X_train_temporal, X_train_static, 
            X_val_temporal, X_val_static, 
            y_train, y_val, 
            params, device
        )
        
        print("<========== END TRAIN-VAL PHASE ===============")
        best_result_by_split[folder] = best_hyperparameters    

    # Save the best hyperparameters
    output_filepath = f"../hyperparameters/{way_to_build_graph}/GNN_multimodal.json"
    utils.saveBestHyperparameters(best_result_by_split, output_filepath)

    torch.cuda.empty_cache()
