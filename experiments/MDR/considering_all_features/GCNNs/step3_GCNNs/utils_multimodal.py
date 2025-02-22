import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, confusion_matrix
from typing import Dict, Any, Tuple, List

import random

import sys
sys.path.append('../../../../../../classification_architectures/')
import multgcnn as models_multimodal

###############################################################
# Functions to save and load best hyperparameters and results #
###############################################################

def saveBestHyperparameters(best_hyperparameters, filename):
    """
    Save the best hyperparameters to a JSON file.
    
    Args:
        best_hyperparameters (dict): The dictionary containing the best hyperparameters.
        filename (str): The name of the file to save the hyperparameters.
    """
    with open(filename, 'w') as file:
        json.dump(best_hyperparameters, file, indent=4)
        
def loadBestHyperparameters(filename):
    """
    Load the best hyperparameters from a JSON file.
    
    Args:
        filename (str): The name of the file to load the hyperparameters from.
    
    Returns:
        dict: The dictionary containing the loaded hyperparameters.
    """
    with open(filename, 'r') as file:
        hyperparameters = json.load(file)
        
    return hyperparameters

#########################
# Function to load data #
#########################

import pandas as pd
import numpy as np
import torch

def load_data(device: torch.device, split: str) -> tuple:
    """
    Load and preprocess static and temporal data for training, validation, and testing.

    Parameters
    ----------
    device : torch.device
        Device to load the data onto (CPU or GPU).
    split : str
        Data split name (e.g., 'train', 'val', 'test').

    Returns
    -------
    tuple
        - X_train_temporal (torch.Tensor): Temporal training data.
        - X_val_temporal (torch.Tensor): Temporal validation data.
        - X_test_temporal (torch.Tensor): Temporal test data.
        - X_train_static (torch.Tensor): Static training data.
        - X_val_static (torch.Tensor): Static validation data.
        - X_test_static (torch.Tensor): Static test data.
        - y_train (torch.Tensor): Training labels.
        - y_val (torch.Tensor): Validation labels.
        - y_test (torch.Tensor): Test labels.
    """
    # Load static data
    X_train_static = pd.read_csv(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/X_train_static_0.csv").drop(['Unnamed: 0'], axis=1)
    X_val_static = pd.read_csv(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/X_val_static_0.csv").drop(['Unnamed: 0'], axis=1)
    X_test_static = pd.read_csv(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/X_test_static.csv").drop(['Unnamed: 0'], axis=1)

    # Load temporal data
    X_train_temporal = np.load(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/X_train_tensor_0.npy")
    X_val_temporal = np.load(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/X_val_tensor_0.npy")
    X_test_temporal = np.load(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/X_test_tensor.npy")

    # Load labels
    y_train = pd.read_csv(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/y_train_0.csv", index_col=0)
    y_val = pd.read_csv(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/y_val_0.csv", index_col=0)
    y_test = pd.read_csv(f"../../../../../../ORIGINAL_DATA/MDR/splits_14_days/notbalanced/split_{split}/y_test.csv", index_col=0)

    # Replace missing values (666) with 0
    X_train_static[X_train_static == 666] = 0
    X_val_static[X_val_static == 666] = 0
    X_test_static[X_test_static == 666] = 0
    X_train_temporal[X_train_temporal == 666] = 0
    X_val_temporal[X_val_temporal == 666] = 0
    X_test_temporal[X_test_temporal == 666] = 0

    # Convert static data to tensors
    X_train_static = torch.tensor(X_train_static.values, dtype=torch.float32)
    X_val_static = torch.tensor(X_val_static.values, dtype=torch.float32)
    X_test_static = torch.tensor(X_test_static.values, dtype=torch.float32)

    X_train_temporal = np.nanmean(X_train_temporal, axis=1)
    X_val_temporal = np.nanmean(X_val_temporal, axis=1)
    X_test_temporal = np.nanmean(X_test_temporal, axis=1)

    X_train_temporal = torch.tensor(X_train_temporal, dtype=torch.float32)
    X_val_temporal = torch.tensor(X_val_temporal, dtype=torch.float32)
    X_test_temporal = torch.tensor(X_test_temporal, dtype=torch.float32)

    X_train_temporal = X_train_temporal.unsqueeze(2)
    X_val_temporal = X_val_temporal.unsqueeze(2) 
    X_test_temporal = X_test_temporal.unsqueeze(2) 
    
    X_train_static = X_train_static.unsqueeze(2)
    X_val_static = X_val_static.unsqueeze(2) 
    X_test_static = X_test_static.unsqueeze(2) 

    print("==============>",X_train_temporal.shape)

    # Convert labels to tensors
    y_train = torch.tensor(y_train.values, dtype=torch.float32).squeeze()
    y_val = torch.tensor(y_val.values, dtype=torch.float32).squeeze()
    y_test = torch.tensor(y_test.values, dtype=torch.float32).squeeze()

    # Move tensors to the specified device only if CUDA is available
    if device.type == "cuda":
        return (
            X_train_temporal.to(device), X_val_temporal.to(device), X_test_temporal.to(device),
            X_train_static.to(device), X_val_static.to(device), X_test_static.to(device),
            y_train.to(device), y_val.to(device), y_test.to(device)
        )
    else:
        return (
            X_train_temporal, X_val_temporal, X_test_temporal,
            X_train_static, X_val_static, X_test_static,
            y_train, y_val, y_test
        )

    
############################
# Function to train models #
############################

import torch
from typing import Dict, Any


class WeightedBinaryCrossEntropy(nn.Module):
    """
    Implementation of Weighted Binary Cross Entropy (WBCE) in PyTorch.

    Parameters
    ----------
    w1 : float
        Weight for the positive class.
    w2 : float
        Weight for the negative class.
    """

    def __init__(self, w1: float = 0.82, w2: float = 0.18):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the WBCE loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions after applying the sigmoid function.
        y_true : torch.Tensor
            Ground truth labels (0 or 1).

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        epsilon = 1e-7  # To prevent NaN in logs
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        # Calculate WBCE
        loss_pos = -self.w1 * y_true * torch.log(y_pred)
        loss_neg = -self.w2 * (1 - y_true) * torch.log(1 - y_pred)
        return (loss_pos + loss_neg).mean()




class WeightedBinaryCrossEntropy(nn.Module):
    """
    Implementation of Weighted Binary Cross Entropy (WBCE) in PyTorch.

    Parameters
    ----------
    w1 : float
        Weight for the positive class.
    w2 : float
        Weight for the negative class.
    """

    def __init__(self, w1: float = 0.82, w2: float = 0.18):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the WBCE loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions after applying the sigmoid function.
        y_true : torch.Tensor
            Ground truth labels (0 or 1).

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        epsilon = 1e-7  # To prevent NaN in logs
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        loss_pos = -self.w1 * y_true * torch.log(y_pred)
        loss_neg = -self.w2 * (1 - y_true) * torch.log(1 - y_pred)
        return (loss_pos + loss_neg).mean()


class Optimizer:
    """
    Wrapper for optimization and training steps.

    Parameters
    ----------
    model : torch.nn.Module
        The model to optimize.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay (L2 regularization).
    """

    # def __init__(self, model: torch.nn.Module, lr: float, weight_decay: float):
    #     self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def step(self):
        """Perform a single optimization step."""
        self.optimizer.step()

    def zero_grad(self):
        """Clear gradients of all optimized tensors."""
        self.optimizer.zero_grad()

    def train(
        self,
        model: torch.nn.Module,
        X_temporal: torch.Tensor,
        X_static: torch.Tensor,
        y: torch.Tensor,
        loss_function: nn.Module,
    ) -> torch.Tensor:
        """
        Perform a single training step for the multimodal model.

        Parameters
        ----------
        model : torch.nn.Module
            Multimodal GNN model to train.
        X_temporal : torch.Tensor
            Temporal input features.
        X_static : torch.Tensor
            Static input features.
        y : torch.Tensor
            Target labels.
        loss_function : nn.Module
            Loss function for training.

        Returns
        -------
        torch.Tensor
            The calculated loss for the current training step.
        """
        model.train()
        self.zero_grad()

        # Forward pass
        output = model(X_temporal, X_static).squeeze()

        # Compute loss
        loss = loss_function(output, y)
        loss.backward()
        self.step()

        return loss

    def optimize(
        self,
        S_temporal: torch.Tensor,
        S_static: torch.Tensor,
        X_train_temporal: torch.Tensor,
        X_train_static: torch.Tensor,
        X_val_temporal: torch.Tensor,
        X_val_static: torch.Tensor,
        y_train: torch.Tensor,
        y_val: torch.Tensor,
        params: Dict[str, Any],
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Train the multimodal model and validate it to find the best hyperparameters.

        Parameters
        ----------
        S_temporal : torch.Tensor
            Graph operator for temporal data.
        S_static : torch.Tensor
            Graph operator for static data.
        X_train_temporal : torch.Tensor
            Training data for temporal features.
        X_train_static : torch.Tensor
            Training data for static features.
        X_val_temporal : torch.Tensor
            Validation data for temporal features.
        X_val_static : torch.Tensor
            Validation data for static features.
        y_train : torch.Tensor
            Training labels.
        y_val : torch.Tensor
            Validation labels.
        params : dict
            Dictionary of training parameters and hyperparameters.
        device : torch.device
            Device to perform training on (CPU or GPU).

        Returns
        -------
        dict
            The best hyperparameters found during training.
        """
        n_epochs = params['n_epochs']
        early_stopping_patience = params['early_stopping_patience']
        embedding_dim_static = params['embedding_dim_static']
        embedding_dim_temporal = params['embedding_dim_temporal']

        h_dropout = params['h_dropout']
        h_learning_rate = params['h_learning_rate']
        h_decay = params['h_decay']
        h_hid_dim = params['h_hid_dim']
        h_fc_dim = params['h_fc_dim']
        h_seed = params['seed']
        h_layers = params['h_layers']

        loss_function = WeightedBinaryCrossEntropy(w1=0.82, w2=0.18)

        best_hyperparams = {}
        best_val_loss = float('inf')

        for dropout in h_dropout:
            for lr in h_learning_rate:
                for decay in h_decay:
                    for hid_dim in h_hid_dim:
                        for fc_dim in h_fc_dim:
                            for seed in h_seed:
                                for n_layers in h_layers:
                                    torch.cuda.empty_cache()
                                    counter_early_stop = 0
                                    best_epoch_val_loss = float('inf')

                                    # Initialize multimodal model
                                    model = models_multimodal.MultimodalGraphNet(
                                        n_layers=n_layers,
                                        dropout=dropout,
                                        hid_dim=hid_dim,
                                        S_temporal=S_temporal,
                                        S_static=S_static,
                                        in_dim_temporal=params['in_dim_GCN'],
                                        in_dim_static=params['in_dim_GCN'],
                                        embedding_dim_static=embedding_dim_static,
                                        embedding_dim_temporal=embedding_dim_temporal,
                                        fc_out_dim=fc_dim,
                                        seed=seed
                                    ).to(device)

                                    self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

                                    for epoch in range(n_epochs):
                                        # Training step
                                        train_loss = self.train(
                                            model, X_train_temporal, X_train_static, y_train, loss_function
                                        )

                                        # Validation step
                                        model.eval()
                                        with torch.no_grad():
                                            output_val = model(X_val_temporal, X_val_static).squeeze()
                                            val_loss = loss_function(output_val, y_val)

                                            if val_loss < best_epoch_val_loss:
                                                best_epoch_val_loss = val_loss
                                                counter_early_stop = 0
                                            else:
                                                counter_early_stop += 1

                                            if counter_early_stop >= early_stopping_patience:
                                                print(f"Early stopping at epoch {epoch}")
                                                break

                                    if best_epoch_val_loss < best_val_loss:
                                        best_val_loss = best_epoch_val_loss
                                        best_hyperparams = {
                                            'dropout': dropout,
                                            'lr': lr,
                                            'decay': decay,
                                            'hid_dim': hid_dim,
                                            'fc_dim': fc_dim,
                                            'seed': seed,
                                            'n_layers': n_layers
                                        }

        return best_hyperparams



class Tester:
    """
    Tester class for validating a multimodal GNN model on test data using specified hyperparameters.
    """

    def test(
        self,
        best_result_by_split: Dict[str, Any],
        params: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[Dict[str, List[float]], List[Any], Dict[str, torch.nn.Module], List[torch.Tensor]]:
        """
        Validate the model on the test set using the best hyperparameters.

        Parameters
        ----------
        best_result_by_split : dict
            Best hyperparameters for each split.
        params : dict
            Dictionary of training parameters and hyperparameters.
        device : torch.device
            Device to run the validation on (CPU or GPU).
        static : bool, optional
            If True, use static data. Default is False.

        Returns
        -------
        tuple
            - Results: Dictionary with test metrics (accuracy, ROC AUC, sensitivity, specificity).
            - Interpretability: List of interpretability-related outputs.
            - GNN models: Dictionary of trained GNN models by folder.
            - Embeddings: List of embeddings from the test set.
        """
        results = {'test_acc': [], 'roc_auc': [], 'sensitivity': [], 'specificity': []}
        interpretability = []
        gnn_models = {}
        embeddings = []

        n_epochs = params['n_epochs']
        early_stopping_patience = params['early_stopping_patience']
        folders = params["folders"]
        way_to_build_graph = params["way_to_build_graph"]

        # Define loss function
        loss_function = WeightedBinaryCrossEntropy(w1=0.82, w2=0.18)

        for folder in folders:
            torch.cuda.empty_cache()
            best_validation_loss = float('inf')
            counter_early_stop = 0
            print("Claves en best_result_by_split:", best_result_by_split.keys())

            best_hyperparameters = best_result_by_split[folder]

            seed=best_hyperparameters['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            X_train_temporal, X_val_temporal, X_test_temporal, \
            X_train_static, X_val_static, X_test_static, \
            y_train, y_val, y_test = load_data(device, folder)

            S_temporal = pd.read_csv(f"../../step2_graphRepresentation/{way_to_build_graph}/s{str(int(folder)+1)}/graph_Xtr_th_0.975.csv")
            S_static = pd.read_csv(f"../../step2_graphRepresentation/{way_to_build_graph}/s{str(int(folder)+1)}/static_graph_Xtr_th_0.975.csv")
                
            S_temporal = torch.tensor(S_temporal.values, dtype=torch.float32).to(device)
            S_static = torch.tensor(S_static.values, dtype=torch.float32).to(device)

            # Initialize model
            model = models_multimodal.MultimodalGraphNet(
                n_layers=best_hyperparameters['n_layers'],
                dropout=best_hyperparameters['dropout'],
                hid_dim=best_hyperparameters['hid_dim'],
                S_temporal=S_temporal,
                S_static=S_static,
                in_dim_temporal=params['in_dim_GCN'],
                in_dim_static=params['in_dim_GCN'],
                embedding_dim_static=params['embedding_dim_static'],
                embedding_dim_temporal=params['embedding_dim_temporal'],
                fc_out_dim=best_hyperparameters['fc_dim'],
                seed=best_hyperparameters['seed']
            ).to(device)

            # Define optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=best_hyperparameters['lr'],
                weight_decay=best_hyperparameters['decay']
            )

            # Training loop
            for epoch in range(n_epochs):
                model.train()
                optimizer.zero_grad()

                # Forward pass
                output_train = model(X_train_temporal, X_train_static).squeeze()
                loss_train = loss_function(output_train, y_train)
                loss_train.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    output_val = model(X_val_temporal, X_val_static).squeeze()
                    loss_val = loss_function(output_val, y_val)

                    # Early stopping
                    if loss_val < best_validation_loss:
                        best_validation_loss = loss_val
                        counter_early_stop = 0
                    else:
                        counter_early_stop += 1

                    if counter_early_stop >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            gnn_models[folder] = model

            # Testing
            model.eval()
            with torch.no_grad():
                output_test = model(X_test_temporal, X_test_static).squeeze()
                pred_test = torch.round(output_test).view(-1)

                # Interpretability outputs
                interpretability.append(output_test)

                # Embeddings
                embeddings.append(model(X_test_temporal, X_test_static))

                # Metrics
                test_acc = (pred_test == y_test).float().mean().item()
                roc_auc = roc_auc_score(y_test.cpu().numpy(), output_test.cpu().numpy())
                tn, fp, fn, tp = confusion_matrix(y_test.cpu().numpy(), pred_test.cpu().numpy()).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)

                results['test_acc'].append(test_acc)
                results['roc_auc'].append(roc_auc)
                results['sensitivity'].append(sensitivity)
                results['specificity'].append(specificity)

        return results, interpretability, gnn_models, embeddings
