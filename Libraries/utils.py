import tensorflow as tf
import numpy as np
import random
import pandas as pd
from tensorflow.keras import backend as K
import logging
tf.get_logger().setLevel(logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

## WEIGHTED BINARY CROSS ENTROPY ##
def weighted_binary_crossentropy(hyperparameters):
    w1 = hyperparameters["w1"]
    w2 = hyperparameters["w2"]
    """
    Binary form of weighted binary cross entropy.
      WBCE(p_t) = -w * (1 - p_t)* log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    Usage:
     model.compile(loss=[weighted_binary_crossentropyv2(hyperparameters)], metrics=["accuracy"], optimizer=adam)
    """
    def loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(w1 * (1. - pt_1) * K.log(pt_1)) \
               -K.sum(w2 * (pt_0) * K.log(1. - pt_0))

    return loss


## RESET KERAS ##
def reset_keras(seed=42):
    K.clear_session()
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)

## CALCULATE METRICS ##
def calculate_and_save_metrics(y_true, y_pred, split_directory, split_index):
    """
    Calculates metrics and saves the results to a CSV file.
    
    Parameters:
    - y_true: array with the real values of the label.
    - y_pred: array with the model predictions.
    - split_directory: str, directory where the metrics will be saved.
    - split_index: index of the current split.
    
    Returns:
    - A dictionary with the calculated metrics.
    """

    accuracy_test = accuracy_score(y_true, np.round(y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred)).ravel()
    roc = roc_auc_score(y_true, y_pred)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1score = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
    
    metrics_dict = {
        'accuracy': accuracy_test,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1score': f1score,
        'roc_auc': roc
    }
    
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(os.path.join(split_directory, f"metrics_split{split_index}.csv"), index=False)
    
    return metrics_dict