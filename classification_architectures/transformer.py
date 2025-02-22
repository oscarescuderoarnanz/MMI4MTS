import numpy as np
import pandas as pd
import sklearn

import random, os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
import sys
sys.path.append("../libraries/")
import utils

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, Dropout, Dense, Dropout, Flatten, Conv1D
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers



def build_model(hyperparameters):
    """
    Builds a Transformer model based on the provided training data and hyperparameters.
    Args:
        - hyperparameters: Dictionary containing hyperparameters.
    Returns:
        - model: A tf.keras.Model with the compiled model.
    """
    
    # CONSIDER THE HYPERPARAMETERS
    hyperparameters['layers'] = [hyperparameters['input_shape'], hyperparameters['middle_layer_dim'], 1]
    dropout = hyperparameters["dropout"]
    num_heads = hyperparameters["num_heads"]
    num_transformer_blocks = hyperparameters["num_transformer_blocks"]


    input = Input(shape=(hyperparameters["timeStep"], hyperparameters["layers"][0]))
    x = input
    
    masked = tf.keras.layers.Masking(mask_value=666)(x)  # MASKING LAYER

    for _ in range(num_transformer_blocks):
        # NORMALIZATION AND ATTENTION
        x_norm = layers.LayerNormalization()(masked)
        x_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input.shape[-1])(x_norm, x_norm)
        x_att_drop = layers.Dropout(dropout)(x_att)
        res = x_att_drop + masked

        # FEED FORWARD PART
        x_ffn_norm = layers.LayerNormalization()(res)
        x_ffn = layers.Dense(
            input.shape[-1], activation='tanh'
        )(x_ffn_norm)
        x = x_ffn + res

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(
        hyperparameters["layers"][1], activation='tanh'
    )(x)
    x = layers.Dropout(dropout)(x)

    output = layers.GlobalMaxPooling1D()(x)  
    output = layers.Dense(1, activation='sigmoid')(output)  # Output: (None, 1)

    model = Model(input, output)
    
    myOptimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"])
    customized_loss = utils.weighted_binary_crossentropy(hyperparameters)
    model.compile(loss=customized_loss, optimizer=myOptimizer)
    
    # COMPILE 
    model.compile(
        loss = customized_loss,
        optimizer=myOptimizer
    )

    
    return model


def run_network(X_train, y_train, X_val, y_val, hyperparameters, seed):
    model = None
    model = build_model(hyperparameters)
    earlystopping = None
    try:
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=hyperparameters["mindelta"],
                                                      patience=hyperparameters["patience"],
                                                      restore_best_weights=True,
                                                      mode="min")
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         callbacks=[earlystopping], batch_size=hyperparameters['batch_size'], epochs=hyperparameters['epochs'],
                         verbose=hyperparameters['verbose'])
        
        return model, hist, earlystopping
    except KeyboardInterrupt:
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0, 0



def myCVGrid(hyperparameters, dropout, lr_scheduler, middle_layer_dim, num_transformer_blocks,
             num_heads, epsilon, split, seed, path):
    bestHyperparameters = {}
    bestMetricDev = np.inf

    for k in range(len(dropout)):
        for l in range(len(middle_layer_dim)):
            for m in range(len(lr_scheduler)):
                for o in range(len(num_transformer_blocks)):
                    for p in range(len(num_heads)):
                        for q in range(len(epsilon)):
                            v_early = []
                            v_metric_dev = []
                            v_hist = []
                            v_val_loss = []

                            hyperparameters_copy = hyperparameters.copy()
                            hyperparameters_copy['dropout'] = dropout[k]
                            hyperparameters_copy['middle_layer_dim'] = middle_layer_dim[l]
                            hyperparameters_copy['lr_scheduler'] = lr_scheduler[m]
                            hyperparameters_copy['num_transformer_blocks'] = num_transformer_blocks[o]
                            hyperparameters_copy['num_heads'] = num_heads[p]
                            hyperparameters_copy['epsilon'] = epsilon[q]
                            
                            for n in range(5):
            
                                X_train = np.load(path + f'X_train_tensor_{n}.npy')
                                y_train = pd.read_csv(path + f'y_train_{n}.csv', index_col=0)
                                
                                X_val = np.load(path + f'X_val_tensor_{n}.npy')
                                y_val = pd.read_csv(path + f'y_val_{n}.csv', index_col=0)


                                utils.reset_keras()
                                model, hist, early = run_network(
                                    X_train,
                                    y_train,
                                    X_val,
                                    y_val,
                                    hyperparameters_copy,
                                    seed
                                )

                                v_early.append(early)
                                v_hist.append(hist)
                                v_val_loss.append(np.min(hist.history["val_loss"]))

                            metric_dev = np.mean(v_val_loss)
                            if metric_dev < bestMetricDev:
                                bestMetricDev = metric_dev
                                bestHyperparameters = {
                                    'dropout': dropout[k],
                                    'middle_layer_dim': middle_layer_dim[l],
                                    'lr_scheduler': lr_scheduler[m],
                                    'num_transformer_blocks': num_transformer_blocks[o],
                                    'num_heads': num_heads[p],
                                    'epsilon': epsilon[q]
                                }

    return bestHyperparameters, X_train, y_train, X_val, y_val
