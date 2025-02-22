import numpy as np
import pandas as pd
import sklearn

import random, os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, GRU, Dropout, Dense
from tensorflow.keras import backend as K
import pickle

import sys 
sys.path.append("../libraries/")
import utils


def build_dynamic_model(hyperparameters, n_timesteps, mask_value, n_dynamic_features, 
                hidden_layer_size, dropout_rate, lr_scheduler):
    # Dynamic preprocessing.
    dynamic_input = tf.keras.layers.Input(shape=(n_timesteps, n_dynamic_features,))
    masked = tf.keras.layers.Masking(mask_value=mask_value)(dynamic_input)
    gru_encoder = tf.keras.layers.GRU(
        hidden_layer_size,
        dropout=dropout_rate,
        return_sequences=False,
        activation='tanh',
        use_bias=False
    )(masked)
    
    # Concatenation
    output = tf.keras.layers.Dense(1, activation="sigmoid")(gru_encoder)
    
    model = tf.keras.Model([dynamic_input], [output])
    customized_loss = utils.weighted_binary_crossentropy(hyperparameters)
    myOptimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(loss=customized_loss, optimizer=myOptimizer)
    
    return model


def build_static_model(hyperparameters, n_timesteps, mask_value, n_static_features, 
                hidden_layer_size, dropout_rate, lr_scheduler):
    # Static preprocessing.
    static_input = tf.keras.layers.Input(shape=(n_static_features))
    hidden_layer = tf.keras.layers.Dense(
        hidden_layer_size,
        activation='tanh'
    )(static_input)
    
    # Concatenation
    output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden_layer)
    
    model = tf.keras.Model([static_input], [output])
    customized_loss = utils.weighted_binary_crossentropy(hyperparameters)
    myOptimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(loss=customized_loss, optimizer=myOptimizer)
    
    return model
    


def run_network(X_train, X_train_static, y_train, X_val, X_val_static, y_val, hyperparameters, seed, model_type='dynamic'):
    model = None
    if model_type == 'dynamic':
        model = build_dynamic_model(
            hyperparameters, hyperparameters["n_timesteps"], hyperparameters["maskValue"], hyperparameters["n_dynamic_features"], 
            hyperparameters["layers"], hyperparameters["dropout_rate"], hyperparameters["lr_scheduler"]
        )
    elif model_type == 'static':
        model = build_static_model(
            hyperparameters, hyperparameters["n_timesteps"], hyperparameters["maskValue"], hyperparameters["n_static_features"], 
            hyperparameters["layers"], hyperparameters["dropout_rate"], hyperparameters["lr_scheduler"]
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Please use 'static' or 'dynamic'.")
    
    try:
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=hyperparameters["monitor"],
            min_delta=hyperparameters["mindelta"],
            patience=hyperparameters["patience"], 
            restore_best_weights=True,
            mode="min"
        )

        # Fit the model based on the selected model_type
        if model_type == 'dynamic':
            hist = model.fit(
                x=X_train, y=y_train,
                validation_data=(X_val, y_val),
                callbacks=[earlystopping], batch_size=hyperparameters['batch_size'], epochs=hyperparameters['epochs'],
                verbose=hyperparameters['verbose'])
            
        elif model_type == 'static':
            hist = model.fit( x=[X_train_static.values], y=y_train,
                             validation_data=([X_val_static.values], y_val),
                             callbacks=[earlystopping], batch_size=hyperparameters['batch_size'], epochs=hyperparameters['epochs'], verbose=0
                            )

        return model, hist, earlystopping
    except KeyboardInterrupt:
        print('Training duration (s):', time.time() - global_start_time)
        return model, None, None




def myCVGrid(hyperparameters, dropout, lr_scheduler, layers, split, seed, path, model_type):
    bestHyperparameters = {}
    bestMetricDev = np.inf

    for k in range(len(dropout)):
        for l in range(len(layers)):
            for m in range(len(lr_scheduler)):
                v_early = []
                v_metric_dev = []
                v_hist = []
                v_val_loss = []

                hyperparameters_copy = hyperparameters.copy()
                hyperparameters_copy['dropout_rate'] = dropout[k]
                hyperparameters_copy['layers'] = layers[l]
                hyperparameters_copy['lr_scheduler'] = lr_scheduler[m]
                
                for n in range(5):

                    X_train = np.load(path + f'X_train_tensor_{n}.npy')
                    y_train = pd.read_csv(path + f'y_train_{n}.csv', index_col=0)
                    
                    X_val = np.load(path + f'X_val_tensor_{n}.npy')
                    y_val = pd.read_csv(path+ f'y_val_{n}.csv', index_col=0)

                    X_train_static = pd.read_csv(path + f'X_train_static_{n}.csv', index_col=0)
                    X_val_static = pd.read_csv(path + f'X_val_static_{n}.csv', index_col=0)

                    utils.reset_keras()
                    model, hist, early = run_network(
                        X_train, X_train_static, 
                        y_train,
                        X_val, X_val_static, 
                        y_val,
                        hyperparameters_copy,  
                        seed,
                        model_type
                    )
                                        
                    v_early.append(early)
                    v_hist.append(hist)
                    v_val_loss.append(np.min(hist.history["val_loss"]))
                    
                metric_dev = np.mean(v_val_loss)
                if metric_dev < bestMetricDev:
                    bestMetricDev = metric_dev
                    bestHyperparameters = {
                        'dropout_rate': dropout[k],
                        'layers': layers[l],
                        'lr_scheduler': lr_scheduler[m]
                    }

    return bestHyperparameters, X_train, y_train, X_train_static, X_val, y_val, X_val_static


## FUNCTIONS OF THE LR MODEL

def build_LR(hyperparameters):
    # Static preprocessing.
    static_input = tf.keras.layers.Input(shape=(2))
    # Output
    output = tf.keras.layers.Dense(1, activation="sigmoid")(static_input)
    
    model = tf.keras.Model([static_input], [output])
    customized_loss = utils.weighted_binary_crossentropy(hyperparameters)
    myOptimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"])
    model.compile(loss=customized_loss, optimizer=myOptimizer)
    
    return model


def run_network_LR(X_train, y_train, X_val, y_val, hyperparameters, seed):
    model = None
    model = build_LR(hyperparameters)
    try:
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=hyperparameters["monitor"],
            min_delta=hyperparameters["mindelta"],
            patience=hyperparameters["patience"],  # 30
            restore_best_weights=True,
            mode="min"
        )
    
        hist = model.fit(
            x=X_train, y=y_train,
            validation_data=(X_val, y_val),
            callbacks=[earlystopping], batch_size=hyperparameters['batch_size'], epochs=hyperparameters['epochs'],
            verbose=hyperparameters['verbose']
        )
                                
        return model, hist, earlystopping
    except KeyboardInterrupt:
        print('Training duration (s):', time.time() - global_start_time)
        return model, None, None


from sklearn.model_selection import KFold
def myCVGrid_LR(X_prev_train, y_prev_train, hyperparameters, lr_scheduler, seed):
    bestHyperparameters = {}
    bestMetricDev = np.inf

    for j in range(len(lr_scheduler)):
        v_early = []
        v_metric_dev = []
        v_hist = []
        v_val_loss = []
        kf = KFold(n_splits=2)

        hyperparameters_copy = hyperparameters.copy()
        hyperparameters_copy['lr_scheduler'] = lr_scheduler[j]
        
        for train, val in kf.split(X_prev_train):
            #Load train and validation
            X_train = X_prev_train.iloc[train]
            y_train = y_prev_train.iloc[train]
            
            X_val = X_prev_train.iloc[val]
            y_val = y_prev_train.iloc[val]
            
            # Train the model 
            model, hist, earlystopping = run_network_LR(X_train, 
                                                        y_train, 
                                                        X_val, 
                                                        y_val, 
                                                        hyperparameters_copy, 
                                                        seed)
            
            
            v_early.append(earlystopping)
            v_hist.append(hist)
            v_val_loss.append(np.min(hist.history["val_loss"]))
            
            metric_dev = np.mean(v_val_loss)
            if metric_dev < bestMetricDev:
                bestMetricDev = metric_dev
                bestHyperparameters = {
                    'lr_scheduler': lr_scheduler[j]
                }

    return bestHyperparameters, X_train, y_train, X_val, y_val


