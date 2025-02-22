import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV
import pickle

import logging
tf.get_logger().setLevel(logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append("../libraries/")
import utils



def build_model(hyperparameters):
    # static preprocessing.
    static_input = tf.keras.layers.Input(shape=(hyperparameters["n_static_features"]))
    hidden_layer = tf.keras.layers.Dense(
        hyperparameters["layers"],
        activation='tanh'
    )(static_input)
    dp_layer = tf.keras.layers.Dropout(hyperparameters["dropout_rate"], noise_shape=None, seed=42)(hidden_layer)
    
    # Concatenation
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dp_layer)
    
    model = tf.keras.Model([static_input], [output])
    customized_loss = utils.weighted_binary_crossentropy(hyperparameters)
    myOptimizer = myOptimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"])
    model.compile(loss=customized_loss, optimizer=myOptimizer)
    
    return model


def run_network(X_train, y_train,
                X_val, y_val,
                hyperparameters, seed):
    model = None
    model = build_model(hyperparameters)
    try:
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=hyperparameters["monitor"],
            min_delta=hyperparameters["mindelta"],
            patience=hyperparameters["patience"],
            restore_best_weights=True,
            mode="min"
        )

        hist = model.fit(
            x=[X_train], y=y_train,
            validation_data=([X_val], y_val),
            callbacks=[earlystopping], batch_size=hyperparameters['batch_size'], epochs=hyperparameters['epochs'],
            verbose=hyperparameters['verbose']
        )

        return model, hist, earlystopping
    except KeyboardInterrupt:
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0, 0


def myCVGrid(hyperparameters, dropout, lr_scheduler, layers, split, seed, path):
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

                    X_train = pd.read_csv(path + f'X_train_static_{n}.csv', index_col=0)
                    y_train = pd.read_csv(path + f'y_train_{n}.csv', index_col=0)
                    
                    X_val = pd.read_csv(path + f'X_val_static_{n}.csv', index_col=0)
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
                        'dropout_rate': dropout[k],
                        'layers': layers[l],
                        'lr_scheduler': lr_scheduler[m]
                    }

    return bestHyperparameters, X_train, y_train, X_val, y_val, 