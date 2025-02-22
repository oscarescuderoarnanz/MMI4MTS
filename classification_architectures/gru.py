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


def build_model(layers, hyperparameters, lr_sch):    
    dynamic_input = tf.keras.layers.Input(shape=(hyperparameters["timeStep"], layers[0]))
    masked = tf.keras.layers.Masking(mask_value=hyperparameters['maskValue'])(dynamic_input)

    lstm_encoder = tf.keras.layers.GRU(
        layers[1],
        dropout=hyperparameters['dropout'],
        return_sequences=False,
        activation='tanh',
        use_bias=True
    )(masked)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(lstm_encoder)

    model = tf.keras.Model(dynamic_input, [output])
    myOptimizer = tf.keras.optimizers.Adam(learning_rate=lr_sch)
    customized_loss = utils.weighted_binary_crossentropy(hyperparameters)
    model.compile(loss=customized_loss, optimizer=myOptimizer)
        
    return model


def run_network(X_train, y_train, X_val, y_val, hyperparameters, seed):
    model = None
    model = build_model(hyperparameters['layers'], hyperparameters, hyperparameters['lr_scheduler'])
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
                hyperparameters_copy['dropout'] = dropout[k]
                hyperparameters_copy['layers'] = layers[l]
                hyperparameters_copy['lr_scheduler'] = lr_scheduler[m]
                
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
                        'layers': layers[l],
                        'lr_scheduler': lr_scheduler[m]
                    }

    return bestHyperparameters, X_train, y_train, X_val, y_val