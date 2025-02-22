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
import TFT_library_masking as TFT



def build_model(hyperparameters, n_timesteps, mask_value, n_dynamic_features, n_static_features, category_counts, 
                hidden_layer_size, dropout_rate, num_heads, lr_scheduler):
    # Inputs.
    static_input = tf.keras.layers.Input(shape=(n_static_features))
    dynamic_input = tf.keras.layers.Input(shape=(n_timesteps, n_dynamic_features,))
    masked = tf.keras.layers.Masking(mask_value=mask_value)(dynamic_input)

    # Get the embeddings
    static_emb, dynamic_emb = TFT.get_TFT_embeddings(static_input, n_static_features, category_counts,
                                                     masked, n_dynamic_features,
                                                     hidden_layer_size)

    # Perform the FS of the static features and pass through a GRN
    static_encoder, static_weights = TFT.static_combine_and_mask(static_emb, hidden_layer_size, dropout_rate)

    static_context_h = TFT.gated_residual_network(static_encoder,
                                                    hidden_layer_size,
                                                    dropout_rate=dropout_rate,
                                                    use_time_distributed=False)


    lstm_encoder = tf.keras.layers.GRU(
        hidden_layer_size,
        dropout=dropout_rate,
        return_sequences=False,
        activation='tanh',
        use_bias=True
    )(masked,
     initial_state=[static_context_h])
    
    output = tf.keras.layers.Dense(1, activation="sigmoid")(lstm_encoder)
    
    model = tf.keras.Model([static_input, dynamic_input], [output])
    customized_loss = utils.weighted_binary_crossentropy(hyperparameters)

    myOptimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(loss=customized_loss, optimizer=myOptimizer)
    
    return model


def run_network(X_train, X_train_static, y_train, X_val, X_val_static, y_val, hyperparameters, seed):
    
    model = None
    model = build_model(
        hyperparameters, hyperparameters["n_timesteps"], hyperparameters["maskValue"],
        hyperparameters["n_dynamic_features"], hyperparameters["n_static_features"],  hyperparameters["category_counts"], 
        hyperparameters["layers"], hyperparameters["dropout_rate"], hyperparameters["num_heads"], hyperparameters["lr_scheduler"]
    )
    try:
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=hyperparameters["monitor"],
            min_delta=hyperparameters["mindelta"],
            patience=hyperparameters["patience"], 
            restore_best_weights=True,
            mode="min"
        )

        hist = model.fit(
            x=[X_train_static.values, X_train], y=y_train,
            validation_data=([X_val_static.values, X_val], y_val),
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

    return bestHyperparameters, X_train, y_train, X_train_static, X_val, y_val, X_val_static
    