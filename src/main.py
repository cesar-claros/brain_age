#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner
import wandb
# from wandb.keras import WandbCallback
import os
from utils import *
from tuner import *
from keras_tuner import HyperParameters
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from model_eval import model_evaluation

# %%
# Define seed
SEED = 12345
# Define preprocessing procedure
PREPROC_TYPE = 'std'
# Epochs for search 
EPOCHS_SEARCH = 60
# Bayesian Optimization parameters
MAX_TRIALS_BO = 60
INIT_POINTS_BO = 10
# Maps used in search
MAPS = ['Stiffness','Volume','DR','Stiffness-Volume','Stiffness-DR','Volume-DR','Stiffness-Volume-DR']
# MAPS = ['Stiffness-Volume-DR']
# Define env variables
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = ""
os.environ['WANDB_CACHE_DIR'] = ''
os.environ['WANDB_CONFIG_DIR'] = ''
os.environ['WANDB_DIR'] = ''

#%%
def main():
    print("Starting training process...")
    print("Loading IDs...")
    # IDs splitting
    train_id = pd.read_csv('/work/cniel/sw/BrainAge/splits/train_split.csv', delimiter=',', header=None).to_numpy().squeeze()
    val_id = pd.read_csv('/work/cniel/sw/BrainAge/splits/val_split.csv', delimiter=',', header=None).to_numpy().squeeze()
    test_id = pd.read_csv('/work/cniel/sw/BrainAge/splits/test_split.csv', delimiter=',', header=None).to_numpy().squeeze()
    ids = (train_id, val_id, test_id)
    
    for i, map_type in enumerate(MAPS):
        # Data preparation
        data_dict = data_splitting(map_type, ids)
        train_data, train_target = data_dict['train_data'], data_dict['train_target'] 
        val_data, val_target = data_dict['val_data'], data_dict['val_target']

        # Define model
        obj = keras_tuner.Objective("val_mean_absolute_error", direction="min")
        ## instantiate the new Tuner with tuning algorithm and required parameters
        hp = HyperParameters()
        hp.Fixed('num_input_maps', value=len(map_type.split('-')))
        tuner = MyTuner(
                    oracle=keras_tuner.oracles.BayesianOptimization(
                    objective=obj,
                    max_trials=MAX_TRIALS_BO,
                    num_initial_points=INIT_POINTS_BO,
                    hyperparameters=hp,
                    seed=SEED),
                hypermodel=build_model,
                overwrite=True,
                directory='/work/cniel/sw/BrainAge/results',
                project_name='brain_age_{map}'.format(map=map_type))
        # Callback for Model Checkpoint
        path_modelcheckpoint = '/work/cniel/sw/BrainAge/results/models/best_{map}_model.h5'.format(map=map_type)
        mc = tf.keras.callbacks.ModelCheckpoint(path_modelcheckpoint, monitor='val_mean_absolute_error', 
                                                mode='min', verbose=1, 
                                                save_best_only=True)
        ## initiates the hyperparameter tuning process
        tf.keras.utils.set_random_seed(SEED)
        tuner.search(train_data, train_target, val_data, val_target, 
                        epochs=EPOCHS_SEARCH, objective=obj, map_type=map_type,
                        cb_chkpnt=mc, seed=SEED)
        model_evaluation(data_dict, map_type, tuner, path_modelcheckpoint, seed=SEED)


if __name__ == "__main__":
    main()