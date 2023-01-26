import pandas as pd
import numpy as np
import tensorflow as tf
import wandb
# from wandb.keras import WandbCallback
import os
# from utils import *
# from tuner import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tuner import build_model
import time

def model_evaluation(data_dict, map_type, tuner,  path_modelcheckpoint, seed=12345):
    val_data, val_target = data_dict['val_data'], data_dict['val_target']
    test_data, test_target = data_dict['test_data'], data_dict['test_target']
    train_data_all, train_target_all = data_dict['train_data_all'], data_dict['train_target_all']
    # Get best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    # Get best model trained only with training data
    best_model_train = tf.keras.models.load_model(path_modelcheckpoint)
    # Get predictions from best model during training
    y_pred_test = best_model_train.predict(test_data).squeeze()
    y_pred_val = best_model_train.predict(val_data).squeeze()

    # Get best epoch and best mae
    val_maes = [tuner.oracle.trials[id].score for id in tuner.oracle.trials]
    val_epochs = [tuner.oracle.trials[id].best_epoch for id in tuner.oracle.trials]
    best_idx = np.argmin(val_maes)
    best_epochs_across_trials = val_epochs[best_idx]

    # Get best model trained only with training data and retrain with all data
    tf.keras.utils.set_random_seed(seed)
    best_model_train_val_1 = tf.keras.models.load_model(path_modelcheckpoint)
    tf.keras.utils.set_random_seed(seed)
    best_model_train_val_1.fit(x=train_data_all, y=train_target_all, epochs=best_epochs_across_trials)
    # Get predictions from best model during training
    y_pred_test_1 = best_model_train_val_1.predict(test_data).squeeze()
    # Save model 1
    path_model_1 = '/work/cniel/sw/BrainAge/results/models/best_{map}_model_1.h5'.format(map=map_type)
    best_model_train_val_1.save(path_model_1)

    # Get best model trained only with training data and retrain with all data
    tf.keras.utils.set_random_seed(seed)
    best_model_train_val_2 = build_model(best_hyperparameters)
    tf.keras.utils.set_random_seed(seed)
    best_model_train_val_2.fit(x=train_data_all, y=train_target_all, epochs=2*best_epochs_across_trials)
    # Get predictions from best model during training
    y_pred_test_2 = best_model_train_val_2.predict(test_data).squeeze()
    # Save model 2
    path_model_2 = '/work/cniel/sw/BrainAge/results/models/best_{map}_model_2.h5'.format(map=map_type)
    best_model_train_val_2.save(path_model_2)

    # Store performance metrics
    # Temporal dataframe for results
    df_results_temp = pd.DataFrame(
        {
            'Map': map_type,
            'Hyperparameters': {0:best_hyperparameters.values},
            'R2 score (val)': r2_score(val_target,y_pred_val),
            'MAE (val)': mean_absolute_error(val_target,y_pred_val),
            'RMSE (val)': mean_squared_error(val_target,y_pred_val, squared=False),
            'R2 score (test)': r2_score(test_target,y_pred_test),
            'MAE (test)': mean_absolute_error(test_target,y_pred_test),
            'RMSE (test)': mean_squared_error(test_target,y_pred_test, squared=False),
            'R2 score (test*)': r2_score(test_target,y_pred_test_1),
            'MAE (test*)': mean_absolute_error(test_target,y_pred_test_1),
            'RMSE (test*)': mean_squared_error(test_target,y_pred_test_1, squared=False),
            'R2 score (test**)': r2_score(test_target,y_pred_test_2),
            'MAE (test**)': mean_absolute_error(test_target,y_pred_test_2),
            'RMSE (test**)': mean_squared_error(test_target,y_pred_test_2, squared=False)
        }
    )
    df_ages_temp = pd.DataFrame({
            "Actual Age":test_target, "Predicted Age":y_pred_test, 
            "Predicted Age (*)":y_pred_test_1,
            "Predicted Age (**)":y_pred_test_2, 
            "Map":map_type
            })

    # df_results = pd.concat([df_results,df_results_temp])
    # df_ages = pd.concat([df_ages, df_ages_temp])
    # Results logs
    run = wandb.init(settings=wandb.Settings(start_method="fork"),
                        reinit=True, project="brain_age", 
                        group="test-set", name='results-{}'.format(map_type),
                        config=best_hyperparameters.values, dir=os.getenv('WANDB_DIR'))
    wandb.log({"Table: Predictions in test set": wandb.Table(data=df_ages_temp)})
    wandb.log({"Table: Metrics in test set": wandb.Table(data=df_results_temp)})

    run.finish()

def model_timing(data_dict, map_type, path_modelcheckpoint, seed=12345):
    test_data, test_target = data_dict['test_data'], data_dict['test_target']
    # Get best model trained only with training data
    best_model_train = tf.keras.models.load_model(path_modelcheckpoint)
    # Get predictions from best model during training
    y_pred_test = best_model_train.predict(test_data).squeeze()
