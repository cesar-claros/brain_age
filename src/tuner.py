import keras_tuner
import wandb
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from wandb.keras import WandbCallback
from model import model_def
from utils import plot_predictions

#%%
# import tf.keras.backend as K


def build_model(hp):
    # Default parameter grid
    param_grid = {
        'num_input_maps': hp.Fixed('num_input_maps', value=1),
        # 'inputs': hp.Choice('inputs', values=['Stiffness','Volume','DR','Stiffness-Volume','Stiffness-DR','Volume-DR','Stiffness-Volume-DR'], ordered=False, default='Stiffness'),
        'arc_type' : hp.Choice('arc_type', values=['1', '2', '3', '4'], ordered=False, default='1'),
        'cat_input_type': hp.Choice('cat_input_type', values=['None', 'sex', 'study', 'sex_study'], ordered=False, default='sex_study'),
        'lr' : hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log"),
        'batch_size' : hp.Int("batch_size", 4, 28, step=4)
    }
    model = model_def(param_grid)
    model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=param_grid['lr'], beta_1=0.9, 
                                        beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                        name='Adam'), 
    # loss = tf.keras.losses.MeanAbsoluteError(),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = [tf.keras.metrics.RootMeanSquaredError(), 
                tf.keras.metrics.MeanAbsoluteError()
                ])
    return model

#%%
class MyTuner(keras_tuner.Tuner):

    # def run_trial(self, trial, trainX, trainY, valX, valY, batch_size, epochs, objective):
    def run_trial(self, trial, trainX, trainY, valX, valY, epochs, objective, map_type, cb_chkpnt, seed=12345):
        hp = trial.hyperparameters
        objective_name_str = objective.name

        ## create the model with the current trial hyperparameters
        tf.keras.utils.set_random_seed(seed)
        model = self.hypermodel.build(hp)

        ## Initiates new run for each trial on the dashboard of Weights & Biases
        run = wandb.init(settings=wandb.Settings(start_method="fork"),
                         reinit=True,
                         project="brain_age", group="{map}".format(map=map_type),
                         config=hp.values, 
                         dir=os.getenv('WANDB_DIR'))

        ## batch size hyper-parameter definition 
        batch_size = hp.get("batch_size")
        ## WandbCallback() logs all the metric data such as
        ## loss, accuracy and etc on dashboard for visualization
        tf.keras.utils.set_random_seed(seed)
        history = model.fit(trainX,
                  trainY,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(valX,valY),
                  callbacks=[WandbCallback(monitor='val_mean_absolute_error',save_model=False),cb_chkpnt])  
        # y_pred_val = model.predict(valX).squeeze()
        # df_ages = pd.DataFrame({"Predicted Age":y_pred_val, "Actual Age":valY, "Map":map_type, "Run":run.name})
        # wandb.log({"Table: Predictions in validation set" : wandb.Table(data=df_ages)})
        # wandb.log({"Plots: Predictions in validation set" : plot_predictions(y_pred_val, valY, map_type)})
    
        ## if val_accurcy used, use the val_accuracy of last epoch model which is fully trained
        val_mae_per_epoch = history.history['val_mean_absolute_error']
        best_val_mae = np.min(val_mae_per_epoch)  ## minimum value metric epoch
        best_epoch = val_mae_per_epoch.index(best_val_mae)+1

        # Ensure metrics of trials are updated locally.
        keras_tuner_trial = self.oracle.trials[trial.trial_id]
        keras_tuner_trial.best_epoch = best_epoch
        ## Send the objective data to the oracle for comparison of hyperparameters
        # self.oracle.update_trial(trial.trial_id, {'val_epoch':best_epoch})
        self.oracle.update_trial(trial.trial_id, {objective_name_str:best_val_mae})
        ## save the trial model
        # self.save_model(trial.trial_id, model)
        
        ## ends the run on the Weights & Biases dashboard
        run.finish()
