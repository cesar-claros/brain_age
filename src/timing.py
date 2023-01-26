#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from utils import *
#%%
MAPS = ['Stiffness','Volume','DR','Stiffness-Volume','Stiffness-DR','Volume-DR','Stiffness-Volume-DR']
means = []
stds = []
#%%
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
    test_data, test_target = data_dict['test_data'], data_dict['test_target']
    # Get best model trained only with training data
    path_modelcheckpoint = '/work/cniel/sw/BrainAge/results/models/best_{map}_model.h5'.format(map=map_type)
    print("Loading model")
    time.sleep(5)
    best_model_train = tf.keras.models.load_model(path_modelcheckpoint)
    time.sleep(5)
    
    inference_times = []
    for i in range(len(test_id)):
        start = time.time()
        y_pred_sample = best_model_train([data[i:i+1] for data in test_data])
        end = time.time()
        inference_times.append(end-start)
        
    means.append(np.mean(inference_times))
    stds.append(np.std(inference_times))
        # print('%d, %0.7f' % (i, (end-start)/10))

inference_times_df = pd.DataFrame(
    {
        'maps':MAPS,
        'mean':means,
        'std':stds
    }
)

# %%
