#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from utils import *
import matplotlib.pyplot as plt
#%%
# MAPS = ['Stiffness','Volume','DR','Stiffness-Volume','Stiffness-DR','Volume-DR','Stiffness-Volume-DR']
# means = []
# stds = []
predictions = []
#%%
MAPS = ['Stiffness','Volume','DR','Stiffness-Volume-DR']
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
    
    # inference_times = []
    # for i in range(len(test_id)):
        # start = time.time()
        # y_pred = best_model_train([data[i:i+1] for data in test_data])
    y_pred_test = best_model_train(test_data)
    y_pred_test = y_pred_test.cpu().numpy().squeeze()
    predictions.append(y_pred_test)

#%%
ax_min, ax_max = np.min([test_target,y_pred_test]), np.max([test_target,y_pred_test])
ax_min, ax_max = np.floor(ax_min)-2, np.ceil(ax_max)+2
xy = np.arange(ax_min, ax_max)
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.plot(xy,xy, alpha=0.5, color='red', linestyle='--')
    # ax.scatter(test_target,y_pred_test, color='blue', alpha=0.8)
    # ax.set_xlim([ax_min, ax_max])
    # ax.set_ylim([ax_min, ax_max])
    # ax.set_xlabel('True age', fontsize=15)
    # ax.set_ylabel('Predicted age', fontsize=15)
    # ax.set_title('Model Predictions using {} maps'.format(MAPS[i]))
    # plt.savefig("predictions_{}.pdf".format(MAPS[i]),bbox_inches='tight')
        # print(y_pred)
        # end = time.time()
        # inference_times.append(end-start)
    # fig, ax = plt.subplots()
    # ax.plot(test_target,y_pred)
    # means.append(np.mean(inference_times))
    # stds.append(np.std(inference_times))
        # print('%d, %0.7f' % (i, (end-start)/10))
#%%
rows, cols = 2, 2
fig = plt.figure(figsize=(8,8), dpi=90)
gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
ax = gs.subplots(sharey='row', sharex='col')
k = 0
for i in range(rows):
    for j in range(cols):
        ax[i,j].plot(xy,xy, alpha=0.5, color='red', linestyle='--')
        ax[i,j].scatter(test_target,predictions[k], color='blue', alpha=0.8)
        ax[i,j].set_xlim([ax_min, ax_max])
        ax[i,j].set_ylim([ax_min, ax_max])
        ax[i,j].text(.01, .99, f'Input = [{MAPS[k]}]', ha='left', va='top', transform=ax[i,j].transAxes)
        if j==0:
            ax[i,j].set_ylabel('Predicted age')
        if i==1:
            ax[i,j].set_xlabel('True age')
        k += 1
        # ax[i,j].legend(loc='upper left')
plt.savefig("predictions.pdf".format(MAPS[i]),bbox_inches='tight')

#%%
rows, cols = 1, 4
fig = plt.figure(figsize=(13,3), dpi=90)
gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
ax = gs.subplots(sharey='row', sharex='col')
k = 0
for i in range(rows):
    for j in range(cols):
        ax[j].plot(xy,xy, alpha=0.5, color='red', linestyle='--')
        ax[j].scatter(test_target,predictions[k], color='blue', alpha=0.7)
        ax[j].set_xlim([ax_min, ax_max])
        ax[j].set_ylim([ax_min, ax_max])
        ax[j].text(.01, .99, f'Input = [{MAPS[k]}]', ha='left', va='top', transform=ax[j].transAxes)
        if j==0:
            ax[j].set_ylabel('Predicted age')
        # if i==1:
        ax[j].set_xlabel('True age')
        k += 1
        # ax[i,j].legend(loc='upper left')
plt.savefig("predictions_horizontal.pdf".format(MAPS[i]),bbox_inches='tight')

#%%
ax_min, ax_max = np.min([test_target,y_pred_test]), np.max([test_target,y_pred_test])
ax_min, ax_max = np.floor(ax_min)-2, np.ceil(ax_max)+2
xy = np.arange(ax_min, ax_max)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(xy,xy, alpha=0.5, color='red', linestyle='--')
ax.scatter(test_target,y_pred_test, color='blue', alpha=0.8)
ax.set_xlim([ax_min, ax_max])
ax.set_ylim([ax_min, ax_max])
ax.set_xlabel('True age')
ax.set_ylabel('Predicted age')

#%%
inference_times_df = pd.DataFrame(
    {
        'maps':MAPS,
        'mean':means,
        'std':stds
    }
)

# %%
