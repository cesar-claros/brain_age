#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from utils import *
import matplotlib.pyplot as plt

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

# %%
i = 6
map_types = MAPS[i].split('-')
# train_data, train_target, val_data, val_target, test_data, test_target = data_loading(map_type, ids, preproc='nan_to_num')
train_data, train_target, val_data, val_target, test_data, test_target = data_loading(map_types[0], ids, preproc='nan_to_num')
train_data_1, val_data_1, test_data_1 = data_loading(map_types[1], ids, only_map=True, preproc='nan_to_num')
train_data_2, val_data_2, test_data_2 = data_loading(map_types[2], ids, only_map=True, preproc='nan_to_num')
train_data_1 = np.where(train_data_1!=0.0,train_data_1,np.nan)

# %%
ages, idx = np.unique(train_target,return_index=True)
subset_inc = 11
subset_ages, subset_idx = ages[::subset_inc], idx[::subset_inc]
# slice_number = [18,18,18,18,18,18,18]

#%%
maps = ['Stiffness','Volume','Damping\nratio']
color_maps = ['viridis','cividis','magma']

coordinates = [[0.88, 0.63, 0.04, 0.22],[0.88, 0.37, 0.04, 0.22],[0.88, 0.11, 0.04, 0.22]]
fig = plt.figure(figsize=(10,5), dpi=90)
gs = fig.add_gridspec(3, len(subset_ages), hspace=0, wspace=0)
ax = gs.subplots(sharey='row', sharex='col')
# fig.suptitle('Sharing x per column, y per row')
for i, data in enumerate([train_data[0], train_data_1, train_data_2]):
    max_counts_subjects_slices = np.argmin(np.count_nonzero(np.isnan(train_data[0]), axis=(1,2)), axis=1)
    slice_number = max_counts_subjects_slices[subset_idx]
    min_value, max_value = np.nanmin(data[subset_idx,:,:,slice_number]), np.nanmax(data[subset_idx,:,:,slice_number])
    for j in range(len(subset_ages)):
        map = data[subset_idx[j],:,:,slice_number[j]]
        map = np.fliplr(np.where(np.isnan(map),0,map))
        im = ax[i,j].imshow(map.T, cmap=color_maps[i], aspect='auto', vmin=min_value, vmax=max_value)
        if i==2:
            ax[i,j].set_xlabel('Age={:d}'.format(int(subset_ages[j])))
        if j==0:
            ax[i,j].set_ylabel('{}'.format(maps[i]))
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])

    # add space for colour bar
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes(coordinates[i])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.formatter.set_powerlimits((0,0))
plt.savefig("brain_maps.pdf",bbox_inches='tight')
    # cbar.set_label('{}'.format(maps[i]))
    # fig.set_facecolor("red")
# plt.colorbar()
# ax1.plot(x, y)
# ax2.plot(x, y**2, 'tab:orange')
# ax3.plot(x + 1, -y, 'tab:green')
# ax4.plot(x + 2, -y**2, 'tab:red')

# for ax in fig.get_axes():
    # ax.label_outer()
#%%
# %%
