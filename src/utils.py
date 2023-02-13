#%%
import nibabel as nib
import tensorflow as tf
# from tensorflow.keras import layers
import pathlib
import plotly
# from scipy.interpolate import RegularGridInterpolator
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import OneHotEncoder
# from keras.utils.vis_utils import plot_model
# from sklearn.metrics import mean_absolute_error as mae
# from numpy.random import seed
import os
import pandas as pd
# import sklearn
# import sklearn.model_selection
import numpy as np
# import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go
# from kerashypetune import KerasRandomSearchCV
# plotly.offline.init_notebook_mode(connected=True)
# plotly.io.renderers.default = "browser"
#%%
def common_path(arr, pos='prefix'):
    # The longest common prefix of an empty array is "".
    if not arr:
        print("Longest common", pos, ":", "")
    # The longest common prefix of an array containing 
    # only one element is that element itself.
    elif len(arr) == 1:
        print("Longest common", pos, ":", str(arr[0]))
    else:
        dir = range(len(arr[0])) if pos=="prefix" else range(-1,-len(arr[0])+1,-1)
        # Sort the array
        arr.sort()
        result = ""
        # Compare the first and the last string character
        # by character.
        for i in dir:
            #  If the characters match, append the character to
            #  the result.
            if arr[0][i] == arr[-1][i]:
                result += arr[0][i]
            # Else, stop the comparison
            else:
                break
    if pos=="suffix":
        result = result[::-1]
    print("Longest common", pos, ":", result)
    return result

def read_files(data_folder_path, label_folder_path, set_id, only_map=False):
    labels = pd.read_csv(label_folder_path+'labels_final.csv')
    labels_list = []
    map_list = []
    sex_list = []
    study_list = []
    meta_list = []
    for root, dirs, files in os.walk(data_folder_path):
        common_prefix = common_path(files, pos="prefix")
        common_suffix = common_path(files, pos="suffix")
        for id in set_id:
            age =  labels.loc[labels["ID"] == id,'Age'].to_numpy()[0]
            sex =  labels.loc[labels["ID"] == id,'Antipodal_Sex'].to_numpy()[0]
            study = labels.loc[labels["ID"] == id,'Study_ID'].to_numpy()[0]
            filename = common_prefix + str(id) + common_suffix
            if not os.path.exists(root+filename):
                filename = common_prefix + "{:0>3d}".format(id) + common_suffix
            nib_raw = nib.load(data_folder_path + filename)
            meta = nib_raw.header
            map = nib_raw.get_fdata()[:,:,:]
            labels_list.append(age)
            sex_list.append(sex)
            map_list.append(map)
            study_list.append(study)
            meta_list.append(meta)
    X_map = np.array(map_list).astype(np.float32)
    X_sex = np.array(sex_list)
    X_study = np.array(study_list)
    y = np.array(labels_list).astype(np.float32)
    m = np.array(meta_list)
    if only_map:
        output = X_map
    else:
        output = (X_map, X_sex, X_study, y, m)
    return output

def preprocess(X_train, X_test, X_val=None, preproc_type='std'):
    X_train_pp, X_test_pp = np.zeros_like(X_train), np.zeros_like(X_test)
    if preproc_type == 'scaling':
        X_train_pp = np.nan_to_num(X_train,nan=-750)/10000
        X_test_pp = np.nan_to_num(X_test,nan=-750)/10000

    elif preproc_type == 'std':
        mu = np.nanmean(X_train)
        sigma = np.nanstd(X_train)
        X_train_pp = (X_train-mu)/sigma
        mu_adj = np.nanmean(X_train_pp)
        X_train_pp[np.isnan(X_train_pp)] = mu_adj

        X_test_pp = (X_test-mu)/sigma
        X_test_pp[np.isnan(X_test_pp)] = mu_adj

        if type(X_val)==np.ndarray:
            X_val_pp = (X_val-mu)/sigma
            X_val_pp[np.isnan(X_val_pp)] = mu_adj

    elif preproc_type == 'max-min':
        delta = np.nanmax(X_train)-np.nanmin(X_train)
        min = np.nanmin(X_train)
        X_train_pp = (X_train-min)/delta
        mu_adj = np.nanmin(X_train_pp) #?
        X_train_pp[np.isnan(X_train_pp)] = mu_adj

        X_test_pp = (X_test-min)/delta
        X_test_pp[np.isnan(X_test_pp)] = mu_adj

    elif preproc_type == 'nan_to_num':
        # X_train_pp = np.nan_to_num(X_train,nan=-1)
        X_train_pp = X_train
        # X_test_pp = np.nan_to_num(X_test,nan=-1)
        X_test_pp = X_test
        if type(X_val)==np.ndarray:
            # X_val_pp = np.nan_to_num(X_val,nan=-1)
            X_val_pp = X_val
        print("unmodified maps")


    if type(X_val)==np.ndarray:
        ret = (X_train_pp, X_test_pp, X_val_pp)
    else:
        ret = (X_train_pp, X_test_pp)

    return ret


def plot_predictions(y_pred, y, map_type):
    x = np.linspace(np.floor(np.min([y_pred, y]))-5, np.ceil(np.max([y_pred, y]))+5,10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,y=x,mode='lines',
            opacity=0.5,
            line=dict(
                color='rgb(255, 0, 0)', 
                width=3, dash='dash')
                ))
    fig.add_trace(go.Scatter(x=y_pred,y=y,mode='markers',
            opacity=0.65,
            marker=dict(
                color='rgb(0, 0, 255)',
                size=15)
                ))
    fig.update_layout(autosize=False,width=500,height=500,showlegend=False,
                        margin=dict(l=10,t=60,r=10,b=10,pad=0),
                        title={
                            'text': "Brain age predictions using<br>{map} maps".format(map=map_type),
                            'x':0.5,
                            'xanchor': 'center',
                            'y':0.95
                            },
                        xaxis_title="Predicted Age",
                        yaxis_title="Actual Age",
                        font=dict(size=14),
                        )
    return fig

def data_loading(map_type, ids, path='/work/cniel/sw/BrainAge/datasets/', only_map=False, preproc='std'):
    # Define the map type
    # assert (map_type=='Stiffness' or map_type=='DR' or map_type=='Volume' or 
            # map_type=='Stiffness-Volume' or map_type=='Stiffness-DR' or map_type=='Volume-DR' or 
            # map_type=='Stiffness-Volume-DR')
    train_id, val_id, test_id = ids
    # Load brain maps
    folder_path_input = path+'{map}_FINAL/'.format(map=map_type) # maps path
    folder_path_labels = path # labels path
    print('Loading training set for {map} maps...'.format(map=map_type))
    X_train_map, X_train_sex, X_train_study, y_train, m_train = read_files(folder_path_input, folder_path_labels, train_id)
    print('Loading validation set for {map} maps...'.format(map=map_type))
    X_val_map, X_val_sex, X_val_study, y_val, m_val = read_files(folder_path_input, folder_path_labels, val_id)
    print('Loading test set for {map} maps...'.format(map=map_type))
    X_test_map, X_test_sex, X_test_study, y_test, m_test = read_files(folder_path_input, folder_path_labels, test_id)
    # Preprocessing map 
    X_train_pp, X_test_pp, X_val_pp = preprocess(X_train_map, X_test_map, X_val_map, preproc_type=preproc)
    # One hot encoding for categorical variables
    # define one hot encoding
    if not only_map:
        encoder = OneHotEncoder(sparse=False)
        # transform categorical variables
        X_train_sex = encoder.fit_transform(X_train_sex.reshape(-1,1))
        X_train_study = encoder.fit_transform(X_train_study.reshape(-1,1))
        X_train_cat = np.concatenate((X_train_sex,X_train_study), axis=1)
        X_val_sex = encoder.fit_transform(X_val_sex.reshape(-1,1))
        X_val_study = encoder.fit_transform(X_val_study.reshape(-1,1))
        X_val_cat = np.concatenate((X_val_sex,X_val_study), axis=1)
        X_test_sex = encoder.fit_transform(X_test_sex.reshape(-1,1))
        X_test_study = encoder.fit_transform(X_test_study.reshape(-1,1))
        X_test_cat = np.concatenate((X_test_sex,X_test_study), axis=1)
        # Arranging data for CNN input 
        train_data = [X_train_pp, X_train_cat]
        train_target = y_train
        val_data = [X_val_pp, X_val_cat]
        val_target = y_val
        test_data = [X_test_pp, X_test_cat]
        test_target = y_test
        output = (train_data, train_target, val_data, val_target, test_data, test_target)
    else:
        train_data = X_train_pp
        val_data = X_val_pp
        test_data = X_test_pp
        output = (train_data, val_data, test_data)

    return output

def data_splitting(map_type, ids):
    
    if map_type == 'Stiffness' or map_type == 'Volume' or map_type =='DR':
        map_types = map_type.split('-')
        train_data, train_target, val_data, val_target, test_data, test_target = data_loading(map_type, ids)
        # Fit with the entire dataset.
        X_train_pp_all = np.concatenate((train_data[0], val_data[0]))
        X_train_cat_all = np.concatenate((train_data[-1], val_data[-1]))
        train_data_all = [X_train_pp_all, X_train_cat_all]
    elif map_type == 'Stiffness-Volume' or map_type == 'Stiffness-DR' or map_type == 'Volume-DR':
        map_types = map_type.split('-')
        train_data, train_target, val_data, val_target, test_data, test_target = data_loading(map_types[0], ids)
        train_data_1, val_data_1, test_data_1 = data_loading(map_types[1], ids, only_map=True)
        train_data.insert(1, train_data_1)
        val_data.insert(1, val_data_1)
        test_data.insert(1, test_data_1)
        # Fit with the entire dataset.
        X_train_pp_all = np.concatenate((train_data[0], val_data[0]))
        X_train_pp_all_1 = np.concatenate((train_data[1], val_data[1]))
        X_train_cat_all = np.concatenate((train_data[-1], val_data[-1]))
        train_data_all = [X_train_pp_all, X_train_pp_all_1, X_train_cat_all]
    elif map_type == 'Stiffness-Volume-DR':
        map_types = map_type.split('-')
        train_data, train_target, val_data, val_target, test_data, test_target = data_loading(map_types[0], ids)
        train_data_1, val_data_1, test_data_1 = data_loading(map_types[1], ids, only_map=True)
        train_data_2, val_data_2, test_data_2 = data_loading(map_types[2], ids, only_map=True)
        train_data.insert(1, train_data_1)
        val_data.insert(1, val_data_1)
        test_data.insert(1, test_data_1)
        train_data.insert(2, train_data_2)
        val_data.insert(2, val_data_2)
        test_data.insert(2, test_data_2)
        # Fit with the entire dataset.
        X_train_pp_all = np.concatenate((train_data[0], val_data[0]))
        X_train_pp_all_1 = np.concatenate((train_data[1], val_data[1]))
        X_train_pp_all_2 = np.concatenate((train_data[2], val_data[2]))
        X_train_cat_all = np.concatenate((train_data[-1], val_data[-1]))
        train_data_all = [X_train_pp_all, X_train_pp_all_1, X_train_pp_all_2, X_train_cat_all]
    y_train_all = np.concatenate((train_target, val_target))
    train_target_all = y_train_all
    data_dict = {
        'train_data': train_data, 
        'train_target': train_target,
        'val_data': val_data,
        'val_target': val_target,
        'test_data': test_data,
        'test_target': test_target,
        'train_data_all': train_data_all,
        'train_target_all': train_target_all
    }
    return data_dict