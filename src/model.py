import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def model_def(param):
    # Add categorical input type
    cat_shape = 10 
    if param['cat_input_type'] == 'None':
        condition = lambda x: x
    elif param['cat_input_type'] == 'sex':
        condition = lambda x: x[:,:2]
    elif param['cat_input_type'] == 'study':
        condition = lambda x: x[:,2:]
    elif param['cat_input_type'] == 'sex_study':
        condition = lambda x: x
    else:
        raise Exception('Categorical input type selected is undefined')

    if param['num_input_maps'] == 1: 
        map_model = tf.keras.Sequential()
        map_model.add(tf.keras.Input(shape=(91,109,55)))
        map_model.add(layers.Reshape((91,109,55,1)))
        x = map_model.output
        model_inputs = [map_model.input]
    elif param['num_input_maps'] == 2:
        map_model_1 = tf.keras.Sequential()
        map_model_1.add(tf.keras.Input(shape=(91,109,55)))
        map_model_1.add(layers.Reshape((91,109,55,1)))
        map_model_2 = tf.keras.Sequential()
        map_model_2.add(tf.keras.Input(shape = (91,109,55)))
        map_model_2.add(layers.Reshape((91,109,55,1)))
        x = tf.keras.layers.concatenate([map_model_1.output, map_model_2.output])
        model_inputs = [map_model_1.input, map_model_2.input]
    elif param['num_input_maps'] == 3:
        map_model_1 = tf.keras.Sequential()
        map_model_1.add(tf.keras.Input(shape=(91,109,55)))
        map_model_1.add(layers.Reshape((91,109,55,1)))
        map_model_2 = tf.keras.Sequential()
        map_model_2.add(tf.keras.Input(shape = (91,109,55)))
        map_model_2.add(layers.Reshape((91,109,55,1)))
        map_model_3 = tf.keras.Sequential()
        map_model_3.add(tf.keras.Input(shape = (91,109,55)))
        map_model_3.add(layers.Reshape((91,109,55,1)))
        x = tf.keras.layers.concatenate([map_model_1.output, map_model_2.output, map_model_3.output])
        model_inputs = [map_model_1.input, map_model_2.input, map_model_3.input]
    
    initializer = lambda n_chan: tf.keras.initializers.he_normal()
    n_chan = 8
    n = 1

    if param['arc_type'] == '1':
        map_shape = x.get_shape()
        model_cat = tf.keras.Sequential()
        model_cat.add(tf.keras.Input(shape=cat_shape))
        model_cat.add(layers.Lambda(condition))
        model_cat.add(layers.Dense(np.prod(map_shape[1:])))
        model_cat.add(layers.Reshape(map_shape[1:]))
        if param['cat_input_type'] == 'None':
            x = x
        else:
            x = layers.Add()([x,model_cat.output])
    
    for layer in range(5):
        x = layers.Conv3DTranspose(2**layer*n_chan, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer=initializer(n_chan), use_bias=False)(x)
        x = layers.ReLU()(x)
        x = layers.Conv3DTranspose(2**layer*n_chan, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer=initializer(n_chan), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    
    concat_layer_shape = x.get_shape()# model.layers
    if param['arc_type'] == '1':
        x = layers.Reshape((np.prod(concat_layer_shape[1:4])*(concat_layer_shape[-1]),))(x)
        x = layers.Dense(640, activation="relu")(x) # try activation="softplus"
        x = layers.Dense(100, activation="relu")(x) # try activation="softplus"
    elif param['arc_type'] == '2':
        x = layers.Reshape((np.prod(concat_layer_shape[1:4])*(concat_layer_shape[-1]),))(x)
        model = tf.keras.models.Model(inputs=model_inputs, outputs=x)
        model_cat = tf.keras.Sequential()
        model_cat.add(tf.keras.Input(shape=cat_shape))
        model_cat.add(layers.Lambda(condition))
        if param['cat_input_type'] == 'None':
            y = x
        else:
            y = layers.concatenate([model.output, model_cat.output])
        x = layers.Dense(640, activation="relu")(y) # try activation="softplus"
        x = layers.Dense(100, activation="relu")(x) # try activation="softplus"
    elif param['arc_type'] == '3':
        x = layers.Reshape((np.prod(concat_layer_shape[1:4])*(concat_layer_shape[-1]),))(x)
        x = layers.Dense(640, activation="relu")(x) # try activation="softplus"
        model = tf.keras.models.Model(inputs=model_inputs, outputs=x)

        model_cat = tf.keras.Sequential()
        model_cat.add(tf.keras.Input(shape=cat_shape))
        model_cat.add(layers.Lambda(condition))
        if param['cat_input_type'] == 'None':
            y = x
        else:
            y = layers.concatenate([model.output, model_cat.output])
        x = layers.Dense(100, activation="relu")(y) # try activation="softplus"
    elif param['arc_type'] == '4':
        x = layers.Reshape((np.prod(concat_layer_shape[1:4])*(concat_layer_shape[-1]),))(x)
        x = layers.Dense(640, activation="relu")(x) # try activation="softplus"
        x = layers.Dense(100, activation="relu")(x) # try activation="softplus"
        model = tf.keras.models.Model(inputs=model_inputs, outputs=x)

        model_cat = tf.keras.Sequential()
        model_cat.add(tf.keras.Input(shape=cat_shape))
        model_cat.add(layers.Lambda(condition))
        if param['cat_input_type'] == 'None':
            x = x
        else:
            x = tf.keras.layers.concatenate([model.output, model_cat.output])
    else:
        raise Exception('Architecture type selected is undefined')
    
    if param['num_input_maps'] == 1: 
        final_inputs = [map_model.input, model_cat.input]
    elif param['num_input_maps'] == 2:  
        final_inputs = [map_model_1.input, map_model_2.input, model_cat.input]
    elif param['num_input_maps'] == 3: 
        final_inputs = [map_model_1.input, map_model_2.input, map_model_3.input, model_cat.input]

    x = layers.Dense(1, activation="linear")(x) 
    final_model = tf.keras.models.Model(inputs=final_inputs, outputs=x)
    return final_model
