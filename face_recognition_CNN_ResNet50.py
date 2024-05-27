
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255.)

def load_train(path):
    """
    Carga de entrenamiento del conjunto de datos desde la ruta
    """
    
    train_gen_flow = datagen.flow_from_dataframe(
        
        # dataframe= pd.read_csv('/datasets/faces/labels.csv'),
        # directory= path, 
        dataframe=pd.read_csv(path + 'labels.csv'),
        directory=path + 'final_files/',
       
        x_col='file_name',
        y_col='real_age',
        target_size=(150,150), # La imagen mas pequeña es de 47,47
        batch_size=16,
        class_mode='raw',
        seed=12345,
        subset='training'
    )

    return train_gen_flow

def load_test(path):
    
    """
    Carga de validación/prueba del conjunto de datos desde la ruta
    """
    test_gen_flow = datagen.flow_from_dataframe(

        dataframe=pd.read_csv(path + 'labels.csv'),
        directory=path + 'final_files/',
        
        # dataframe= pd.read_csv('/datasets/faces/labels.csv'),
        # directory= path, 
        x_col='file_name',
        y_col='real_age',
        target_size=(150,150), # La imagen mas pequeña es de 47,47
        batch_size=16,
        class_mode='raw',
        seed=12345,
        subset='validation'
    )
    return test_gen_flow


def create_model(input_shape):
    
    """Definiendo el modelo"""
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape)
    
    base_model.trainable=False
    
    model = Sequential(
        [
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='relu'), ### Preguntar esta parte
        ]
    )
    model.compile(optimizer=Adam(learning_rate= 0.0001),loss='mean_squared_error', metrics=['mae'])

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    """
    Entrena el modelo dados los parámetros
    """
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
             )
    
    return model