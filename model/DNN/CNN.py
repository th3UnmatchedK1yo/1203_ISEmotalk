import os
import json
from typing import Optional
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from base_model.base import BaseModel
from abc import ABC, abstractmethod
from utils import curve
from .dnn_control import DNN


class CNN(DNN, ABC):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        """
        Khởi tạo mô hình CNN.
        
        Tham số:
            model (Sequential): Mô hình Keras Sequential.
            trained (bool): Cờ chỉ định nếu mô hình đã được huấn luyện trước.
        """
        super(CNN, self).__init__(model, trained)
        print(self.model.summary())


    @classmethod
    def make(cls, config_path="config.json"):
        """
        Tạo một mô hình CNN từ tệp cấu hình.
        
        Tham số:
            config_path (str): Đường dẫn đến tệp JSON cấu hình.
        
        Trả về:
            CNN: Một instance của lớp CNN.
        """
        config = cls.load_config(config_path)
        input_shape = tuple(config["input_shape"])
        num_classes = config["num_classes"]
        
        model = Sequential()
        
       
        for layer in config["conv_layers"]:
            model.add(Conv1D(
                            layer["filters"],
                            kernel_size=layer["kernel_size"], 
                            strides=layer["strides_1"], 
                            padding=layer["padding"], 
                            activation=layer["activation"], 
                            input_shape=input_shape)
                            )
            model.add(BatchNormalization())
            model.add(MaxPooling1D(
                                   pool_size=layer["pool_size"], 
                                   strides=layer["strides_2"]
                                   )
                      )
            if layer["dropout"] > 0:
                model.add(Dropout(layer["dropout"]))
        
        
        model.add(Flatten())
        
       
        for layer in config["dense_layers"]:
            model.add(Dense(layer["units"], activation=layer["activation"]))
            model.add(BatchNormalization())
            if layer["dropout"] > 0:
                model.add(Dropout(layer["dropout"]))
        
        
        model.add(Dense(num_classes, activation=config["output_layer"]["activation"]))
        
        return cls(model)
    


    @abstractmethod
    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """
        Định hình lại dữ liệu đầu vào theo định dạng yêu cầu.
        
        Tham số:
            data (np.ndarray): Dữ liệu đầu vào.
        
        Trả về:
            np.ndarray: Dữ liệu đầu vào đã được định hình lại.
        """
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        return data
