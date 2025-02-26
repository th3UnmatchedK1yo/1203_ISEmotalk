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
class CNN(BaseModel, ABC):
    def __init__(
            self,
            model: Sequential,
            trained: bool = False
    ) -> None:
        super(CNN, self).__init__(model, trained)
        print(self.model.summary())

    @classmethod
    def make(
            cls,
            input_shape: tuple,
            n_kernels: int,
            kernel_size: int,
            hidden_size: int,
            dropout: float,
            n_classes: int,
            lr: float
    ):
        model = Sequential()

        # Conv1D Layer 1
        model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5, strides=2))

        # Conv1D Layer 2
        model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5, strides=2))
        model.add(Dropout(0.2))

        # Conv1D Layer 3
        model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5, strides=2))

        # Conv1D Layer 4
        model.add(Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5, strides=2))
        model.add(Dropout(0.2))

        # Conv1D Layer 5
        model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5, strides=2))
        model.add(Dropout(0.2))

        # Flatten Layer
        model.add(Flatten())

        # Dense Layer 1
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())

        # Dense Output Layer
        model.add(Dense(n_classes, activation='softmax'))

        return cls(model)

    def save(self, path: str, name: str) -> None:
        """
        Lưu mô hình dưới dạng JSON + HDF5 (.h5)
        """
        h5_path = os.path.join(path, name + ".h5")
        self.model.save_weights(h5_path)

        json_path = os.path.join(path, name + ".json")
        with open(json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    @classmethod
    def load(cls, path: str, name: str):
        """
        Tải mô hình đã lưu từ JSON + HDF5
        """
        model_json_path = os.path.abspath(os.path.join(path, name + ".json"))
        with open(model_json_path, "r") as json_file:
            loaded_model_json = json_file.read()
        
        model = model_from_json(loaded_model_json)

        model_path = os.path.abspath(os.path.join(path, name + ".h5"))
        model.load_weights(model_path)

        return cls(model, trained=True)

    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            verbose=1,
            batch_size: int = 32,
            n_epochs: int = 50,
    ) -> None:
        
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)

        model_checkpoint = ModelCheckpoint(
            filepath="best_model.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        lr_reduction = ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            verbose=1,
            factor=0.5,
            min_lr=1e-5
        )

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_val, y_val),
            verbose=1,
            shuffle=True,
            callbacks=[model_checkpoint, early_stop, lr_reduction]
        )

        acc = history.history['accuracy']
        loss = history.history['loss']

        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']

        curve(acc, val_acc, "Accuracy", "acc")
        curve(loss, val_loss, "Loss", "loss")

        self.trained = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("There is no trained model.")

        samples = self.reshape_input(x)
        return np.argmax(self.model.predict(samples), axis=1)


    @abstractmethod
    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        return data
