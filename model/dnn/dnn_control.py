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

class DNN(BaseModel, ABC):

    def __init__(self, model: Sequential, trained: bool = False) -> None:
        """
        Khởi tạo mô hình DNN.
        
        Tham số:
            model (Sequential): Mô hình Keras Sequential.
            trained (bool): Cờ chỉ định nếu mô hình đã được huấn luyện trước.
        """
        super(DNN, self).__init__(model, trained)
        print(self.model.summary())

    @classmethod
    def load(cls, path: str, name: str):
        """
        Tải mô hình từ tệp JSON và HDF5.
        
        Tham số:
            path (str): Đường dẫn đến thư mục chứa tệp mô hình.
            name (str): Tên cơ sở của tệp mô hình.
        
        Trả về:
            DNN: Một instance của lớp DNN với mô hình đã tải.
        """
        model_json_path = os.path.abspath(os.path.join(path, name + ".json"))
        with open(model_json_path, "r") as json_file:
            loaded_model_json = json_file.read()
        
        model = model_from_json(loaded_model_json)
        model_path = os.path.abspath(os.path.join(path, name + ".h5"))
        model.load_weights(model_path)
        
        return cls(model, trained=True)

    def save(self, path: str, name: str) -> None:
        """
        Lưu trọng số và kiến trúc mô hình vào các tệp.
        
        Tham số:
            path (str): Thư mục để lưu các tệp mô hình.
            name (str): Tên cơ sở cho các tệp mô hình.
        """
        h5_path = os.path.join(path, name + ".h5")
        self.model.save_weights(h5_path)
        
        json_path = os.path.join(path, name + ".json")
        with open(json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    def train(
            self,
            x_train,
            y_train,
            x_val=None,
            y_val=None,
            batch_size=32, 
            n_epochs=50
        ) -> None:
        """
        Huấn luyện mô hình DNN.
        
        Tham số:
            x_train (np.ndarray): Dữ liệu huấn luyện.
            y_train (np.ndarray): Nhãn huấn luyện.
            x_val (np.ndarray, tùy chọn): Dữ liệu xác thực. Mặc định là None.
            y_val (np.ndarray, tùy chọn): Nhãn xác thực. Mặc định là None.
            batch_size (int, tùy chọn): Số mẫu mỗi lần cập nhật gradient. Mặc định là 32.
            n_epochs (int, tùy chọn): Số epoch để huấn luyện mô hình. Mặc định là 50.
        """
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)
        
        # Thiết lập callbacks
        model_checkpoint = ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1)
        lr_reduction = ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1, factor=0.5, min_lr=1e-5)

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_val, y_val),
            verbose=1,
            shuffle=True,
            callbacks=[model_checkpoint, early_stop, lr_reduction]
        )

        # Lưu lịch sử huấn luyện
        acc = history.history['accuracy']
        loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']

        # Vẽ đồ thị
        curve(acc, val_acc, "Accuracy", "acc")
        curve(loss, val_loss, "Loss", "loss")
        self.trained = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn lớp cho dữ liệu đầu vào.
        
        Tham số:
            x (np.ndarray): Dữ liệu đầu vào.
        
        Trả về:
            np.ndarray: Nhãn lớp dự đoán.
        """
        if not self.trained:
            raise RuntimeError("Chưa có mô hình nào được huấn luyện.")
        
        samples = self.reshape_input(x)
        return np.argmax(self.model.predict(samples), axis=1)

    @abstractmethod
    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """
        Định hình lại dữ liệu đầu vào theo định dạng yêu cầu.
        
        Tham số:
            data (np.ndarray): Dữ liệu đầu vào.
        
        Trả về:
            np.ndarray: Dữ liệu đầu vào đã được định hình lại.
        """
        pass