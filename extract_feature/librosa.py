import os
import re
import sys
import librosa
from random import shuffle
import numpy as np
from typing import Tuple, Union
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import utils








# Data augmentation
def noise(data):
    """
    Thêm nhiễu Gaussian vào tín hiệu âm thanh.
    """
    if data is None or len(data) == 0:
        return data
    max_val = np.amax(data)
    # Nếu max_val=0 thì bỏ qua thêm nhiễu (tránh chia 0 hoặc chuẩn hóa sai)
    noise_amp = 0 if max_val == 0 else 0.035 * np.random.uniform() * max_val
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.8):
    """
    Thay đổi tốc độ phát (time-stretch) của tín hiệu âm thanh.
    """
    if data is None or len(data) == 0:
        return data
    # Đảm bảo rate nằm trong khoảng hợp lý
    if rate < 0.5 or rate > 2.0:
        rate = 0.8
    try:
        return librosa.effects.time_stretch(data, rate)
    except Exception as e:
        print(f"Error during stretch: {e}")
        return data

def shift(data):
    """
    Dịch chuyển (time-shift) tín hiệu âm thanh.
    """
    if data is None or len(data) == 0:
        return data
    # Ví dụ: dịch chuyển 20% độ dài tín hiệu
    max_shift = int(0.2 * len(data))
    shift_range = np.random.randint(-max_shift, max_shift)
    return np.roll(data, shift_range)

def pitch(data, sr, n_steps=0.7):
    """
    Thay đổi cao độ (pitch) của tín hiệu âm thanh.
    """
    if data is None or len(data) == 0:
        return data
    # Giới hạn n_steps trong [-12, 12]
    if abs(n_steps) > 12:
        n_steps = np.sign(n_steps) * 12
    try:
        return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
    except Exception as e:
        print(f"Error during pitch: {e}")
        return data


# feature extraction
def zcr(data, frame_length=2048, hop_length=512, fixed_size=100):
    if data is None or len(data) == 0:
        return np.zeros(fixed_size)
    zcr_values = librosa.feature.zero_crossing_rate(
        y=data, frame_length=frame_length, hop_length=hop_length
    ).squeeze()

    # Pad hoặc cắt sao cho đúng fixed_size
    zcr_values = np.pad(zcr_values, (0, max(0, fixed_size - len(zcr_values))), 
                        mode='constant')[:fixed_size]
    return zcr_values

def rmse(data, frame_length=2048, hop_length=512, fixed_size=100):
    if data is None or len(data) == 0:
        return np.zeros(fixed_size)
    rms_values = librosa.feature.rms(
        y=data, frame_length=frame_length, hop_length=hop_length
    ).squeeze()

    rms_values = np.pad(rms_values, (0, max(0, fixed_size - len(rms_values))), 
                        mode='constant')[:fixed_size]
    return rms_values

def mfcc(data, sr=22050, frame_length=2048, hop_length=512, n_mfcc=13, fixed_size=100):
    if data is None or len(data) == 0:
        # Nếu rỗng, trả về vector 0 với độ dài n_mfcc * fixed_size
        return np.zeros(n_mfcc * fixed_size)
    mfcc_values = librosa.feature.mfcc(
        y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
    )
     
   
    mfcc_padded = []
    for coeff in mfcc_values:
        coeff = np.pad(coeff, (0, max(0, fixed_size - len(coeff))), 
                       mode='constant')[:fixed_size]
        mfcc_padded.append(coeff)

    # Gộp tất cả lại thành 1D
    mfcc_flatten = np.array(mfcc_padded).flatten()
    return mfcc_flatten



def extract_features(data, sr=22050, frame_length=2048, hop_length=512, 
                    n_mfcc=13, duration=2.5, fixed_size=100):

    # Chuẩn hóa độ dài âm thanh 
    max_length = int(sr * duration)
    data = np.pad(data, (0, max(0, max_length - len(data))), 
                  mode='constant')[:max_length]

    # Tính ZCR, RMSE, MFCC
    zcr_values = zcr(data, frame_length, hop_length, fixed_size)
    rmse_values = rmse(data, frame_length, hop_length, fixed_size)
    mfcc_values = mfcc(data, sr, frame_length, hop_length, n_mfcc, fixed_size)

    # Gộp tất cả thành 1 vector
    features = np.hstack((zcr_values, rmse_values, mfcc_values))

    # Kiểm tra phương sai để tránh chuẩn hóa về 1 giá trị cố định
    if np.std(features) == 0:
        return features  
    else:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(-1, 1)).flatten()
        return features_scaled


def get_features(path, sr=22050, duration=2.5, offset=0.6):
    """
    - Load file âm thanh
    - Trích xuất đặc trưng gốc
    - Áp dụng augmentation (noise, pitch, noise+pitch)

    """
    try:
        # Kiểm tra file có tồn tại không
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            return None

        # Load file
        data, sample_rate = librosa.load(path, sr=sr, duration=duration, offset=offset)
        if data is None or len(data) == 0:
            print(f"[Warning] Empty audio: {path}")
            return None

        # Đặc trưng gốc
        base_features = extract_features(data, sample_rate)

        # Augmentations
        noised_data = noise(data)
        noised_features = extract_features(noised_data, sample_rate)

        pitched_data = pitch(data, sample_rate, n_steps=0.7)
        pitched_features = extract_features(pitched_data, sample_rate)

        # Kết hợp noise + pitch
        pitched_noised_data = noise(pitched_data)
        pitched_noised_features = extract_features(pitched_noised_data, sample_rate)

        # Gộp tất cả thành 1 vector
        
        all_features = np.vstack((
            base_features,
            noised_features,
            pitched_features,
            pitched_noised_features
        ))

        
        return all_features

    except Exception as e:
        print(f"[Error] Processing file {path}: {e}")
        return None



