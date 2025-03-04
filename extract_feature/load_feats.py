import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

from .extract_librosa import extract_features
from .augmentation import noise, pitch, shift, stretch


FEATURE_DIR = Path("features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)  

def get_path (path: str ):
    """Tải dữ liệu RAVDESS và tạo DataFrame.

    Args:
        dataset_path (str): Đường dẫn tới thư mục chứa dataset.

    Returns:
        pd.DataFrame: DataFrame chứa đường dẫn file và nhãn cảm xúc.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"[Error] Dataset folder '{dataset_path}' không tồn tại!")

    file_emotion = []
    file_path = []

    for actor_path in dataset_path.iterdir():
        if not actor_path.is_dir():
            continue  
        
        for file in actor_path.iterdir():
            if file.suffix != ".wav": 
                continue
            try:
                part = file.stem.split("-")  
                file_emotion.append(int(part[2]))  
                file_path.append(str(file))
            except (IndexError, ValueError):
                print(f"[Warning] Không thể đọc thông tin từ file: {file}")

    if not file_path:
        raise ValueError("[Error] Không tìm thấy file âm thanh nào trong dataset!")


    emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])
    path_df = pd.DataFrame(file_path, columns=["Path"])
    result_df = pd.concat([emotion_df, path_df], axis=1)

 
    result_df.Emotions.replace({
        1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
        5: "Angry", 6: "Fear", 7: "Disgust", 8: "Surprise"
    }, inplace=True)

    return result_df

def save_features(features, filename):
    """Lưu đặc trưng vào file .npy để sử dụng lại sau này."""
    filepath = FEATURE_DIR / f"{filename}.npy"
    np.save(filepath, features)
    print(f"[INFO] Đã lưu đặc trưng tại: {filepath}")

def load_features(filename):
    """Tải đặc trưng đã lưu nếu tồn tại."""
    filepath = FEATURE_DIR / f"{filename}.npy"
    if filepath.exists():
        print(f"[INFO] Đang tải đặc trưng từ: {filepath}")
        return np.load(filepath)
    return None

def get_features(path, sr=22050, duration=2.5, offset=0.6, save=True):
    """Tải file âm thanh, trích xuất đặc trưng và lưu lại nếu cần.

    Args:
        path (str): Đường dẫn file âm thanh.
        sr (int): Tần số lấy mẫu.
        duration (float): Độ dài âm thanh trích xuất.
        offset (float): Offset khi load file.
        save (bool): Nếu True, lưu đặc trưng vào thư mục `features/`.

    Returns:
        np.ndarray: Đặc trưng âm thanh đã trích xuất.
    """
    filename = Path(path).stem 


    existing_features = load_features(filename)
    if existing_features is not None:
        return existing_features

    try:
        if not os.path.exists(path):
            print(f"[Warning] File không tồn tại: {path}")
            return None

        data, sample_rate = librosa.load(path, sr=sr, duration=duration, offset=offset)
        if data is None or len(data) == 0:
            print(f"[Warning] File rỗng: {path}")
            return None

        base_features = extract_features(data, sample_rate)

      
        noised_features = extract_features(noise(data), sample_rate)
        stretched_features = extract_features(stretch(data), sample_rate)
        shifted_features = extract_features(shift(data), sample_rate)
        pitched_features = extract_features(pitch(data, sample_rate, n_steps=0.7), sample_rate)


        pitched_noised_features = extract_features(noise(pitch(data, sample_rate, n_steps=0.7)), sample_rate)

        all_features = np.vstack((
            base_features,
            noised_features,
            stretched_features,
            shifted_features,
            pitched_features,
            pitched_noised_features
        ))

        if save:
            save_features(all_features, filename)

        return all_features

    except Exception as e:
        print(f"[Error] Lỗi khi xử lý file {path}: {e}")
        return None
    


