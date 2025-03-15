import wave
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wav
import numpy as np
import os

def curve(train: list, val: list, title: str, y_label: str, save_path: str = "utils/plots/") -> None:
    """
   Vẽ đồ thị loss và accuracy và lưu vào file thay vì hiển thị.

    Args:
        train (list): Danh sách loss/accuracy của tập train.
        val (list): Danh sách loss/accuracy của tập validation.
        title (str): Tiêu đề đồ thị.
        y_label (str): Nhãn trục Y.
        save_path (str, optional): Thư mục lưu ảnh, mặc định là "plots/".
    """
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    
    os.makedirs(save_path, exist_ok=True)
    
    file_path = os.path.join(save_path, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(file_path)  
    print(f"Saved plot at {file_path}")
    
    plt.close()  

def radar(data_prob: np.ndarray, class_labels: list, save_path: str = "utils/plots/") -> None:
    """
    Vẽ biểu đồ radar xác suất cảm xúc và lưu vào file.

    Args:
        data_prob (np.ndarray): Mảng xác suất của từng class.
        class_labels (list): Danh sách các nhãn cảm xúc.
        save_path (str, optional): Thư mục lưu ảnh, mặc định là "plots/".
    """
    angles = np.linspace(0, 2 * np.pi, len(class_labels), endpoint=False)
    
    data = np.concatenate((data_prob, [data_prob[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    class_labels = class_labels + [class_labels[0]]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, data, "bo-", linewidth=2)
    ax.fill(angles, data, facecolor="r", alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, class_labels)
    ax.set_title("Emotion Recognition", va="bottom")

    ax.set_rlim(0, 1)
    ax.grid(True)


    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, "radar_chart.png")
    plt.savefig(file_path)  
    print(f"Saved radar chart at {file_path}")
    
    plt.close() 
