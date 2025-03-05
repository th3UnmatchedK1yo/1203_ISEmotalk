import os
import sys
from pydub import AudioSegment

def convert_mp3_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Chuyển đổi file MP3 sang WAV.
    
    Args:
        input_path (str): Đường dẫn file MP3 đầu vào.
        output_path (str, optional): Đường dẫn file WAV đầu ra. Nếu không cung cấp, lưu vào cùng thư mục với input.
    
    Returns:
        str: Đường dẫn file WAV đã được lưu.
    """
    # Kiểm tra sự tồn tại của file đầu vào
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} không tồn tại!")

    # Kiểm tra định dạng file
    if not input_path.lower().endswith(".mp3"):
        raise ValueError("Định dạng file không phải MP3!")

    # Đọc file MP3
    try:
        audio = AudioSegment.from_mp3(input_path)
    except Exception as e:
        raise RuntimeError(f"Không thể đọc file MP3: {e}")

    # Xác định đường dẫn đầu ra
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".wav"

    # Chuyển đổi và lưu file WAV
    try:
        audio.export(output_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"Không thể xuất file WAV: {e}")

    print(f"Đã chuyển đổi: {input_path} → {output_path}")
    return output_path


# Chạy script từ dòng lệnh
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Sử dụng: python convert_mp3_to_wav.py <input_mp3> [output_wav]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        convert_mp3_to_wav(input_file, output_file)
    except Exception as e:
        print(f"Lỗi: {e}")
