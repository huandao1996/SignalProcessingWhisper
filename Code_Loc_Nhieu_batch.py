import os
import librosa
import librosa.display
import noisereduce as nr
import soundfile as sf


def remove_noise(input_file, output_file):
    """
    Lọc nhiễu một file âm thanh và lưu kết quả.

    Args:
        input_file (str): Đường dẫn file âm thanh đầu vào.
        output_file (str): Đường dẫn file âm thanh đầu ra.
    """
    print(f"🔄 Đang xử lý: {input_file} ...")

    # Đọc file âm thanh
    audio, sr = librosa.load(input_file, sr=None)

    # Lọc nhiễu
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)

    # Lưu file đã xử lý
    sf.write(output_file, reduced_noise, sr)
    print(f"✅ Đã lưu file sau khi lọc nhiễu: {output_file}")


def batch_noise_reduction(input_folder, output_folder):
    """
    Lọc nhiễu tất cả các file .wav trong thư mục đầu vào và lưu vào thư mục đầu ra.

    Args:
        input_folder (str): Thư mục chứa file âm thanh gốc.
        output_folder (str): Thư mục để lưu file âm thanh đã lọc nhiễu.
    """
    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lặp qua tất cả các file .wav trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  # Chỉ xử lý file WAV
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename}")

            # Gọi hàm lọc nhiễu cho từng file
            remove_noise(input_path, output_path)


# 🏁 Chạy chương trình lọc nhiễu hàng loạt
input_folder = "noisy_audio"  # Thư mục chứa file âm thanh gốc
output_folder = "cleaned_audio"  # Thư mục để lưu file đã lọc nhiễu

batch_noise_reduction(input_folder, output_folder)
print("🎉 Hoàn tất xử lý tất cả các file!")
