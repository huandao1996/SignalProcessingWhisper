import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


# Bước 1: Load tệp âm thanh

# Load file âm thanh
file_path = "test.wav"
y, sr = librosa.load(file_path, sr=None)  # Đọc tín hiệu và tần số lấy mẫu

# Vẽ sóng âm gốc
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Tín hiệu âm thanh gốc")
plt.show()

#Bước 2: Chuyển sang miền tần số bằng STFT

import scipy.signal

# Thực hiện STFT
n_fft = 1024  # Số điểm FFT
hop_length = 512  # Bước nhảy giữa các cửa sổ
win_length = 1024  # Độ dài cửa sổ

stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
magnitude, phase = np.abs(stft), np.angle(stft)  # Lấy biên độ và pha

# Vẽ phổ của tín hiệu gốc
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title("Phổ STFT của tín hiệu gốc")
plt.colorbar(format="%+2.0f dB")
plt.show()

#Bước 3: Ước lượng phổ nhiễu

# Lấy một đoạn tín hiệu (ví dụ: 0.5 giây đầu tiên)
num_noise_frames = 20  # Số khung dùng để ước lượng nhiễu
noise_stft = magnitude[:, :num_noise_frames]
noise_mean = np.mean(noise_stft, axis=1, keepdims=True)  # Trung bình phổ nhiễu

# Vẽ phổ của nhiễu ước lượng
plt.figure(figsize=(10, 4))
plt.plot(noise_mean, label="Ước lượng phổ nhiễu")
plt.xlabel("Tần số (bins)")
plt.ylabel("Biên độ")
plt.title("Ước lượng phổ nhiễu")
plt.legend()
plt.show()

#Bước 4: Tạo mặt nạ giảm nhiễu (Noise Gate)

alpha = 1.0  # Hệ số giảm nhiễu (1.0 đến 2.0)
beta = 0.002  # Ngưỡng tối thiểu

# Tính hệ số giảm nhiễu G(f, t)
G = np.maximum((magnitude - alpha * noise_mean) / magnitude, beta)

# Áp dụng G vào phổ
magnitude_denoised = G * magnitude

# Vẽ mặt nạ giảm nhiễu
plt.figure(figsize=(10, 4))
librosa.display.specshow(G, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title("Mặt nạ Spectral Gating")
plt.colorbar()
plt.show()

# Bước 5: Áp dụng mặt nạ và chuyển về miền thời gian
# Kết hợp lại pha để tạo STFT mới
stft_denoised = magnitude_denoised * np.exp(1j * phase)

# Biến đổi về miền thời gian
y_denoised = librosa.istft(stft_denoised, hop_length=hop_length, win_length=win_length)

# Vẽ tín hiệu sau khi giảm nhiễu
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y_denoised, sr=sr)
plt.title("Tín hiệu sau khi giảm nhiễu")
plt.show()


#Bước 6: Lưu tệp âm thanh sau khi giảm nhiễu
import soundfile as sf

output_file = "clean_audio.wav"
sf.write(output_file, y_denoised, sr)
print(f"Đã lưu file âm thanh giảm nhiễu: {output_file}")


