import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Загрузка аудиофайла
audio_path = "C:/Users/Yulia-PC/Desktop/Waltz_No_2.mp3"  # не забудь заменить путь!
y, sr = librosa.load(audio_path)

# Создание спектрограммы
plt.figure(figsize=(12, 4))
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

# Визуализация
librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-spectrogram')
plt.tight_layout()
plt.show()