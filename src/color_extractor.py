import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2
from PIL import Image


def extract_colors_from_spectrogram(audio_path, num_colors=5):
    # 1. Загружаем музыку и создаем спектрограмму
    y, sr = librosa.load(audio_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # 2. Преобразуем спектрограмму в изображение
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')  # убираем оси для чистого изображения

    # 3. Сохраняем спектрограмму как временное изображение
    temp_path = "temp_spectrogram.png"
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    # 4. Загружаем изображение и извлекаем цвета
    image = cv2.imread(temp_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 5. Преобразуем изображение в список пикселей
    pixels = image.reshape(-1, 3)

    # 6. Используем K-Means для нахождения главных цветов
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)

    # 7. Получаем цвета кластеров
    colors = kmeans.cluster_centers_.astype(int)

    return colors


# ТЕСТИРУЕМ НА ТВОЁМ ВАЛЬСЕ!
audio_path = "C:/Users/Yulia-PC/PythonProject/data/Waltz_No_2.mp3"
colors = extract_colors_from_spectrogram(audio_path)

print("ЦВЕТОВАЯ ПАЛИТРА ВАЛЬСА ШОСТАКОВИЧА:")
for i, color in enumerate(colors):
    print(f"Цвет {i + 1}: RGB{tuple(color)}")

def show_color_palette(colors, title="Цветовая палитра музыки"):
    plt.figure(figsize=(10, 2))
    for i, color in enumerate(colors):
        plt.fill_between([i, i+1], 0, 1, color=color/255)
        plt.text(i + 0.5, 0.5, f'#{color[0]:02X}{color[1]:02X}{color[2]:02X}',
                ha='center', va='center', fontsize=12, color='white' if np.mean(color) < 128 else 'black')
    plt.xlim(0, len(colors))
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Показываем палитру Шостаковича
show_color_palette(colors, "Цвета вальса Шостаковича №2")