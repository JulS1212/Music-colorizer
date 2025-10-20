import librosa
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from sklearn.cluster import KMeans


def music_to_colors(audio_path, num_colors=5):
    # Загружаем аудио
    y, sr = librosa.load(audio_path, duration=30)  # первые 30 секунд для скорости

    # Извлекаем музыкальные характеристики
    features = extract_music_features(y, sr)

    # Создаем базовые цвета на основе характеристик
    base_colors = generate_base_colors(features, num_colors)

    return base_colors


def extract_music_features(y, sr):
    """Извлекаем ключевые музыкальные параметры"""
    features = {}

    # 1. ТЕМП (битрейт)
    features['tempo'] = librosa.beat.tempo(y=y, sr=sr)[0]  # ударов в минуту

    # 2. ЭНЕРГИЯ (громкость)
    features['energy'] = np.mean(librosa.feature.rms(y=y))

    # 3. ЯРКОСТЬ ЗВУКА (спектральный центроид)
    features['brightness'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # 4. ГАРМОНИЧНОСТЬ
    features['harmony'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # 5. РАЗНООБРАЗИЕ СИГНАЛА
    features['complexity'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Нормализуем значения
    features['tempo_norm'] = np.clip(features['tempo'] / 200, 0.1, 0.9)
    features['energy_norm'] = np.clip(features['energy'] * 10, 0.2, 0.9)
    features['brightness_norm'] = np.clip(features['brightness'] / 5000, 0.1, 0.8)
    features['harmony_norm'] = np.clip(features['harmony'] / 8000, 0.1, 0.7)
    features['complexity_norm'] = np.clip(features['complexity'] / 6000, 0.1, 0.6)

    return features


def generate_base_colors(features, num_colors):
    """Генерируем цвета на основе музыкальных параметров"""
    colors = []

    # Базовый цвет на основе темпа и энергии
    base_hue = features['tempo_norm']  # темп → оттенок
    base_saturation = features['energy_norm']  # энергия → насыщенность
    base_value = features['brightness_norm']  # яркость → значение

    # Создаем основной цвет
    main_color = colorsys.hsv_to_rgb(base_hue, base_saturation, base_value)
    colors.append(main_color)

    # Генерируем дополнительные цвета на основе других параметров
    for i in range(1, num_colors):
        # Создаем вариации на основе гармонии и сложности
        hue_variation = (base_hue + features['harmony_norm'] * 0.3 * i) % 1.0
        sat_variation = np.clip(base_saturation + features['complexity_norm'] * 0.2 * (-1) ** i, 0.2, 0.9)
        val_variation = np.clip(base_value + 0.1 * i * (-1) ** i, 0.3, 0.95)

        variant_color = colorsys.hsv_to_rgb(hue_variation, sat_variation, val_variation)
        colors.append(variant_color)

    # Конвертируем в RGB 0-255
    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    return colors_rgb


def show_music_palette(colors, title="Музыкальная палитра"):
    """Визуализируем цветовую палитру"""
    plt.figure(figsize=(10, 2))
    for i, color in enumerate(colors):
        plt.fill_between([i, i + 1], 0, 1, color=np.array(color) / 255)
        hex_color = f'#{color[0]:02X}{color[1]:02X}{color[2]:02X}'
        text_color = 'white' if sum(color) < 380 else 'black'
        plt.text(i + 0.5, 0.5, hex_color,
                 ha='center', va='center', fontsize=10, color=text_color, weight='bold')
    plt.xlim(0, len(colors))
    plt.axis('off')
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


# ТЕСТИРУЕМ!
audio_path = "C:/Users/Yulia-PC/PythonProject/data/Waltz_No_2.mp3"
colors = music_to_colors(audio_path)

print("🎵 МУЗЫКАЛЬНЫЕ ХАРАКТЕРИСТИКИ:")
print(f"Темп: {librosa.beat.tempo(y=librosa.load(audio_path)[0], sr=librosa.load(audio_path)[1])[0]:.1f} BPM")
print(f"Энергия: {np.mean(librosa.feature.rms(y=librosa.load(audio_path)[0])):.3f}")
print(
    f"Яркость: {np.mean(librosa.feature.spectral_centroid(y=librosa.load(audio_path)[0], sr=librosa.load(audio_path)[1])):.1f}")

print("\n🎨 УНИКАЛЬНЫЕ ЦВЕТА МУЗЫКИ:")
for i, color in enumerate(colors):
    print(f"Цвет {i + 1}: RGB{color} → #{color[0]:02X}{color[1]:02X}{color[2]:02X}")

show_music_palette(colors)