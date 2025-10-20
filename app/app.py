import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from io import BytesIO
import tempfile
import os

st.set_page_config(page_title="Music Color Analyzer", page_icon="🎵", layout="wide")

# Заголовок приложения
st.title("🎵 Music Color Analyzer")
st.markdown("Загрузите музыку и узнайте её цветовую палитру!")


# Функции из нашего проекта (копируем их сюда)
def extract_music_features(y, sr):
    features = {}
    features['tempo'] = librosa.beat.tempo(y=y, sr=sr)[0]
    features['energy'] = np.mean(librosa.feature.rms(y=y))
    features['brightness'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['harmony'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['complexity'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Нормализация
    features['tempo_norm'] = np.clip(features['tempo'] / 200, 0.1, 0.9)
    features['energy_norm'] = np.clip(features['energy'] * 10, 0.2, 0.9)
    features['brightness_norm'] = np.clip(features['brightness'] / 5000, 0.1, 0.8)
    features['harmony_norm'] = np.clip(features['harmony'] / 8000, 0.1, 0.7)
    features['complexity_norm'] = np.clip(features['complexity'] / 6000, 0.1, 0.6)

    return features


def music_to_colors(audio_data, sr, num_colors=5):
    features = extract_music_features(audio_data, sr)
    colors = []

    base_hue = features['tempo_norm']
    base_saturation = features['energy_norm']
    base_value = features['brightness_norm']

    # Основной цвет
    main_color = colorsys.hsv_to_rgb(base_hue, base_saturation, base_value)
    colors.append(main_color)

    # Дополнительные цвета
    for i in range(1, num_colors):
        hue_variation = (base_hue + features['harmony_norm'] * 0.3 * i) % 1.0
        sat_variation = np.clip(base_saturation + features['complexity_norm'] * 0.2 * (-1) ** i, 0.2, 0.9)
        val_variation = np.clip(base_value + 0.1 * i * (-1) ** i, 0.3, 0.95)

        variant_color = colorsys.hsv_to_rgb(hue_variation, sat_variation, val_variation)
        colors.append(variant_color)

    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors_rgb, features


# Загрузка файла
uploaded_file = st.file_uploader("Выберите аудиофайл (MP3, WAV)", type=['mp3', 'wav'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    with st.spinner('Анализируем музыку...'):
        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Анализируем музыку
            y, sr = librosa.load(tmp_path, duration=30)
            colors, features = music_to_colors(y, sr)

            # Показываем результаты
            st.success("Анализ завершён!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎨 Цветовая палитра")
                # Создаём визуализацию палитры
                fig, ax = plt.subplots(figsize=(10, 2))
                for i, color in enumerate(colors):
                    ax.fill_between([i, i + 1], 0, 1, color=np.array(color) / 255)
                    hex_color = f'#{color[0]:02X}{color[1]:02X}{color[2]:02X}'
                    text_color = 'white' if sum(color) < 380 else 'black'
                    ax.text(i + 0.5, 0.5, hex_color,
                            ha='center', va='center', fontsize=10,
                            color=text_color, weight='bold')
                ax.set_xlim(0, len(colors))
                ax.axis('off')
                st.pyplot(fig)

            with col2:
                st.subheader("📊 Музыкальные характеристики")
                st.metric("Темп", f"{features['tempo']:.1f} BPM")
                st.metric("Энергия", f"{features['energy']:.3f}")
                st.metric("Яркость", f"{features['brightness']:.1f} Hz")
                st.metric("Сложность", f"{features['complexity']:.1f} Hz")

            # Показываем цвета в виде списка
            st.subheader("🎨 Коды цветов")
            for i, color in enumerate(colors):
                hex_code = f'#{color[0]:02X}{color[1]:02X}{color[2]:02X}'
                st.code(f"Цвет {i + 1}: RGB{color} → {hex_code}")

        except Exception as e:
            st.error(f"Ошибка при анализе: {e}")
        finally:
            # Удаляем временный файл
            os.unlink(tmp_path)

# Футер
st.markdown("---")
st.markdown("### 🎵 Создано с помощью Music Color Analyzer")