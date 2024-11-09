import numpy as np
from scipy.io import wavfile

# import sounddevice as sd


def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


# 播放 440Hz (A4) 的音符
frequency = 440  # Hz
duration = 1.0  # 秒
wave = generate_sine_wave(frequency, duration)
# sd.play(wave, 44100)
# sd.wait()


def generate_instrument_tone(frequency, duration, harmonics, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.zeros_like(t)

    for i, amplitude in enumerate(harmonics, 1):
        wave += amplitude * np.sin(2 * np.pi * frequency * i * t)

    return wave / np.max(np.abs(wave))  # 正規化


def apply_adsr(wave, sample_rate, attack, decay, sustain, release):
    total_length = len(wave)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)

    envelope = np.ones(total_length)

    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay
    decay_end = attack_samples + decay_samples
    envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)

    # Sustain
    envelope[decay_end:-release_samples] = sustain

    # Release
    envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)

    return wave * envelope


def save_wave(filename, wave, sample_rate=44100):
    # 確保波形在 -1 到 1 之間
    wave = np.clip(wave, -1, 1)
    # 轉換為 16-bit PCM
    wave_int = np.int16(wave * 32767)
    wavfile.write(filename, sample_rate, wave_int)


"""
# 使用範例
frequency = 440 + 44
duration = 2.0
# 模擬鋼琴音色（簡化版）
harmonics = [1.0, 0.5, 0.25, 0.125]  # 泛音振幅

# 生成基本音色
wave = generate_instrument_tone(frequency, duration, harmonics)

# 加入 ADSR 包絡
wave = apply_adsr(wave, 44100, attack=0.1, decay=0.2, sustain=0.7, release=0.4)

# 儲存檔案
save_wave("piano_note1.wav", wave)
"""
frequency_list = [
    440,
    440 + 44,
    440 + 44 + 44,
    440 + 44 + 44 + 44,
    440 + 44 + 44 + 44 + 44,
    440 + 44 + 44 + 44 + 44 + 44,
    440 + 44 + 44 + 44 + 44 + 44 + 44,
]

for frequency in frequency_list:
    duration = 2.0
    harmonics = [1.0, 0.5, 0.25, 0.125]

    # 生成基本音色
    wave = generate_instrument_tone(frequency, duration, harmonics)

    # 加入 ADSR 包絡
    wave = apply_adsr(wave, 44100, attack=0.1, decay=0.2, sustain=0.7, release=0.4)

    # 儲存檔案
    save_wave(f"piano_note_{frequency}Hz.wav", wave)
