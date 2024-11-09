import os

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


def GetSoundData(sample_rate, duration, frequency, amplitude, bit_depth=16):

    # 基本參數設定
    # sample_rate = 44100  # 取樣率
    # duration = 30  # 持續時間（秒）
    # frequency = 852  # 頻率（Hz，此為標準 A 音）
    # amplitude = 0.5  # 振幅（0-1之間）

    # 生成時間序列
    t = np.linspace(0, duration, int(sample_rate * duration))

    # 生成正弦波
    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    if bit_depth == 16:
        # 將浮點數轉換為 16 位整數
        # signal = np.int16(signal * 32767)
        signal = (signal * (2 ** (bit_depth - 1) - 1)).astype(np.int16)
    else:
        # 將浮點數轉換為 位元深度 整數
        signal = (signal * (2 ** (bit_depth - 1) - 1)).astype(np.int32)

    return t, signal


def WavToMp3(wavFileName, oMp3FileName):
    # 載入 WAV 檔案
    sound = AudioSegment.from_wav(wavFileName)

    # 輸出為 MP3 檔案
    sound.export(oMp3FileName, format="mp3")


def WavToFlac(wavFileName, oFlacFileName):
    # 載入 WAV 檔案
    sound = AudioSegment.from_wav(wavFileName)

    # 輸出為 MP3 檔案
    sound.export(oFlacFileName, format="flac")


def DrawWave(wavFileName):
    sample_rate, data = wavfile.read(wavFileName)
    duration = len(data) / sample_rate
    t = np.linspace(0, duration, len(data))

    print("採樣率：", sample_rate)
    print("數據形狀：", data.shape)
    print("數據類型：", data.dtype)
    print("頻道數量：", len(data.shape))

    if len(data.shape) == 1:
        plt.rcParams["font.family"] = ["Taipei Sans TC Beta", "sans-serif"]
        plt.figure(figsize=(10, 5))
        plt.plot(t[:1000], data[:1000])  # 只顯示前 1000 個樣本點
        plt.title("正弦波形圖", color="b")
        plt.xlabel("時間 (秒)")
        plt.ylabel("振幅")
        plt.grid(True)
        plt.show()

    if len(data.shape) == 2:
        # 選擇左聲道
        left_channel = data[:, 0]

        # 選擇右聲道
        right_channel = data[:, 1]
        # 執行 FFT
        # l_fft_data = np.fft.fft(left_channel)
        # r_fft_data = np.fft.fft(right_channel)

        # 計算頻率bins
        # l_frequencies = np.fft.fftfreq(len(l_fft_data), 1 / sample_rate)
        # r_frequencies = np.fft.fftfreq(len(r_fft_data), 1 / sample_rate)

        # 計算幅度
        # l_magnitudes = np.abs(l_fft_data)
        # r_magnitudes = np.abs(r_fft_data)

        plt.rcParams["font.family"] = ["Taipei Sans TC Beta", "sans-serif"]

        plt.figure(figsize=(10, 6))
        plt.subplot(211)
        plt.title("左聲道波形")
        plt.plot(t[:1000], left_channel[:1000])
        plt.ylabel("振幅")

        plt.subplot(212)
        plt.title("右聲道波形")
        plt.plot(t[:1000], right_channel[:1000])
        plt.xlabel("時間（秒）")
        plt.ylabel("振幅")

        plt.tight_layout()
        plt.show()


def SaveStereoChannel(wavFileName, sample_rate, signal):
    oWavFileName = f"{wavFileName}.wav"
    # 儲存為 WAV 檔案
    wavfile.write(oWavFileName, sample_rate, signal)

    return wavFileName, oWavFileName


def SaveDiffChannel(wavFileName, sample_rate, r_signal, l_signal):
    # 創建立體聲數組，設定左右聲道設為 0
    stereo_data = np.zeros((len(l_signal), 2), dtype=l_signal.dtype)

    stereo_data[:, 0] = l_signal  # 左聲道
    stereo_data[:, 1] = r_signal  # 右聲道設為 0

    oWavFileName = f"Diff_{wavFileName}.wav"
    # 儲存為 WAV 檔案
    wavfile.write(oWavFileName, sample_rate, stereo_data)

    return wavFileName, oWavFileName


def SaveLeftChannel(wavFileName, sample_rate, signal):
    # 創建立體聲數組，設定左右聲道設為 0
    stereo_data = np.zeros((len(signal), 2), dtype=np.int16)

    stereo_data[:, 0] = signal  # 左聲道
    stereo_data[:, 1] = 0  # 右聲道設為 0

    oWavFileName = f"L_{wavFileName}.wav"
    # 儲存為 WAV 檔案
    wavfile.write(oWavFileName, sample_rate, stereo_data)

    return wavFileName, oWavFileName


def SaveRightChannel(wavFileName, sample_rate, signal):
    # 創建立體聲數組，設定左右聲道設為 0
    stereo_data = np.zeros((len(signal), 2), dtype=np.int16)

    stereo_data[:, 0] = 0  # 左聲道設為 0
    stereo_data[:, 1] = signal  # 右聲道

    oWavFileName = f"R_{wavFileName}.wav"
    # 儲存為 WAV 檔案
    wavfile.write(oWavFileName, sample_rate, stereo_data)

    return wavFileName, oWavFileName


def merge_mp3_files(file_list, output_file):
    # 初始化合併音訊
    combined = AudioSegment.empty()

    # 依序讀取並合併檔案
    for file in file_list:
        audio = AudioSegment.from_mp3(file)
        combined += audio

    # 輸出合併後的檔案
    combined.export(output_file, format="mp3")


def merge_flac_files(file_list, output_file):
    # 初始化合併音訊
    combined = AudioSegment.empty()

    # 依序讀取並合併檔案
    for file in file_list:
        audio = AudioSegment.from_file(file)
        combined += audio

    # 輸出合併後的檔案
    combined.export(output_file, format="flac")


def GenerateDiffSignal(
    sample_rate, l_frequency, r_frequency, amplitude, duration, bit_depth=16
):
    # sample_rate = 48000  # 取樣率
    # duration = 300  # 持續時間（秒）
    # amplitude = 0.75  # 振幅（0-1之間）

    # r_frequency = 303
    # l_frequency = 300

    wavFileName = f"sine_wave_L{l_frequency}_R{r_frequency}_{abs(l_frequency-r_frequency)}Hz_{duration}s"

    t, r_signal = GetSoundData(
        sample_rate=sample_rate,
        duration=duration,
        frequency=r_frequency,
        amplitude=amplitude,
        bit_depth=bit_depth,
    )

    t, l_signal = GetSoundData(
        sample_rate=sample_rate,
        duration=duration,
        frequency=l_frequency,
        amplitude=amplitude,
        bit_depth=bit_depth,
    )

    # 儲存為 WAV 檔案
    _, oWavFileName = SaveDiffChannel(wavFileName, sample_rate, r_signal, l_signal)

    return wavFileName, oWavFileName


def GenerateSignal(sample_rate, frequency, amplitude, duration):
    # 基本參數設定
    # sample_rate = 48000  # 取樣率
    # duration = 300  # 持續時間（秒）
    # frequency = 963  # 頻率（Hz）
    # frequency = 852  # 頻率（Hz）
    # frequency = 741  # 頻率（Hz）
    # frequency = 639  # 頻率（Hz）
    # frequency = 528  # 頻率（Hz）
    # frequency = 417  # 頻率（Hz）
    # frequency = 396  # 頻率（Hz）
    # frequency = 285  # 頻率（Hz）
    # frequency = 174  # 頻率（Hz）
    # frequency = 2  # 頻率（Hz）
    # amplitude = 0.75  # 振幅（0-1之間）

    wavFileName = f"sine_wave_{frequency}Hz_{duration}s"

    t, signal = GetSoundData(
        sample_rate=sample_rate,
        duration=duration,
        frequency=frequency,
        amplitude=amplitude,
    )

    # 儲存為 WAV 檔案
    _, oWavFileName = SaveStereoChannel(
        wavFileName=wavFileName, sample_rate=sample_rate, signal=signal
    )
    # SaveLeftChannel(wavFileName, sample_rate, signal)
    # SaveRightChannel(wavFileName, sample_rate, signal)

    return wavFileName, oWavFileName


def main():
    """
    WavFileName, oWavFileName = GenerateSignal(
        sample_rate=48000,
        frequency=40,
        amplitude=0.75,
        duration=1200,
    )

    WavToFlac(oWavFileName, f"{WavFileName}.flac")

    DrawWave(oWavFileName)

    try:
        os.remove(oWavFileName)
        print(f"File '{oWavFileName}' deleted successfully.")
    except FileNotFoundError:
        print(f"File '{oWavFileName}' not found.")
    """
    # frequency_list = [432, 174, 285, 396, 417, 528, 639, 741, 852, 963]
    # frequency_list = [396, 417, 528, 639, 741, 852, 963]
    frequency_list = [
        432,
        432 + 36,
        432 + 36 + 36,
        432 + 36 + 36 + 36,
        432 + 36 + 36 + 36 + 36,
        432 + 36 + 36 + 36 + 36 + 36,
        432 + 36 + 36 + 36 + 36 + 36 + 36,
        432 + 36 + 36 + 36 + 36 + 36 + 36 + 36,
    ]

    file_list = list()

    for frequency in frequency_list:
        WavFileName, oWavFileName = GenerateSignal(
            sample_rate=48000,
            frequency=frequency,
            amplitude=0.75,
            duration=600,
        )

        oFlacFileName = f"{WavFileName}.flac"

        WavToFlac(oWavFileName, oFlacFileName)

        file_list.append(oFlacFileName)

        try:
            os.remove(oWavFileName)
            print(f"File '{oWavFileName}' deleted successfully.")
        except FileNotFoundError:
            print(f"File '{oWavFileName}' not found.")

    merge_flac_files(file_list, "DoReMi_10min.flac")
    """
    frequency_list = [
        # [480, 440],
        # [440, 480],
        # [340, 300],
        # [300, 340],
        # [872, 832],
        # [832, 872],
        # [180, 182],
        # [182, 180],
        # [88, 48],
        # [48, 88],
        # [300, 303],
        # [303, 300],
        # [300, 306],
        # [306, 300],
        # [300, 309],
        # [309, 300],
        # [300, 310],
        # [310, 300],
        # [180, 190],
        # [190, 180],
        # [300, 300.5],
        # [300.5, 300],
        # [432, 433.50],
        # [433.5, 432],
        # [160, 163],
        # [163, 160],
        # [160, 161.5],
        # [161.5, 160],
        [140, 180],
        [180, 140],
        # [36000, 16000],
        # [16000, 36000],
    ]

    for frequencys in frequency_list:
        WavFileName, oWavFileName = GenerateDiffSignal(
            sample_rate=48000,
            l_frequency=frequencys[0],
            r_frequency=frequencys[1],
            amplitude=0.75,
            duration=1200,
            bit_depth=32,
        )

        WavToFlac(oWavFileName, f"Diff_{WavFileName}.flac")
        # DrawWave(oWavFileName)

        try:
            os.remove(oWavFileName)
            print(f"File '{oWavFileName}' deleted successfully.")
        except FileNotFoundError:
            print(f"File '{oWavFileName}' not found.")
    """
    """
    WavFileName, oWavFileName = GenerateDiffSignal(
        sample_rate=48000,
        l_frequency=480,
        r_frequency=440,
        amplitude=0.75,
        duration=600,
        bit_depth=32,
    )

    WavToFlac(oWavFileName, f"Diff_{WavFileName}.flac")

    DrawWave(oWavFileName)
    
    try:
        os.remove(oWavFileName)
        print(f"File '{oWavFileName}' deleted successfully.")
    except FileNotFoundError:
        print(f"File '{oWavFileName}' not found.")
    """
    """
    merge_mp3_files(
        [
            "Ranning_50min.mp3",
            "Ranning_50min.mp3",
            "Ranning_10min.mp3",
        ],
        "Ranning_110min.mp3",
    )
    """
    """
    merge_flac_files(
        [
            "Diff_sine_wave_L300_R306_6Hz_1200s.flac",
            "Diff_sine_wave_L306_R300_6Hz_1200s.flac",
            "Diff_sine_wave_L300_R306_6Hz_1200s.flac",
            "Diff_sine_wave_L306_R300_6Hz_1200s.flac",
        ],
        "Diff_6Hz_80min.flac",
    )
    """


if __name__ == "__main__":
    main()
