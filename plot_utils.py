import numpy as np
import matplotlib.pyplot as plt

def plot_sine_wave(frequency, amplitude, duration, sampling_rate=1000):
    """
    繪製正弦波

    參數:
    frequency -- 頻率 (Hz)
    amplitude -- 振幅
    duration -- 持續時間 (秒)
    sampling_rate -- 取樣率 (默認為 1000)
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    plt.plot(t, y)
    plt.title(f"Sine Wave: {frequency}Hz, {amplitude} Amplitude")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
