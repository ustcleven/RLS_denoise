import paths

from scipy.io import wavfile
from IPython.display import clear_output
import librosa
import librosa.core as lc
import matplotlib.pyplot as plt
import numpy as np


def separate_stereo_channels(filename):
    fs, signal = wavfile.read('{}/{}{}'.format(paths.DATA_PATH, filename, paths.AUDIO_FORMAT))

    filename_0 = '{}_0'.format(filename)
    filename_1 = '{}_1'.format(filename)

    wavfile.write(paths.DATA_PATH / (filename_0 + paths.AUDIO_FORMAT), fs, signal[:, 0])
    wavfile.write(paths.DATA_PATH / (filename_1 + paths.AUDIO_FORMAT), fs, signal[:, 1])

    _, signal_0 = wavfile.read(paths.DATA_PATH / (filename_0 + paths.AUDIO_FORMAT))
    _, signal_1 = wavfile.read(paths.DATA_PATH / (filename_1 + paths.AUDIO_FORMAT))

    return fs, signal_0, signal_1, filename_0, filename_1


def load_separate_channels(filename):
    fs, _, _, filename_0, filename_1 = separate_stereo_channels(filename)

    signal_0, fs = librosa.load(paths.DATA_PATH / (filename_0 + paths.AUDIO_FORMAT), fs)
    signal_1, fs = librosa.load(paths.DATA_PATH / (filename_1 + paths.AUDIO_FORMAT), fs)

    return signal_0, signal_1, fs


def dual_stft(signal_0, signal_1, window_size, hop_percentage):
    hop_length = int(hop_percentage * window_size / 100)

    Zxx_0 = lc.stft(signal_0, n_fft=window_size, hop_length=hop_length)
    Zxx_1 = lc.stft(signal_1, n_fft=window_size, hop_length=hop_length)

    n_frames = Zxx_0.shape[1]
    n_freqs = Zxx_0.shape[0]

    print('Number of frames: {}'.format(n_frames))
    print('Frequency resolution: {}'.format(n_freqs))

    return Zxx_0, Zxx_1, n_freqs, n_frames, hop_length


def spectrogram(Zxx, Zxx_output, amp):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 10))

    z = np.abs(Zxx)
    z_min, z_max = np.abs(z).min(), np.abs(z).max() * amp

    z_output = np.abs(Zxx_output)
    z_output_min, z_output_max = np.abs(z_output).min(), np.abs(z_output).max() * amp

    ax1.pcolormesh(z, vmin=z_min, vmax=z_max)
    ax1.set_title('STFT Magnitude')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time [sec]')

    ax2.pcolormesh(z_output, vmin=z_output_min, vmax=z_output_max)
    ax2.set_title('STFT Magnitude')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [sec]')


def plot_signals(signal_0, signal_1, title, signal_0_name, signal_1_name):
    plt.figure(figsize=(15, 7))
    plt.plot(signal_0, 'b')
    plt.plot(signal_1, 'r')
    plt.title(title)
    plt.legend((signal_0_name, signal_1_name), loc='upper right')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')


def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block), progress * 100)
    print(text)
