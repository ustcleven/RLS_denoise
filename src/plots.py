import matplotlib.pyplot as plt
import numpy as np


def spectrogram(t, f, Zxx):
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
