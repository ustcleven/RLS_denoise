import scipy.signal as sc
from scipy.io import wavfile
import numpy as np

import params
import helper
import STFT_fast_RLS

if __name__ == "__main__":

    fs, signal_1, signal_2, filename_1, filename_2 = helper.separate_stereo_channels(params.FILENAME)

    # audio_file = params.DATA_PATH+filename_1
    # return_code = subprocess.call(["afplay", audio_file])

    n_overlap = params.OVERLAP_PERCENTAGE*params.WINDOW_SIZE/100
    f_1, t_1, Zxx_1 = sc.stft(signal_1, fs, nperseg=params.WINDOW_SIZE, noverlap=n_overlap)
    f_2, t_2, Zxx_2 = sc.stft(signal_2, fs, nperseg=params.WINDOW_SIZE, noverlap=n_overlap)

    print(Zxx_1.shape)

    # plots.spectrogram(t_1, f_1, Zxx_1)
    # plots.spectrogram(t_2, f_2, Zxx_2)

    epsilon = STFT_fast_RLS.stft_domain_fast_rls_defered_filter_coefs(Zxx_1, Zxx_2)
    print(epsilon.shape)
    epsilon_time = sc.istft(epsilon, fs, nperseg=params.WINDOW_SIZE, noverlap=n_overlap)

    print(signal_1.shape)
    print(np.shape(epsilon_time[0]))

    wavfile.write(params.DATA_PATH+params.FILENAME+'_output'+params.AUDIO_FORMAT, fs, epsilon_time[0])
