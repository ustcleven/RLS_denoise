import scipy.signal as sc

import params
import plots
import helper
import hotwordCleaner

if __name__ == "__main__":

    fs, signal_1, signal_2 = helper.separate_stereo_channels('signal2')

    f_1, t_1, Zxx_1 = sc.stft(signal_1, fs, nperseg=params.WINDOW_SIZE, noverlap=params.OVERLAP_PERCENTAGE)
    f_2, t_2, Zxx_2 = sc.stft(signal_2, fs, nperseg=params.WINDOW_SIZE, noverlap=params.OVERLAP_PERCENTAGE)

    plots.spectrogram(t_1, f_1, Zxx_1)
    plots.spectrogram(t_2, f_2, Zxx_2)

    x_1 = Zxx_1.T
    x_2 = Zxx_2.T
    epsilon = hotwordCleaner.stft_domain_fast_rls_defered_filter_coefs(x_1, x_2)
