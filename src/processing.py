import librosa
import scipy.signal as sc
import numpy as np

import paths
import helper


def rir_filter(rir_filename, rir_fs, s, resampling_f=0):
    rir_1, _ = librosa.load('{}/{}{}'.format(paths.RIR_PATH, rir_filename, paths.AUDIO_FORMAT), rir_fs)
    if resampling_f != 0:
        rir_1 = sc.resample(rir_1, resampling_f)
    s_filtered = sc.lfilter(rir_1, 1, s)
    return s_filtered


def add_speech(speech_filename, signal_0, signal_1, amp, position='end'):
    fs, _, _, speech_filename_0, _ = helper.separate_stereo_channels(speech_filename)

    speech, fs = librosa.load(paths.DATA_PATH / (speech_filename_0 + paths.AUDIO_FORMAT), fs)

    n_samples = len(signal_0)
    n_samples_speech = len(speech)

    if position == 'mid':
        signal_0[int(n_samples/2) - n_samples_speech:int(n_samples/2)] += speech * amp
        signal_1[int(n_samples/2) - n_samples_speech:int(n_samples/2)] += speech * amp
    else:
        position = 'end'
        signal_0[n_samples - n_samples_speech:n_samples] += speech * amp
        signal_1[n_samples - n_samples_speech:n_samples] += speech * amp

    print('Speech added at position {} of the signal.'.format(position))

    return signal_0, signal_1


def add_delay(distance_mics, fs, signal_0):
    n_samples_delay = int((distance_mics/340.0)*fs)
    signal_0_delayed = [0] * n_samples_delay + list(signal_0[:len(signal_0) - n_samples_delay])
    signal_1 = np.asarray(signal_0_delayed)
    print('Delay: {} samples'.format(n_samples_delay))
    return signal_0, signal_1
