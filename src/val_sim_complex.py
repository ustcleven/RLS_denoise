import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import scipy
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import matplotlib
import time
import argparse

font = {        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

def callback_mix(premix, dsr_g=0, sir=0, ref_mic=0, flag_i = True):

    # first normalize all separate recording to have unit power at microphone one
    peak_abs = np.max(abs(premix[:,ref_mic,:]))
    mic_norm = premix * 1.0/peak_abs * dsr_g
    
    #print(dsr_g)
    if flag_i:
        initial_snr = np.sum(np.power(mic_norm[0 ,ref_mic,:], 2)) / (1e-15 + np.sum(np.power(mic_norm[1,ref_mic,:], 2)))        
        snr_ratio = 10.0**(sir / 10.0)
        #print(snr_ratio)
        noise_coeff = np.sqrt(initial_snr / snr_ratio)
        mic_norm[1,:,:] *= noise_coeff
    mic_norm = np.sum(mic_norm, axis = 0)
    
    return np.clip(mic_norm, -1.0, 1.0)

class val_simple_afe:
    def __init__(self):
        self.fs = 16000
        self.absorp_rate = 0.7
        self.room = pra.ShoeBox([4.34,5.95, 2.73], fs=self.fs, absorption = self.absorp_rate, max_order = 17)
        self.source = np.array([2.15, 4.5, .78])
        self.interferer = np.array([2.8642, 3.4142, .78 ])
        mic_pos = np.c_[
            [1.45, 1.9645, .78],  # mic 3
            [1.45, 2.0355, .78],  # mic 1
        ]
        self.mics = pra.MicrophoneArray(mic_pos, self.room.fs)
        self.room.add_microphone_array(self.mics)

        self.callback_mix_kwargs = {
            'dsr_g' : 1,
            'sir' : 10,  # SIR target is 10 decibels
            'ref_mic' : 0,
            'flag_i' : True
        }

        self.poly_filt =  np.loadtxt('/home/nfsdata/fixtures/subband/subband_coefficients_hifi3.txt', dtype='float', comments='#')

    def audio_lab_augmentation_only(self, utterance_np_array, noise_np_array, desired_snr, desired_gain, mix_with_bg):
        """use pyroom acoustics to mix two signls
        
        Returns:
            (np array): num_mic * samples
        """
        self.room.add_source(self.source, delay=0., signal=utterance_np_array)
        
        if mix_with_bg:
            self.room.add_source(self.interferer, delay=0., signal=noise_np_array)
        else:
            self.room.add_source(self.interferer, delay=0., signal=np.zeros(utterance_np_array.shape))
        self.callback_mix_kwargs['dsr_g'] = desired_gain
        self.callback_mix_kwargs['sir'] = desired_snr
        self.callback_mix_kwargs['flag_i'] = mix_with_bg
        # Run the simulation
        self.room.simulate(
                callback_mix=callback_mix,
                callback_mix_kwargs=self.callback_mix_kwargs,
                )
        mics_signals = self.room.mic_array.signals
        return mics_signals



test = val_simple_afe()
FS, test_uter = read(args.utterance)    
FS, test_interf = read(args.interference)    
start = time.time()
#complex_mic0, complex_mic1, mics_signals = test.audio_lab_augmentation(test_uter[0:16000], test_interf[0:16000], 15, 0.5, True)
mics_signals = test.audio_lab_augmentation(test_uter, test_interf[0:len(test_uter)], 5, 0.8, True)
end = time.time()
print(end - start)
#print(mics_signals.shape)
#print(complex_mic0.shape)
mic_sigs = np.transpose(mics_signals)
write('test_val_simple.wav', rate=FS, data=mic_sigs)