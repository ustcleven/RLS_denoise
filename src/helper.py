from scipy.io import wavfile
import params


def separate_stereo_channels(filename):
    fs, signal = wavfile.read(params.DATA_PATH + filename + params.AUDIO_FORMAT)
    output_filenames = [params.DATA_PATH + filename + '_0' + params.AUDIO_FORMAT,
                        params.DATA_PATH + filename + '_1' + params.AUDIO_FORMAT]

    wavfile.write(output_filenames[0], fs, signal[:, 0])
    wavfile.write(output_filenames[1], fs, signal[:, 1])

    _, signal_1 = wavfile.read(output_filenames[0])
    _, signal_2 = wavfile.read(output_filenames[1])

    return fs, signal_1, signal_2
