import wave
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

# when label exists, save the figure as '{label}.png'
def plot_spectrum(filename, label = ''):
    print('loading : ' + filename)

    wf = wave.open(filename, 'rb')
    nchannels, sampwidth, framerate, nframes, comptype, compname = wf.getparams()

    print("nchannels", nchannels)
    print("sampwidth(byte)", sampwidth)
    print("framerate", framerate)
    print("nframes", nframes)
    print("sec", nframes / framerate)

    # read data
    buf = wf.readframes(nframes)

    # to exchenge data byte to int
    # data must be a multiple of 2 : byte(8) => int16
    data_length = len(buf)
    if sampwidth == 2:
        offset = data_length - data_length % 2
        buf = buf[0:offset]
        data = np.frombuffer(buf, dtype='int16')
    elif sampwidth == 4:
        offset = data_length - data_length % 4
        buf = buf[0:offset]
        data = np.frombuffer(buf, dtype='int32')
    # normalize the amplitude from -1 to 1
    amp = (2 ** 8) ** sampwidth / 2
    data = data / amp

    # plot each channnel
    for i in range(nchannels):
        ch_data = data[i::nchannels]
        x = fftpack.fftfreq(n=ch_data.size, d=1 / framerate)
        y = 10 * np.log10(np.abs(fftpack.fft(ch_data).real))
        plt.plot(x, y, label="ch"+ str(i+1))

    plt.xlim(0, framerate/2)  # remove -sampling_frequency/2 ~ 0
    # plt.xlim(400, 480)  # remove -sampling_frequency/2 ~ 0
    # plt.ylim(0, 50)
    plt.legend(loc=1)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title(target_wave.split("/")[-1])
    if (label):
        plt.savefig('../figure/' + label)
    plt.show()

########
# main #
########

# target_wave = '../media/continuous_ch2.wav'
# target_wave = '../media/intermittent_ch2.wav'
# target_wave = '../media/a.wav'
# target_wave = '../media/i.wav'
# target_wave = '../media/u.wav'
# target_wave = '../media/e.wav'
# target_wave = '../media/o.wav'
target_wave = '../media/440.wav'

# plot_spectrum(target_wave, target_wave.split("/")[-1].split(".")[0] )
plot_spectrum(target_wave)