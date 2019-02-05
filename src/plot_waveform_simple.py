import wave
import numpy as np
import matplotlib.pyplot as plt

# when label exists, save the figure as '{label}.png'
def plot_wave(filename, label = ''):
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

    # make time axis
    x = np.arange(0, float(nframes)/framerate, 1.0/framerate)

    # plot each channnel
    for i in range(nchannels):
        # Tentatively
        if i == 0:
            plt.plot(x, data[i::nchannels])

    plt.title(filename.split('/')[-1])
    plt.xlabel("time[s]")
    plt.ylabel("amplitude[dB]")
    plt.ylim([-1, 1])
    if (label):
        plt.savefig('../figure/' + label)
    plt.show()

########
# main #
########

target_wave = '../media/intermittent_ch2.wav'
plot_wave(target_wave, '01')