import wave
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# when label exists, save the figure as '{label}.png'
def plot_spectrogram(filename, label = ''):
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

    # debug
    x = data[0::nchannels]
    sampling_rate = framerate

    NFFT = 2**12 # フレームの大きさ
    OVERLAP = NFFT / 2 # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
    frame_length = data.shape[0] # wavファイルの全フレーム数
    time_song = float(frame_length) / sampling_rate  # 波形長さ(秒)
    time_unit = 1 / float(sampling_rate) # 1サンプルの長さ(秒)

    # FFTのフレームの時間を決めていきます
    # time_rulerに各フレームの中心時間が入っています
    start = (NFFT / 2) * time_unit
    stop = time_song
    step =  (NFFT - OVERLAP) * time_unit
    time_ruler = np.arange(start, stop, step)

    window = np.hamming(NFFT)

    spec = np.zeros([len(time_ruler), 1 + int(NFFT / 2)]) #転置状態で定義初期化
    pos = 0

    for fft_index in range(len(time_ruler)):
        # 💥 1.フレームの切り出します
        frame = x[pos:pos+NFFT]
        # フレームが信号から切り出せない時はアウトです
        if len(frame) == NFFT:
            # 💥 2.窓関数をかけます
            windowed = window * frame
            # 💥 3.FFTして周波数成分を求めます
            # rfftだと非負の周波数のみが得られます
            fft_result = np.fft.rfft(windowed)
            # 💥 4.周波数には虚数成分を含むので絶対値をabsで求めてから2乗します
            # グラフで見やすくするために対数をとります
            fft_data = 2 * np.log(np.abs(fft_result))
            # fft_data = np.log(np.abs(fft_result))
            # fft_data = np.abs(fft_result) ** 2
            # fft_data = np.abs(fft_result)
            # これで求められました。あとはspecに格納するだけです
            for i in range(len(spec[fft_index])):
                spec[fft_index][-i-1] = fft_data[i]

            # 💥 4. 窓をずらして次のフレームへ
            pos += int(NFFT - OVERLAP)

    print(spec.shape)
    ### プロットします
    # matplotlib.imshowではextentを指定して軸を決められます。aspect="auto"で適切なサイズ比になります
    plt.imshow(spec.T, extent=[0, time_song, 0, sampling_rate/2], aspect="auto")
    plt.xlabel("time[s]")
    plt.ylabel("frequency[Hz]")
    # plt.xlim(0, 0.4)
    # plt.ylim(0, 2500)
    plt.colorbar()
    plt.show()

########
# main #
########

# target_wave = '../media/continuous_ch2.wav'
target_wave = '../media/intermittent_ch2.wav'
# target_wave = '../media/a.wav'
# target_wave = '../media/i.wav'
# target_wave = '../media/u.wav'
# target_wave = '../media/e.wav'
# target_wave = '../media/o.wav'
# target_wave = '../media/440.wav'

# plot_spectrogram(target_wave, target_wave.split("/")[-1].split(".")[0] )
plot_spectrogram(target_wave)