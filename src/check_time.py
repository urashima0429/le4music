import numpy as np
import time
from scipy import fftpack
import matplotlib.pyplot as plt

def mydft(data):
    n = data.shape[0]
    x = np.zeros_like(data, dtype=np.complex)
    for i in range(n): #各周波数に関して
        w = np.exp(-1j * 2 * np.pi * i / float(n))
        for j in range(n): #信号*重みの総和をとる
            x[i] += data[j] * (w ** j)
    return x

def myfft(data):
    n = data.shape[0]
    x = np.zeros_like(data, dtype=np.complex)

    if n > 2:
        even = myfft(x[0:n:2]) # 0,2,4,6 ...
        odd  = myfft(x[1:n:2]) # 1,3,5,7 ...

        me = np.exp(-1j * (2 * np.pi * np.arange(0, n/2))/float(n))
        x[0:int(n/2)] = even + me * odd
        x[int(n/2):n] = even - me * odd
        return x

    else:
        x[0] = data[0] + data[1]
        x[1] = data[0] - data[1]
        return x

# make test wave1
start = 0
stop = 1
sampling_frequency = 64
hoge = np.linspace(start,stop,int( (stop-start)*sampling_frequency ),endpoint=False)
wave = np.cos(2 * np.pi * hoge * 20)+ np.cos(2 * np.pi * hoge * 30) + np.sin(2 * np.pi * hoge * 1)

time0 = time.time()

# fft_wave = fftpack.fft(wave)
# fft_wave = np.fft.fft(wave)
# fft_wave = mydft(wave)
fft_wave = myfft(wave)

time1 = time.time() - time0
print(time1)

fft_fre = fftpack.fftfreq(n=wave.size, d=1/sampling_frequency)

plt.plot(fft_fre,fft_wave.real,label="real part")   # remove imaginary part
plt.xlim(0,sampling_frequency/2)                    # remove -sampling_frequency/2 ~ 0
plt.ylim(0, sampling_frequency * ((stop-start)/2) * 1.3)
plt.legend(loc=1)
plt.xlabel("frequency (Hz)")
plt.show()