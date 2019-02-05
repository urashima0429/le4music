import wave
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import fftpack
import sys
from memory_profiler import profile

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


# constant
window_width = 1000
window_height = 800
margin_width = 30
margin_height = 150

item_num = 4

class MainWindow(pg.QtGui.QMainWindow):

    def __init__(self):
        super().__init__()
        self.speech_recognition()
        # self.load_wavefile('../media/test2.wav')
        self.load_wavefile('../media/continuous_ch2.wav')
        # self.load_wavefile('../media/intermittent_ch2.wav')

        # self.cul_fundamental_frequency()
        self.sensitivity = self.wave_period / 100
        self.is_left_tracking = False
        self.is_right_tracking = False

        self.init_main_window()

        self.vbox0 = pg.QtGui.QVBoxLayout()

        # spec
        spec = self.specrtogram()
        self.iw = self.__plot_image(spec)
        self.vbox0.addWidget(self.iw)

        # select wave
        self.init_selection_window()
        self.vbox0.addWidget(self.sw)

        # show wave detail
        self.init_detail_window()
        self.vbox0.addWidget(self.dw)

        # show selected wave spec and ceps
        self.cul_spectrum()
        self.cul_cepstrum()
        self.init_spectrum_cepstrum_window()
        self.vbox0.addWidget(self.scpw)

        centralWid = pg.QtGui.QWidget()
        centralWid.setLayout(self.vbox0)
        self.setCentralWidget(centralWid)

        return

    def init_main_window(self):
        # window setting
        self.setMinimumSize(window_width + margin_width, window_height + margin_height)
        self.setMaximumSize(window_width + margin_width, window_height + margin_height)
        self.setWindowTitle('le4music')
        self.setWindowIcon(pg.QtGui.QIcon('../icon/pank.png'))
        pg.setConfigOption('background', 'w')

        # status bar
        self.statusBar().showMessage('Ready')

        # menu bar
        # exit
        exitAction = pg.QtGui.QAction(pg.QtGui.QIcon('../icon/exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtWidgets.qApp.quit)

        # open wav
        openfileAction = pg.QtGui.QAction(pg.QtGui.QIcon('../icon/wav.png'), '&Open', self)
        openfileAction.setShortcut('Ctrl+O')
        openfileAction.setStatusTip('Open new File')
        openfileAction.triggered.connect(self.open_wavefile)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(openfileAction)
        menubar.show()

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAction)
        self.toolbar.addAction(openfileAction)
        return

    def update_selection_window(self):
        self.source_wave.setData(self.x, self.y)
        self.cul_fundamental_frequency()
        self.fundamental_frequency.setData(self.x, 1.8 * self.y2/ (self.y2.max() - self.y2.min()) - 0.9)
        self.sw.setXRange(self.x[0], self.x[-1], 0)
        return

    # @profile
    def init_selection_window(self):

        self.sw = pg.PlotWidget(
            viewBox=pg.ViewBox(
                border=pg.mkPen(color='#000000'),
                invertX=False, invertY=False
            )
        )
        self.sw.setMinimumSize(window_width - margin_width, window_height / item_num)
        self.sw.setMaximumSize(window_width - margin_width, window_height / item_num)
        # self.setCentralWidget(pw)
        self.sw.setBackground("#FFFFFF00")
        self.source_wave = pg.PlotCurveItem(
                self.x,
                self.y,
                pen=pg.mkPen(color="b"),
                antialias=True
            )
        self.sw.addItem(self.source_wave)


        # make up data
        N = 2 ** 9
        sence = 0.5
        length = int(self.sample_length / N)
        stride = self.y.strides[0]
        f0 = np.zeros(length)
        trimmer = np.rot90(np.triu(np.ones((N, N))))

        a = self.y[0:-1:] * self.y[1::] < 0
        b = np.zeros(length)
        for i in range(length):
            b[i] = np.sum(a[i * N:(i+1) * N]) * self.sampling_rate / N

        for i in range(length-1):
            t0 = N * (i+1)
            tmp = self.y[t0:t0+N]
            Xt = as_strided(tmp, (N, N), (0, stride))
            Xt_r = as_strided(tmp, (N, N), (-stride, stride))
            ac = np.sum(Xt * Xt_r * trimmer, axis=1)
            dydx = ac[1::] - ac[:-1:]

            tmp2 = np.where((dydx[:-1:] >= 0) & (dydx[1::] < 0))[0]
            if tmp2.shape[0] == 0:
                f0[i] = 0
            else:
                tmp3 = self.sampling_rate / (tmp2[0] + 1)
                f0[i] = tmp3

        sorted = np.sort(f0)
        q3 = sorted[-int(sorted.shape[0] / 4)]
        q1 = sorted[int(sorted.shape[0] / 4)]
        qrange = q3 - q1
        qmax = q3 + 1.5 * qrange
        qmin = q1 - 1.5 * qrange

        for i in range(length):
            if (f0[i] < qmin) | (f0[i] > qmax):
                if i==0:
                    f0[i] = f0[i + 1]
                elif i == length-1:
                    f0[i] = f0[i - 1]
                else:
                    f0[i] = (f0[i-1] + f0[i+1]) / 2

        sorted = np.sort(b)
        q3 = b[-int(sorted.shape[0] / 4)]
        q1 = b[int(sorted.shape[0] / 4)]
        qrange = q3 - q1
        qmax = q3 + 1.5 * qrange
        qmin = q1 - 1.5 * qrange

        for i in range(length):
            if (b[i] < qmin) | (b[i] > qmax):
                if i==0:
                    b[i] = b[i + 1]
                elif i == length-1:
                    b[i] = b[i - 1]
                else:
                    b[i] = (b[i-1] + b[i+1]) / 2

        for i in range(length):
            if (b[i] < f0[i] * (2 - sence)) | (f0[i] * (2 + sence) < b[i]):
                f0[i] = 0

        self.fundamental_frequency = pg.PlotCurveItem(
                np.linspace(0, self.wave_period, f0.shape[0]),
                f0 * 1.8 / (f0.max()-f0.min()) - 0.9,
                pen=pg.mkPen(color="r"),
                antialias=True
            )
        self.sw. addItem(self.fundamental_frequency)



        # self.test = pg.PlotCurveItem(
        #         np.linspace(0, self.wave_period, b.shape[0]),
        #         b * 2 / (b.max()-b.min()) - 1,
        #         pen=pg.mkPen(color="g"),
        #         antialias=True
        #     )
        # self.sw.addItem(self.test)

        cutoff = 12
        window = np.hamming(N)
        trimmer2 = np.hstack((np.ones(cutoff), np.zeros(N - cutoff)))

        c = np.zeros(length)
        for i in range(length-1):
            frame = self.y[t0:t0+N] * window
            spec = np.log(np.abs(fftpack.fft(frame).real))
            cept = fftpack.ifft(trimmer2 * fftpack.fft(spec)).real
            result = np.zeros(5)
            result[0] = np.sum(np.log(self.dispersion_a) + np.power((cept - self.avarage_a), 2) / (2 * np.power(self.dispersion_a, 2)))
            result[1] = np.sum(np.log(self.dispersion_b) + np.power((cept - self.avarage_b), 2) / (2 * np.power(self.dispersion_b, 2)))
            result[2] = np.sum(np.log(self.dispersion_c) + np.power((cept - self.avarage_c), 2) / (2 * np.power(self.dispersion_c, 2)))
            result[3] = np.sum(np.log(self.dispersion_d) + np.power((cept - self.avarage_d), 2) / (2 * np.power(self.dispersion_d, 2)))
            result[4] = np.sum(np.log(self.dispersion_e) + np.power((cept - self.avarage_e), 2) / (2 * np.power(self.dispersion_e, 2)))
            c[i] = np.argmax(result)

        self.aiueo = pg.PlotCurveItem(
                np.linspace(0, self.wave_period, c.shape[0]),
                c * 1.8 / (c.max() - c.min() + 1e-7) - 0.9,
                pen=pg.mkPen(color="g"),
                antialias=True
            )
        self.sw.addItem(self.aiueo)

        start = self.x[0]
        stop = self.x[-1]
        self.selection_rect = pg.QtGui.QGraphicsRectItem(start, -1, stop-start, 2)
        self.selection_rect.setPen(pg.mkPen((1, 1, 1, 100)))
        self.selection_rect.setBrush(pg.mkBrush((1, 1, 1, 50)))
        self.sw.addItem(self.selection_rect)

        self.sw.setMouseTracking(True)
        self.sw.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.sw.scene().sigMouseMoved.connect(self.mouse_moved)

        self.sw.setXRange(0,self.wave_period,0)
        self.sw.setYRange(-1,1,0)

    def mouse_clicked(self, evt):
        pos = evt.pos()
        if self.sw.sceneBoundingRect().contains(pos):
            start = self.x[self.selected_start_frame]
            stop = self.x[self.selected_stop_frame]
            if self.is_left_tracking:
                self.is_left_tracking = False
                self.cul_spectrum()
                self.cul_cepstrum()
                self.update_spectrum_cepstrum_window()

            elif start - self.sensitivity < self.pos and self.pos < start + self.sensitivity:
                self.is_left_tracking = True
            elif self.is_right_tracking:
                self.is_right_tracking = False
                self.cul_spectrum()
                self.cul_cepstrum()
                self.update_spectrum_cepstrum_window()

            elif stop - self.sensitivity < self.pos and self.pos < stop + self.sensitivity:
                self.is_right_tracking = True
        return

    def mouse_moved(self, evt):
        pos = evt
        if self.sw.sceneBoundingRect().contains(pos):
            self.pos = np.min(np.max(self.sw.plotItem.vb.mapSceneToView(pos).x(), 0))
            if self.is_left_tracking:
                tmp = int(self.pos * self.sampling_rate)
                tmp2 = self.selected_stop_frame - 1
                self.selected_start_frame = min(tmp, tmp2)
                self.update_detail_window()

            elif self.is_right_tracking:
                tmp = int(self.pos * self.sampling_rate)
                tmp2 = self.selected_start_frame + 1
                self.selected_stop_frame = max(tmp, tmp2)
                self.update_detail_window()

        return

    def update_detail_window(self):
        x = self.x[self.selected_start_frame:self.selected_stop_frame]
        y = self.y[self.selected_start_frame:self.selected_stop_frame]
        self.selection_rect.setRect(x[0], -1, x[-1] - x[0], 2)
        self.selected_wave.setData(x, y)
        self.dw.setXRange(x[0], x[-1], 0)
        return

    def init_detail_window(self):
        self.dw = pg.PlotWidget(
            viewBox=pg.ViewBox(
                border=pg.mkPen(color='#000000'),
                invertX=False, invertY=False
            )
        )
        self.dw.setMinimumSize(window_width - margin_width, window_height / item_num)
        self.dw.setMaximumSize(window_width - margin_width, window_height / item_num)
        # self.setCentralWidget(pw)
        self.dw.setBackground("#FFFFFF00")

        self.selected_wave = pg.PlotCurveItem(
            self.x[self.selected_start_frame:self.selected_stop_frame],
            self.y[self.selected_start_frame:self.selected_stop_frame],
            pen=pg.mkPen(color="b"),
            antialias=True
        )
        self.dw.addItem(self.selected_wave)
        start = self.x[self.selected_start_frame]
        stop = self.x[self.selected_stop_frame]
        self.dw.setXRange(start ,stop, 0)
        self.dw.setYRange(-1,1,0)
        return

    def update_spectrum_cepstrum_window(self):
        self.cul_spectrum()
        self.spectrum_wave.setData(self.spx, self.spy)
        self.cul_cepstrum()
        self.cepstrum_wave.setData(self.cpx, self.cpy)
        return

    def init_spectrum_cepstrum_window(self):
        self.scpw = pg.PlotWidget(
            viewBox=pg.ViewBox(
                border=pg.mkPen(color='#000000'),
                invertX=False, invertY=False
            )
        )
        self.scpw.setMinimumSize(window_width - margin_width, window_height / item_num)
        self.scpw.setMaximumSize(window_width - margin_width, window_height / item_num)
        # self.setCentralWidget(pw)
        self.scpw.setBackground("#FFFFFF00")

        self.spectrum_wave = pg.PlotCurveItem(
            self.spx,
            self.spy,
            pen=pg.mkPen(color="b"),
            antialias=True
        )
        self.scpw.addItem(self.spectrum_wave)

        self.cepstrum_wave = pg.PlotCurveItem(
            self.cpx,
            self.cpy,
            pen=pg.mkPen(color="r"),
            antialias=True
        )
        self.scpw.addItem(self.cepstrum_wave)

        start = 0
        stop = self.sampling_rate/2
        self.scpw.setXRange(start ,stop, 0)
        self.scpw.setYRange(self.spy.min(),self.spy.max(),0)
        return

    def load_wavefile(self, filepath):

        wf = wave.open(filepath, 'rb')
        nchannels, sampwidth, framerate, nframes, comptype, compname = wf.getparams()
        # print("nchannels", nchannels)
        # print("sampwidth(byte)", sampwidth)
        # print("framerate", framerate)
        # print("nframes", nframes)
        # print("sec", nframes / framerate)
        self.channel_num = nchannels
        self.quantifying_byte = sampwidth
        self.sampling_rate = framerate
        self.sample_length = nframes
        self.selected_start_frame = 0
        self.selected_stop_frame = self.sample_length-1
        self.wave_period = self.sample_length / self.sampling_rate
        self.sampling_period = 1 / self.sampling_rate

        self.sensitivity = self.wave_period / 100
        self.is_left_tracking = False
        self.is_right_tracking = False

        # to exchenge data byte to int
        # data must be a multiple of 2 : byte(8) => int16
        buf = wf.readframes(self.sample_length)
        data_length = len(buf)
        if self.quantifying_byte == 2:
            offset = data_length - data_length % 2
            buf = buf[0:offset]
            data = np.frombuffer(buf, dtype='int16')
        elif self.quantifying_byte == 4:
            offset = data_length - data_length % 4
            buf = buf[0:offset]
            data = np.frombuffer(buf, dtype='int32')

        # normalize the  amplitude from -1 to 1
        amp = (2 ** 8) ** self.quantifying_byte / 2
        wavedata = data / amp
        self.wavedata = wavedata
        self.x = np.arange(0, self.wave_period, self.sampling_period)
        self.y = self.wavedata[0::self.channel_num]
        return

    def open_wavefile(self):
        filepath = pg.Qt.QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filepath:
            self.load_wavefile(filepath)
            try:
                self.update_selection_window()
                self.update_detail_window()
                self.cul_spectrum()
                self.cul_cepstrum()
                self.update_spectrum_cepstrum_window()
            except:
                print(sys.exc_info())

        return

    def cul_spectrum(self):
        y = self.y[self.selected_start_frame:self.selected_stop_frame]
        self.spx = fftpack.fftfreq(n=y.size, d=1 / self.sampling_rate)
        self.spy = 10 * np.log10(np.abs(fftpack.fft(y).real))
        return

    def cul_cepstrum(self):
        cutoff = 12
        self.cpx = self.spx
        y = self.spy
        test = fftpack.fft(y)
        trimmer2 = np.hstack((np.ones(cutoff), np.zeros(test.shape[0]-cutoff)))
        test2 = (trimmer2 * test)
        self.cpy = fftpack.ifft(test2).real
        return

    def __plot_image(self, img_array):
        # add item in plotwidget
        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.setMinimumSize(window_width, window_height/item_num)
        imageWidget.setMaximumSize(window_width, window_height/item_num)
        vb = imageWidget.addViewBox()
        imv = pg.ImageItem()
        imv.setImage(img_array.T[::-1].T)
        vb.addItem(imv)
        return imageWidget

    def specrtogram(self):
        x = self.y

        N = 2 ** 10
        overlap = N / 2

        start = overlap * self.sampling_period
        stop = self.wave_period * 2
        step = (N - overlap) * self.sampling_period
        time_ruler = np.arange(start, stop/2, step)

        window = np.hamming(N)

        spec = np.zeros([len(time_ruler), 1 + int(N / 2)])
        pos = 0

        for fft_index in range(len(time_ruler)):
            frame = x[pos:pos + N]
            if len(frame) == N:
                windowed = window * frame
                fft_result = np.fft.rfft(windowed)
                fft_data = np.log(np.abs(fft_result))
                for i in range(len(spec[fft_index])):
                    spec[fft_index][-i - 1] = fft_data[i]
                pos += int(N - overlap)
        return spec

    def speech_recognition(self):
        self.load_wavefile('../media/continuous_ch2.wav')


        # set
        N = 2 ** 9
        overlap = int(N / 2)
        stride = self.y.strides[0]
        window = np.hamming(N)
        cutoff = 12
        period = int(self.sample_length / 5)
        length = int(period / overlap - 1)
        trimmer2 = np.hstack((np.ones(cutoff), np.zeros(N-cutoff)))

        a = self.y[period * 0:period * 1]
        b = self.y[period * 1:period * 2]
        c = self.y[period * 2:period * 3]
        d = self.y[period * 3:period * 4]
        e = self.y[period * 4:period * 5]

        tmp = as_strided(a, (length, N), (stride * overlap, stride))
        frames_a = tmp * window
        tmp = as_strided(b, (length, N), (stride * overlap, stride))
        frames_b = tmp * window
        tmp = as_strided(c, (length, N), (stride * overlap, stride))
        frames_c = tmp * window
        tmp = as_strided(d, (length, N), (stride * overlap, stride))
        frames_d = tmp * window
        tmp = as_strided(e, (length, N), (stride * overlap, stride))
        frames_e = tmp * window

        spec_a = np.log(np.abs(fftpack.fft(frames_a).real))
        spec_b = np.log(np.abs(fftpack.fft(frames_b).real))
        spec_c = np.log(np.abs(fftpack.fft(frames_c).real))
        spec_d = np.log(np.abs(fftpack.fft(frames_d).real))
        spec_e = np.log(np.abs(fftpack.fft(frames_e).real))

        cept_a = fftpack.ifft(trimmer2 * fftpack.fft(spec_a)).real
        cept_b = fftpack.ifft(trimmer2 * fftpack.fft(spec_b)).real
        cept_c = fftpack.ifft(trimmer2 * fftpack.fft(spec_c)).real
        cept_d = fftpack.ifft(trimmer2 * fftpack.fft(spec_d)).real
        cept_e = fftpack.ifft(trimmer2 * fftpack.fft(spec_e)).real

        self.avarage_a = np.sum(cept_a) / cept_a.shape[0]
        self.avarage_b = np.sum(cept_b) / cept_b.shape[0]
        self.avarage_c = np.sum(cept_c) / cept_c.shape[0]
        self.avarage_d = np.sum(cept_d) / cept_d.shape[0]
        self.avarage_e = np.sum(cept_e) / cept_e.shape[0]

        self.dispersion_a = np.sum(np.power((cept_a - self.avarage_a),2)) / a.shape[0]
        self.dispersion_b = np.sum(np.power((cept_b - self.avarage_b),2)) / b.shape[0]
        self.dispersion_c = np.sum(np.power((cept_c - self.avarage_c),2)) / c.shape[0]
        self.dispersion_d = np.sum(np.power((cept_d - self.avarage_d),2)) / d.shape[0]
        self.dispersion_e = np.sum(np.power((cept_e - self.avarage_e),2)) / e.shape[0]

        return

########
# main #
########
if __name__ == '__main__':
    app = pg.QtGui.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())



