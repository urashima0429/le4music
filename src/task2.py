import wave
import pyaudio
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import fftpack
import sys
import threading
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5 import QtMultimedia

# from PyQt5.QtMultimedi import QSound

# constant
window_width = 1000
window_height = 800
margin_width = 30
margin_height = 150

item_num = 4
display_frame_length = 50
karaoke = "C:/Users/ryugu/Dropbox/新しいフォルダー/kowloon.wav"


class MainWindow(pg.QtGui.QMainWindow):

    def __init__(self):
        super().__init__()

        self.mediaPlayer = QtMultimedia.QMediaPlayer(self)
        sound=QtMultimedia.QMediaContent(
            QtCore.QUrl.fromLocalFile(
                karaoke
            )
        )
        self.mediaPlayer.setMedia(sound)


        self.CHUNK = 2 ** 10
        self.sampling_rate = 44100
        self.p2 = pyaudio.PyAudio()
        self.stream2 = self.p2.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.CHUNK)

        self.init_view()

        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(50)
        return

    def init_view(self):
        self.init_main_window()
        self.vbox0 = pg.QtGui.QVBoxLayout()


        # plot waveform
        self.data = np.zeros(self.CHUNK)
        self.pw = pg.PlotWidget(
            viewBox=pg.ViewBox(
                border=pg.mkPen(color='#000000'),
                invertX=False, invertY=False
            )
        )
        self.pw.setMinimumSize(window_width - margin_width, window_height / item_num)
        self.pw.setMaximumSize(window_width - margin_width, window_height / item_num)
        # self.setCentralWidget(pw)
        self.pw.setBackground("#FFFFFF00")
        self.input_wave = pg.PlotCurveItem(
                np.linspace(0, self.CHUNK, self.data.shape[0]),
                self.data,
                pen=pg.mkPen(color="b"),
                antialias=True
            )
        self.pw.addItem(self.input_wave)
        self.vbox0.addWidget(self.pw)


        # plot fondamental
        self.stride = self.data.strides[0]
        N = int(self.CHUNK / 2)
        self.trimmer = np.rot90(np.triu(np.ones((N, N))))

        self.data2 = np.zeros(display_frame_length)
        self.pw2 = pg.PlotWidget(
            viewBox=pg.ViewBox(
                border=pg.mkPen(color='#000000'),
                invertX=False, invertY=False
            )
        )
        self.pw2.setMinimumSize(window_width - margin_width, window_height / item_num)
        self.pw2.setMaximumSize(window_width - margin_width, window_height / item_num)
        # self.setCentralWidget(pw)
        self.pw2.setBackground("#FFFFFF00")
        self.f0_wave = pg.PlotCurveItem(
                np.linspace(0, self.CHUNK, self.data2.shape[0]),
                self.data2,
                pen=pg.mkPen(color="b"),
                antialias=True
            )
        self.pw2.addItem(self.f0_wave)
        self.vbox0.addWidget(self.pw2)

        # plot spec
        self.window = np.hamming(self.CHUNK)
        self.spec = np.zeros((display_frame_length,self.CHUNK))
        self.iw = pg.GraphicsLayoutWidget()
        self.iw.setMinimumSize(window_width, window_height / item_num * 2)
        self.iw.setMaximumSize(window_width, window_height / item_num * 2)
        self.vb = self.iw.addViewBox()
        self.imv = pg.ImageItem()
        self.imv.setImage(self.spec)
        self.vb.addItem(self.imv)
        self.vbox0.addWidget(self.iw)

        centralWid = pg.QtGui.QWidget()
        centralWid.setLayout(self.vbox0)
        self.setCentralWidget(centralWid)

        self.mediaPlayer.play()
        return

    def update_view(self):
        # load the most recent data
        mrd = np.frombuffer(self.stream2.read(self.CHUNK), dtype="int16") / 32768.0 * 2
        self.update_waveform(mrd)
        self.update_f0(mrd)
        self.update_spectrogram(mrd)

    def update_waveform(self, data):
        self.data = np.append(self.data, data)
        if len(self.data) / self.CHUNK > display_frame_length:
            self.data = self.data[self.CHUNK:]
        self.input_wave.setData(np.linspace(0, self.CHUNK, self.data.shape[0]), self.data)
        self.pw.setYRange(-1, 1)
        return

    def update_f0(self, data):
        N = int(self.CHUNK / 2)
        if len(self.data2) > display_frame_length:
            self.data2 = self.data2[1:]


        tmp = data[N:]
        Xt = as_strided(tmp, (N, N), (0, self.stride))
        Xt_r = as_strided(tmp, (N, N), (-self.stride, self.stride))
        ac = np.sum(Xt * Xt_r * self.trimmer, axis=1)
        dydx = ac[1::] - ac[:-1:]

        tmp2 = np.where((dydx[:-1:] >= 0) & (dydx[1::] < 0))[0]
        f0 = self.sampling_rate / (tmp2[0] + 1)

        # check amplitude
        if tmp.max() - tmp.min() < 0.5:
            f0 = 0

        # check 0 kousa
        a = data[:-1:] * data[1::] < 0
        b = np.sum(a) * self.sampling_rate / self.CHUNK / 2
        sence = 1
        if ((2 - sence) * f0 < b) & (b < (2 + sence) * f0):
            f0 = 0

        self.data2 = np.append(self.data2, f0)
        self.f0_wave.setData(np.linspace(0, self.CHUNK, self.data2.shape[0]), self.data2)
        self.pw2.setYRange(0, 5000)
        return

    def update_spectrogram(self, data):
        fft_data = np.log(np.abs(fftpack.fft(self.window * data).real))

        self.spec = np.append(self.spec, [fft_data], axis=0)

        if self.spec.shape[0] > display_frame_length:
            self.spec = self.spec[1:]
        self.imv.setImage(self.spec)
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

        # exit
        startStopAction = pg.QtGui.QAction(pg.QtGui.QIcon('../icon/st.png'), '&start/stop', self)
        startStopAction.setShortcut('Space')
        startStopAction.setStatusTip('start/stop')
        # startStopAction.triggered.connect()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        menubar.show()

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAction)
        self.toolbar.addAction(startStopAction)
        return

########
# main #
########
if __name__ == '__main__':
    app = pg.QtGui.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())



