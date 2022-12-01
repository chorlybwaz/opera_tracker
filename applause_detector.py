import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import librosa
import madmom
from scipy.fftpack import dct
import torch
import time

import scipy


class ApplauseThread(QtCore.QThread):

    estimation = QtCore.pyqtSignal(float)

    def __init__(self):
        QtCore.QThread.__init__(self)

        # Stream information
        self.sr = 44100
        self.num_channels = 1
        self.frame_size_stream = int(0.100*self.sr)
        self.hop_size_stream = int(0.02*self.sr)
        self.audio_stream = madmom.audio.signal.Stream(sample_rate=self.sr, num_channels=self.num_channels, frame_size=self.frame_size_stream, hop_size=self.hop_size_stream)
        self.nextone = np.zeros(self.frame_size_stream)
        self.nextone_prev = np.zeros(self.frame_size_stream)

        # Spectrogram information
        self.window = np.hanning(self.frame_size_stream + 1)[:-1]
        self.zeroPad = 2**0
        self.fft_size = int(pow(2, np.round(np.log(self.frame_size_stream * self.zeroPad)/np.log(2))))
        self.spec = np.zeros(int(self.fft_size/2), dtype=complex)

        # Sub-bands info
        self.spec_frequencies = np.fft.fftfreq(self.spec.shape[0] * 2, 1. / self.sr)[:self.spec.shape[0]]
        self.spec1_start = self.idx_nearest(self.spec_frequencies, 129)
        self.spec2_start = self.idx_nearest(self.spec_frequencies, 387)
        self.spec3_start = self.idx_nearest(self.spec_frequencies, 926)
        self.spec4_start = self.idx_nearest(self.spec_frequencies, 2003)
        self.spec4_end = self.idx_nearest(self.spec_frequencies, 4134)
        # Compute centered frequencies
        self.spec1_bin_frequencies = self.spec_frequencies[self.spec1_start:self.spec2_start]
        self.spec1_centered_frequencies = self.spec1_bin_frequencies + (self.spec1_bin_frequencies[1] - self.spec1_bin_frequencies[0]) / 2
        self.spec1 = np.zeros(self.spec2_start-self.spec1_start)
        self.spec2_bin_frequencies = self.spec_frequencies[self.spec2_start:self.spec3_start]
        self.spec2_centered_frequencies = self.spec2_bin_frequencies + (self.spec2_bin_frequencies[1] - self.spec2_bin_frequencies[0]) / 2
        self.spec2 = np.zeros(self.spec3_start-self.spec2_start)
        self.spec3_bin_frequencies = self.spec_frequencies[self.spec3_start:self.spec4_start]
        self.spec3_centered_frequencies = self.spec3_bin_frequencies + (self.spec3_bin_frequencies[1] - self.spec3_bin_frequencies[0]) / 2
        self.spec3 = np.zeros(self.spec4_start-self.spec3_start)
        self.spec4_bin_frequencies = self.spec_frequencies[self.spec4_start:self.spec4_end+1]
        self.spec4_centered_frequencies = self.spec4_bin_frequencies + (self.spec4_bin_frequencies[1] - self.spec4_bin_frequencies[0]) / 2
        self.spec4 = np.zeros(self.spec4_end+1-self.spec4_start)
        self.specs_centered_frequencies = [self.spec1_centered_frequencies, self.spec2_centered_frequencies, self.spec3_centered_frequencies, self.spec4_centered_frequencies]
        self.specs_prev = [np.zeros(self.spec2_start-self.spec1_start),
                            np.zeros(self.spec3_start-self.spec2_start),
                            np.zeros(self.spec4_start-self.spec3_start),
                            np.zeros(self.spec4_end+1-self.spec4_start)]

        # Coefficients info
        self.sc = np.zeros(4)
        self.ssp = np.zeros(4)
        self.sf = np.zeros(4)
        self.sfm = np.zeros(4)

        # MFCC infos
        self.k = 0.97
        self.window_mfcc = np.hamming(self.frame_size_stream + 1)[:-1]
        self.spec_mfcc = np.zeros(int(self.fft_size/2), dtype=complex)
        self.num_bands = 20
        self.matMFCC = librosa.filters.mel(sr=self.sr, n_fft=self.fft_size-1, n_mels=self.num_bands, fmin=0, fmax=self.sr/2, norm=1)
        self.num_ceps = 9

        # Applause model
        self.ApplauseModel_name = './models/ApplauseModel_100.ckpt'
        import torchNet
        self.ApplauseModel = torchNet.LSTM_for_BCELoss_hidden(input_size=25, hidden_size=55, num_layers=1, batch_size=1, output_size=1)
        checkpoint = torch.load(self.ApplauseModel_name)
        self.ApplauseModel.load_state_dict(checkpoint['state_dict'])
        self.ApplauseModel.eval()
        [self.ApplauseModel_mean, self.ApplauseModel_var] = np.load('./models/mean_var_Applause_train.npy')
        self.hidden_state = (torch.zeros(1, 1, 55), torch.zeros(1, 1, 55))
        self.applause_prediction = 0

        # Smoothing
        # self.prediction_prev = np.zeros(55)
        # self.weights = np.exp(np.arange(55)/3) / np.exp(54/3)

        # Timer
        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        self.timer_voice = QtCore.QTimer()
        self.timer_voice.timeout.connect(self.run)
        self.timer_voice.start()

    def idx_nearest(self, array, value):
        return  (np.abs(array - value)).argmin()

    def spectral_centroid(self, frame, freqs):
        if np.sum(frame) == 0:
            return 0
        else:
            num = np.dot(frame, freqs)
            den = np.sum(frame)
            return num / den

    def spectral_spread(self, frame, freqs, sc_i):
        if np.sum(frame) == 0:
            return 0
        else:
            num = np.dot(frame, (freqs-sc_i)**2)
            den = np.sum(frame)
            return np.sqrt(num / den)

    def spectral_flux(self, frame, prev_frame):
        return np.sqrt(np.sum((frame - prev_frame)**2))

    def spectral_flatness_measure(self, frame):
        if np.sum(frame) == 0:
            return 0
        else:
            return np.prod(np.abs(frame)**(1/len(frame))) / ((1/len(frame)) * np.sum(np.abs(frame)))

    def extract_ApplauseFeature(self, frame, frame_prev):
        # Compute spec
        self.spec = madmom.audio.stft.stft(frame, window=self.window, fft_size=self.fft_size)
        self.spec = abs(self.spec) +1e-10
        self.spec = self.spec  / np.linalg.norm(self.spec, axis=1)
        self.spec = self.spec[0]

        # Compute sub-bands
        self.spec1 = self.spec[self.spec1_start:self.spec2_start]
        self.spec2 = self.spec[self.spec2_start:self.spec3_start]
        self.spec3 = self.spec[self.spec3_start:self.spec4_start]
        self.spec4 = self.spec[self.spec4_start:self.spec4_end+1]
        self.specs = [self.spec1, self.spec2, self.spec3, self.spec4]

        # Compute coefficients
        for s in range(4):
            self.sc[s] = self.spectral_centroid(self.specs[s], self.specs_centered_frequencies[s])
            self.ssp[s] = self.spectral_spread(self.specs[s], self.specs_centered_frequencies[s], self.sc[s])
            self.sf[s] = self.spectral_flux(self.specs[s], self.specs_prev[s])
            self.sfm[s] = self.spectral_flatness_measure(self.specs[s])
        self.specs_prev = self.specs

        # Compute MFCC
        self.sig_prime = frame - self.k * frame_prev
        self.spec_mfcc = madmom.audio.stft.stft(self.sig_prime, window=self.window_mfcc, fft_size=self.fft_size)
        self.spec_mfcc = abs(self.spec_mfcc) +1e-10
        self.spec_mfcc = self.spec_mfcc  / np.linalg.norm(self.spec_mfcc, axis=1)
        self.spec_mfcc = self.spec_mfcc
        mel_spec = np.dot(self.spec_mfcc, self.matMFCC.T)
        mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, :self.num_ceps]

        # Compute applause feature
        applause_features = np.hstack((self.sc, self.ssp, self.sf, self.sfm, mfcc[0]))
        # Normalizing over train dataset
        applause_features -= self.ApplauseModel_mean
        applause_features /= self.ApplauseModel_var
        # Transforming into tensor
        applause_input = torch.Tensor(applause_features[None, None, :])
        return applause_input

    def run(self):
        # Update data
        self.nextone = self.audio_stream.next()
        self.nextone = np.expand_dims(self.nextone, axis=0)

        if np.linalg.norm(self.nextone, ord=2) > 0.01:
            self.ApplauseModel_input = self.extract_ApplauseFeature(self.nextone, self.nextone_prev)
            self.ApplauseModel_output, self.hidden_state = self.ApplauseModel(self.ApplauseModel_input, self.hidden_state)
            self.applause_prediction = self.ApplauseModel_output[-1].item()

        self.estimation.emit(self.applause_prediction)

        # Update time
        # now = time.time()
        # dt = (now-self.lastupdate)
        # if dt <= 0:
        #     dt = 0.000000000001
        # fps2 = 1.0 / dt
        # self.lastupdate = now
        # self.fps = self.fps * 0.9 + fps2 * 0.1
        # tx = 'Mean Frame Rate Voice:  {fps:.3f} FPS'.format(fps=self.fps )
        # print(tx, self.applause_prediction, "from", int(self.thread().currentThreadId()))
        self.counter += 1




class ApplauseLabel(QtWidgets.QLabel):

    def __init__(self):
        QtWidgets.QLabel.__init__(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.setSizePolicy(sizePolicy)

        # Add text label
        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setText("   Applause activity   ")



class ApplauseWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Ignored)
        self.setSizePolicy(sizePolicy)
        self.setMaximumSize(QtCore.QSize(16777215, 20))
        self.setAutoFillBackground(True)

        # Add color
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.red)
        self.setPalette(p)

    def paint(self, pred):
        if pred > 0.5:
            p = self.palette()
            p.setColor(self.backgroundRole(), QtCore.Qt.green)
            self.setPalette(p)
        else:
            p = self.palette()
            p.setColor(self.backgroundRole(), QtCore.Qt.red)
            self.setPalette(p)


    


def main():

    app = QtWidgets.QApplication([])
    # print("Main application thread is : ", int(app.thread().currentThreadId()))
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setWindowTitle("Chromagram and sheet music")
    # # MainWindow.resize(50, 200) # Size of the principal window
    centralWidget = QtWidgets.QWidget(MainWindow)

    # Vertical layout to divide Score and Buttons
    verticalLayout = QtWidgets.QVBoxLayout(centralWidget)

    # Add applause widget
    applause_label = ApplauseLabel()
    applause_widget = ApplauseWidget()
    applause_horizontalLayout = QtWidgets.QHBoxLayout()
    applause_horizontalLayout.addWidget(applause_label)
    applause_horizontalLayout.addWidget(applause_widget)
    verticalLayout.addLayout(applause_horizontalLayout)

    # # Add applause thread
    applause = ApplauseThread()
    applause.estimation.connect(applause_widget.paint)

    MainWindow.setCentralWidget(centralWidget)

    MainWindow.show()
    app.exec_()                  




if __name__ == '__main__':
    main()



