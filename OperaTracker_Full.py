import os
os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import time

from PyQt5 import QtGui, QtCore, QtWidgets#, QtMultimediaWidgets

import librosa
import madmom
import numpy as np
from scipy.fftpack import dct
import torch

import dataExtraction2



class AlignmentThread(QtCore.QThread):

    position = QtCore.pyqtSignal(int)

    def __init__(self):
        super(AlignmentThread, self).__init__()

        # Stream information
        self.sr = 44100
        self.num_channels = 1
        self.frame_size_stream = int(0.02*self.sr) # 20ms
        self.hop_size_stream = 0.01*self.sr # 10ms
        self.audio_stream = madmom.audio.signal.Stream(sample_rate=self.sr, num_channels=self.num_channels, frame_size=self.frame_size_stream, hop_size=self.hop_size_stream)

        # Spectrogram information
        self.window = np.hamming(self.frame_size_stream+1)[:-1]
        self.zeroPad = 2**0
        self.fft_size = int(pow(2, np.round(np.log(self.frame_size_stream * self.zeroPad)/np.log(2))))
        self.spec = np.zeros(int(self.fft_size/2), dtype=np.complex)

        # Filter matrix for MFCC
        self.num_bands = 120
        self.skip = 20
        self.matMFCC = librosa.filters.mel(sr=self.sr, n_fft=self.fft_size-1, n_mels=self.num_bands, fmin=0, fmax=self.sr/2, norm=1)

        # Applause detector (400ms)
        self.X_applause = np.zeros(40)

        # Timers
        self.timer_align = QtCore.QTimer()
        self.timer_align.timeout.connect(self.start)

    def load_alignment(self, Y, lengths_Y, Y_LR, lengths_Y_LR):
        self.Y = Y
        self.lengths_Y = lengths_Y
        self.Y_LR =Y_LR
        self.lengths_Y_LR = lengths_Y_LR

        # DTW init
        self.dtw = dataExtraction2.audio2audio_alignment(Y=self.Y, lengths_Y=self.lengths_Y, Y_LR=self.Y_LR, lengths_Y_LR=self.lengths_Y_LR)

        # Streams init
        self.audio_stream = madmom.audio.signal.Stream(sample_rate=self.sr, num_channels=self.num_channels, frame_size=self.frame_size_stream, hop_size=self.hop_size_stream)

        # Timer
        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        # Iniialization
        self.index_bar = 0
        self.index_page = 0

    def extract_mfcc(self, frame):
        # Compute spec
        self.spec = madmom.audio.stft.stft(frame, window=self.window, fft_size=self.fft_size)
        self.spec = abs(self.spec)
        # Normalization
        self.spec -= np.min(self.spec)
        if np.max(self.spec)!=0:
            self.spec /= np.max(self.spec)
        # Get mfcc
        mel_spec = np.dot(self.spec, self.matMFCC.T)
        mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, self.skip:]
        # Normalization
        if np.linalg.norm(mfcc) == 0:
            mfcc = np.ones(mfcc.shape[1]) * 1e-10
        mfcc = mfcc / np.linalg.norm(mfcc)
        mfcc = mfcc[0, :]
        return mfcc

    def run(self):
        # Update data
        nextone = self.audio_stream.next()
        nextone = np.expand_dims(nextone, axis=0)

        if np.linalg.norm(nextone, ord=2) > 0.01:
            # Current song
            self.idx_current_song = np.argmin(np.abs(self.dtw.actual_position-self.lengths_Y))

            # Applause Detector
            global applause_output
            self.X_applause = np.concatenate((self.X_applause[1:], [applause_output]))

            # if applause activated
            if self.dtw.actual_position > 1000 and 0 <= np.abs(self.dtw.actual_position-self.lengths_Y[self.idx_current_song]) < 100 and (self.X_applause > 0.75).all():
                self.dtw.actual_position = self.lengths_Y[self.idx_current_song]
                self.dtw.gamma[0, :] = self.dtw.gamma_save_applause
            # else: alignment
            else:
                # Compute MFCC
                self.dtw_input = self.extract_mfcc(nextone)
                # Compute JOLTW
                # self.dtw.local_OLTW(self.dtw_input)
                # self.dtw.local_JOLTW(self.dtw_input)
                self.dtw.local_JOLTWLR(self.dtw_input)
                # Saving previous gamma for detectors
                if applause_output < 0.75:
                    self.dtw.gamma_save_applause = self.dtw.gamma[0, :]


        # print(self.dtw.actual_position)
        self.position.emit(self.dtw.actual_position)

        # Update time
        # now = time.time()
        # dt = (now-self.lastupdate)
        # if dt <= 0:
        #     dt = 0.000000000001
        # fps2 = 1.0 / dt
        # self.lastupdate = now
        # self.fps = self.fps * 0.9 + fps2 * 0.1
        # tx = 'Mean Frame Rate Alignment:  {fps:.3f} FPS'.format(fps=self.fps)
        # print(tx, self.dtw.actual_position, "from", int(self.thread().currentThreadId()))
        self.counter += 1



class ApplauseThread(QtCore.QThread):

    def __init__(self):
        super(ApplauseThread, self).__init__()

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
        self.spec = np.zeros(int(self.fft_size/2), dtype=np.complex)

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
        self.spec_mfcc = np.zeros(int(self.fft_size/2), dtype=np.complex)
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

        # Timers
        self.timer_align = QtCore.QTimer()
        self.timer_align.timeout.connect(self.start)

        # Timer
        self.fps = 0.
        self.lastupdate = time.time()

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


        global applause_output
        applause_output = self.applause_prediction

        # Check thread
        # print('Applause:' applause_output, 'from', int(self.thread().currentThreadId()))


# https://discuss.python.org/t/if-mouse-button-event-draw-rectangle-pyqt5/6064
class ImageWidget(QtWidgets.QWidget):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        # self.setStyleSheet("background-color: rgb(255,0,0); margin:5px; border:1px solid rgb(0, 255, 0); ")
        self.qlabel = QtWidgets.QLabel(self)
        self.qlabel.setMinimumSize(600, 800)
        self.qlabel.setMinimumSize(600, 800)
        self.qlabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.qlabel)

        # Color for quadrilateral
        self.brushRectangle = QtGui.QBrush(QtGui.QColor(128, 128, 255, 128))

        # Load initial picture
        self.image = QtGui.QImage('./scores/cover.jpg').scaled(600, 800, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        self.points = [[QtCore.QPoint(0,0), QtCore.QPoint(0,0), QtCore.QPoint(0,0), QtCore.QPoint(0,0)]]
        self.index_bar = 0
        self.position=0

    def load_pictures(self, time_new_bar, time_new_page, points, pics):
        self.time_new_bar = time_new_bar
        self.time_new_page = time_new_page
        self.points = points
        self.pics = pics

        # Variables for plotting
        self.position = 0
        self.index_page = 0
        self.index_bar = 0

        # Load initial picture
        self.image = self.pics[self.index_page]
        self.update()

    def pictureUpdate(self, position):
        self.position = position
        if (self.index_page < len(self.time_new_page)-1) and not (self.time_new_page[self.index_page] <= self.position < self.time_new_page[self.index_page+1]):
            self.index_page = np.searchsorted(self.time_new_page, self.position, side='right')-1
            self.image = self.pics[self.index_page]
        self.update()

    def areaUpdate(self, position):
        self.position = position
        if (self.index_bar < len(self.time_new_bar)-1) and not (self.time_new_bar[self.index_bar] <= self.position < self.time_new_bar[self.index_bar+1]):
            self.index_bar = np.searchsorted(self.time_new_bar, self.position, side='right')-1
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        painter.setBrush(self.brushRectangle)

        if len(self.points[self.index_bar]) != 2:
            poly = QtGui.QPolygon(self.points[self.index_bar])
            painter.drawPolygon(poly)

        else: # plotting weird bars
            poly1 = QtGui.QPolygon(self.points[self.index_bar][0])
            poly2 = QtGui.QPolygon(self.points[self.index_bar][1])
            painter.drawPolygon(poly1)
            painter.drawPolygon(poly2)

        painter.end()


class SubtitleWidget(QtWidgets.QWidget):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setStyleSheet("background-color: white; border:1px solid black; ")
        self.qlabel = QtWidgets.QLabel(self)
        self.qlabel.setFont(QtGui.QFont('Arial', 15))
        self.qlabel.setMinimumSize(600, 100)
        self.qlabel.setMaximumSize(600, 100)
        self.qlabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.qlabel)

        # Initializing text
        self.qlabel.setText('Select an act, a subtitle language, and click on Start.')

        # Init subtitling variables
        self.list_languages = dataExtraction2.list_languages()
        self.idx_language = self.list_languages.index('English')

    def load_subtitles(self, subtitles_labels, subtitles_times):
        self.subtitles_labels = subtitles_labels
        self.subtitles_times = subtitles_times

        # Variable for plotting
        self.position = 0
        self.index_subtitle = 0

        # Load initial subtitle
        self.qlabel.setText(self.subtitles_labels[self.idx_language][self.index_subtitle])


    def subtitleUpdate(self, position):
        self.position = position
        if (self.index_subtitle < len(self.subtitles_times)-1) and not (self.subtitles_times[self.index_subtitle] <= self.position < self.subtitles_times[self.index_subtitle+1]):
            self.index_subtitle = np.searchsorted(self.subtitles_times, self.position, side='right')-1
        self.qlabel.setText(self.subtitles_labels[self.idx_language][self.index_subtitle])
        self.update()

    def languageUpdate(self, new_language):
        self.new_language = new_language
        self.new_idx_language = self.list_languages.index(self.new_language)
        if self.new_idx_language != self.idx_language:
            self.idx_language = self.new_idx_language
        self.update()



class ComboBox_Parts(QtWidgets.QComboBox):

    def __init__(self):
        QtWidgets.QComboBox.__init__(self)
        self.setStyleSheet("min-width: 370px;")

        # List of operas
        self.addItem('Don Giovanni - Act 1')
        self.addItem('Don Giovanni - Act 2')


class ComboBox_Languages(QtWidgets.QComboBox):

    def __init__(self, subtitle):
        QtWidgets.QComboBox.__init__(self)

        # List of languages
        self.data_languages = dataExtraction2.list_languages()
        for i in range(len(self.data_languages)):
            self.addItem(self.data_languages[i])

        # Connect selected language
        self.currentIndexChanged.connect(self.update)

        # Add class
        self.subtitle = subtitle

    def update(self):
        self.subtitle.languageUpdate(self.currentText())


class PushButton(QtWidgets.QPushButton):

    def __init__(self, menu, score, language, subtitle, align, applause):
        QtWidgets.QComboBox.__init__(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.setSizePolicy(sizePolicy)
        self.setText('Start')

        # Connect click button
        self.clicked.connect(self.update)

        # Add classes
        self.menu = menu
        self.score = score
        self.language = language
        self.subtitle = subtitle
        self.align = align
        self.applause = applause

    def update(self):
        if self.text() == 'Start':
            # Load data
            print('Loading data...')
            self.data = dataExtraction2.data_loading(target=self.menu.currentText(), feature='MFCC')

            # Prepare alignment
            print('Preparing alignment...')
            self.align.load_alignment(Y=self.data.Y, lengths_Y=self.data.lengths_Y, Y_LR=self.data.Y_LR, lengths_Y_LR=self.data.lengths_Y_LR)

            # Preparing PDF scores
            print('Preparing PDF scores...')
            self.score.load_pictures(time_new_bar=self.data.time_new_bar, time_new_page=self.data.time_new_page, points=self.data.points, pics=self.data.pics)
            
            # Preparing subtitles
            print('Preparing subtitles...')
            self.subtitle.load_subtitles(subtitles_labels=self.data.subtitles_labels, subtitles_times=self.data.subtitles_times)

            # Clear buffer
            print('Clear buffer....')
            while np.linalg.norm(self.align.audio_stream.next(), ord=2) > 0.01: # when using internal sound
            # while np.linalg.norm(self.align.audio_stream.next(), ord=2) > 1: # when using microphone (ambiant noise)
                time.sleep(0.01)

            print('Ready!')
            self.align.timer_align.start()
            self.applause.timer_align.start()
            self.setText('Stop')

        else:
            self.align.timer_align.stop()
            self.applause.timer_align.stop()
            self.setText('Start')


def main():

    app = QtWidgets.QApplication([])
    # print("Main application thread is : ", int(app.thread().currentThreadId()))
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setWindowTitle("Opera Tracker")
    MainWindow.resize(600, 1000) # Size of the principal window
    MainWindow.move(737, 0)
    centralWidget = QtWidgets.QWidget(MainWindow)

    # Vertical layout to divide Score and Buttons
    verticalLayout = QtWidgets.QVBoxLayout(centralWidget)

    # Add image widget
    score = ImageWidget()
    verticalLayout.addWidget(score)

    # Add subtitle widget
    subtitle = SubtitleWidget()
    verticalLayout.addWidget(subtitle)

    # Add align thread
    align = AlignmentThread()
    # Connect dtw position to other widgets
    align.position.connect(score.pictureUpdate)
    align.position.connect(score.areaUpdate)
    align.position.connect(subtitle.subtitleUpdate)

    # Add applause thread
    applause = ApplauseThread()
    # Global value for applause
    applause_output = 0

    # Add dropdown button with part names, languages, and start click
    menu_horizontalLayout = QtWidgets.QHBoxLayout()

    # Drop menu with part names
    menu = ComboBox_Parts()
    menu_horizontalLayout.addWidget(menu)

    # Drop menu with languages
    language = ComboBox_Languages(subtitle=subtitle)
    menu_horizontalLayout.addWidget(language)

    # Start/Stop button
    push = PushButton(menu=menu, score=score, language=language, subtitle=subtitle, align=align, applause=applause)
    menu_horizontalLayout.addWidget(push)

    verticalLayout.addLayout(menu_horizontalLayout)

    MainWindow.setCentralWidget(centralWidget)
    MainWindow.show()
    app.exec_()                  




if __name__ == '__main__':
    main()



