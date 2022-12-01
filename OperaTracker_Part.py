import os
os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import time

from PyQt5 import QtGui, QtCore, QtWidgets

import librosa
import numpy as np
import madmom
import pandas as pd
from scipy.fftpack import dct
import torch
import yaml

import dataExtraction



class AlignmentThread(QtCore.QThread):

    position = QtCore.pyqtSignal(int)

    def __init__(self):
        super(AlignmentThread, self).__init__()

        ########
        # MFCC #
        ########
        # Stream information
        self.MFCC_sr = 44100
        self.MFCC_num_channels = 1
        self.MFCC_frame_size_stream = int(0.02*self.MFCC_sr) # 20ms
        self.MFCC_hop_size_stream = 0.01*self.MFCC_sr # 10ms
        self.MFCC_audio_stream = madmom.audio.signal.Stream(sample_rate=self.MFCC_sr, num_channels=self.MFCC_num_channels, frame_size=self.MFCC_frame_size_stream, hop_size=self.MFCC_hop_size_stream)

        # Spectrogram information
        self.MFCC_window = np.hamming(self.MFCC_frame_size_stream+1)[:-1]
        self.MFCC_zeroPad = 2**0
        self.MFCC_fft_size = int(pow(2, np.round(np.log(self.MFCC_frame_size_stream * self.MFCC_zeroPad)/np.log(2))))
        self.MFCC_spec = np.zeros(int(self.MFCC_fft_size/2), dtype=complex)

        # Filter matrix for MFCC
        self.MFCC_num_bands = 120
        self.MFCC_skip = 20
        self.MFCC_matMFCC = librosa.filters.mel(sr=self.MFCC_sr, n_fft=self.MFCC_fft_size-1, n_mels=self.MFCC_num_bands, fmin=0, fmax=self.MFCC_sr/2, norm=1)

        ################
        # Posteriogram #
        ################
        # Stream information
        self.POST_sr = 16000
        self.POST_num_channels = 1
        self.POST_frame_size_stream = int(0.02*self.POST_sr) # 20ms
        self.POST_hop_size_stream = 0.01*self.POST_sr # 10ms
        self.POST_audio_stream = madmom.audio.signal.Stream(sample_rate=self.POST_sr, num_channels=self.POST_num_channels, frame_size=self.POST_frame_size_stream, hop_size=self.POST_hop_size_stream)

        # Spectrogram information
        self.POST_window = np.hamming(self.POST_frame_size_stream+1)[:-1]
        self.POST_zeroPad = 2**4
        self.POST_fft_size = int(pow(2, np.round(np.log(self.POST_frame_size_stream * self.POST_zeroPad)/np.log(2))))
        self.POST_spec = np.zeros(int(self.POST_fft_size/2), dtype=complex)

        # Filter matrix for MFCC
        self.POST_num_bands = 80
        self.POST_matMFCC = librosa.filters.mel(sr=self.POST_sr, n_fft=self.POST_fft_size-1, n_mels=self.POST_num_bands, fmin=0, fmax=self.POST_sr/2, norm=1)

        # Posteriogram information
        self.POST_which_language = '5lang'
        self.POST_LyricsModel_name = './models/NoVocalPhon5lang_CPResnet_rhot6_rhof8_c64_b420traced.pt'
        self.POST_LyricsModel = torch.jit.load(self.POST_LyricsModel_name)
        [self.POST_LyricsModel_mean, self.POST_LyricsModel_var] = np.load('./models/mean_var_DALI_train_5langphon.npy')
        self.POST_LyricsModel_input = torch.zeros(1, 1, self.POST_num_bands, 57)
        self.POST_counter_every4 = 0

        # Timers
        self.timer_align = QtCore.QTimer()
        self.timer_align.timeout.connect(self.start)

    def load_song(self, target_printed='Don Giovanni: 1.01 Ouvertura', target_feature='MFCC'):
        self.target_printed = target_printed
        self.data_names = dataExtraction.list_names2areas()
        for i in range(len(self.data_names.printed_names)):
            if self.data_names.printed_names[i] == target_printed:
                self.target = self.data_names.names[i]

        # DTW init
        self.target_feature = target_feature
        self.dtw = dataExtraction.audio2audio_alignment(target=self.target, feature=self.target_feature)

        # Streams init
        self.MFCC_audio_stream = madmom.audio.signal.Stream(sample_rate=self.MFCC_sr, num_channels=self.MFCC_num_channels, frame_size=self.MFCC_frame_size_stream, hop_size=self.MFCC_hop_size_stream)
        self.POST_audio_stream = madmom.audio.signal.Stream(sample_rate=self.POST_sr, num_channels=self.POST_num_channels, frame_size=self.POST_frame_size_stream, hop_size=self.POST_hop_size_stream)

        # Timer
        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        # Iniialization
        self.index_bar = 0
        self.index_page = 0

    def extract_mfcc(self, frame):

        # Compute spec
        self.MFCC_spec = madmom.audio.stft.stft(frame, window=self.MFCC_window, fft_size=self.MFCC_fft_size)
        self.MFCC_spec = abs(self.MFCC_spec)
        # Normalization
        self.MFCC_spec -= np.min(self.MFCC_spec)
        if np.max(self.MFCC_spec)!=0:
            self.MFCC_spec /= np.max(self.MFCC_spec)
        # Get mfcc
        mel_spec = np.dot(self.MFCC_spec, self.MFCC_matMFCC.T)
        mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, self.MFCC_skip:]
        # Normalization
        if np.linalg.norm(mfcc) == 0:
            mfcc = np.ones(mfcc.shape[1]) * 1e-10
        mfcc = mfcc / np.linalg.norm(mfcc)
        mfcc = mfcc[0, :]
        return mfcc

    def extract_InputToModel(self, frame):
        # Compute spec
        self.POST_spec = madmom.audio.stft.stft(frame, window=self.POST_window, fft_size=self.POST_fft_size)
        self.POST_spec = abs(self.POST_spec)
        # Compute mel spec
        mel_spec = np.dot(self.POST_spec, self.POST_matMFCC.T)
        # Normalizing over train dataset
        mel_spec -= self.POST_LyricsModel_mean
        mel_spec /= self.POST_LyricsModel_var
        # Transforming into tensor
        lyrics_tensor = torch.Tensor(mel_spec[None, None, :, :]).transpose(2, 3)
        # Updating our input to the model
        LyricsModel_input = torch.cat((self.POST_LyricsModel_input[:, :, :, 1:], lyrics_tensor), dim=3)
        return LyricsModel_input

    def extract_posteriogram(self, input_feature):
        output_posteriogram = self.POST_LyricsModel(input_feature).detach().numpy()
        output_posteriogram = output_posteriogram[round(len(output_posteriogram)/2), 0, :]
        output_posteriogram = np.exp(output_posteriogram[:-1]) +1e-10
        return output_posteriogram


    def run(self):
        # Update data
        if self.target_feature == 'MFCC':
            nextone = self.MFCC_audio_stream.next()
            nextone = np.expand_dims(nextone, axis=0)
            if np.linalg.norm(nextone, ord=2) > 0.01:
                # Compute MFCC
                self.dtw_input = self.extract_mfcc(nextone)
                # Compute OLTW
                self.dtw.local_DTW(self.dtw_input)
                # Emit actual position
                self.position.emit(self.dtw.actual_position)

        elif self.target_feature == 'Posteriogram':
            nextone = self.POST_audio_stream.next()
            nextone = np.expand_dims(nextone, axis=0)
            if np.linalg.norm(nextone, ord=2) > 0.01:
                # Compute input to model
                self.POST_LyricsModel_input = self.extract_InputToModel(nextone)
                if self.POST_counter_every4 % 4 == 0:
                    # Compute Posteriogram
                    self.dtw_input = self.extract_posteriogram(self.POST_LyricsModel_input)
                    # Compute OLTW
                    self.dtw.local_DTW(self.dtw_input)
                self.POST_counter_every4 += 1
                # Emit actual position
                self.position.emit(self.dtw.actual_position*4)


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


    def load_pictures(self, target_printed='Don Giovanni: 1.01 Ouvertura'):
        self.target_printed = target_printed
        # List of names
        self.data_names = dataExtraction.list_names2areas()
        for i in range(len(self.data_names.printed_names)):
            if self.data_names.printed_names[i] == self.target_printed:
                self.target = self.data_names.names[i]

        # List PDF Score pages of the target
        self.pdf_scores = []
        for file in os.listdir('./scores/DonGiovanni/'):
            if self.target + '-' in file:
                self.pdf_scores.append('./scores/DonGiovanni/' + file)
        self.pdf_scores.sort()

        # Load pictures and extract sizes
        self.pics = []
        self.sizes = []
        for score in self.pdf_scores:
            pic = QtGui.QImage(score).scaled(600, 800, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            self.pics.append(pic)
            self.sizes.append([pic.width(), pic.height()])

        # Load area file containing coordinates
        self.list_names2areas = dataExtraction.list_names2areas()
        for ind, name in enumerate(self.list_names2areas.names):
            if self.target == name:
                self.area = self.list_names2areas.AlignmentBeatLevelIDs[ind]
        self.ymlfile = './areas/{}.yml'.format(self.area)
        with open(self.ymlfile, 'r') as f:
            self.coordinates = yaml.load(f, Loader=yaml.FullLoader)

        # Load reference bar times
        self.time_new_bar = np.load('./annotations/{}_times.npy'.format(self.area))
        # Special cases
        if self.target == 'Don-Giovanni_Act-1_Scene-5_Recitativo':
            self.time_new_bar = np.concatenate(([0], self.time_new_bar))

        # Load reference page times
        self.time_new_page = np.zeros(self.coordinates[-1]['page'])
        # create a counter for bars which require 2 ares
        count=0
        for i in range(len(self.coordinates)):
            # Check weird areas
            if self.coordinates[i]['beats'][0] == self.coordinates[i]['beats'][1]:
                count += 1
            # Adding time of a new page
            if i == 0:
                self.time_new_page[0] = self.time_new_bar[i]
            elif self.coordinates[i]['page'] != self.coordinates[i-1]['page']:
                self.time_new_page[self.coordinates[i]['page'] - 1] = self.time_new_bar[i-count]

        # Load points from coordinates
        self.points = []
        for b in range(len(self.coordinates)):
            if b != 0 and self.coordinates[b-1]['beats'][0] == self.coordinates[b-1]['beats'][1]:
                new_points = [
                        QtCore.QPoint(self.coordinates[b]['topLeft'][0] * self.sizes[self.coordinates[b]['page']-1][0],
                                    self.coordinates[b]['topLeft'][1] * self.sizes[self.coordinates[b]['page']-1][1]),
                        QtCore.QPoint(self.coordinates[b]['topRight'][0] * self.sizes[self.coordinates[b]['page']-1][0],
                                    self.coordinates[b]['topRight'][1] * self.sizes[self.coordinates[b]['page']-1][1]),
                        QtCore.QPoint(self.coordinates[b]['bottomRight'][0] * self.sizes[self.coordinates[b]['page']-1][0],
                                    self.coordinates[b]['bottomRight'][1] * self.sizes[self.coordinates[b]['page']-1][1]),
                        QtCore.QPoint(self.coordinates[b]['bottomLeft'][0] * self.sizes[self.coordinates[b]['page']-1][0],
                                    self.coordinates[b]['bottomLeft'][1] * self.sizes[self.coordinates[b]['page']-1][1])
                        ]
                self.points[-1] = [self.points[-1], new_points]
            else:
                self.points.append([
                        QtCore.QPoint(self.coordinates[b]['topLeft'][0] * self.sizes[self.coordinates[b]['page']-1][0],
                                    self.coordinates[b]['topLeft'][1] * self.sizes[self.coordinates[b]['page']-1][1]),
                        QtCore.QPoint(self.coordinates[b]['topRight'][0] * self.sizes[self.coordinates[b]['page']-1][0],
                                    self.coordinates[b]['topRight'][1] * self.sizes[self.coordinates[b]['page']-1][1]),
                        QtCore.QPoint(self.coordinates[b]['bottomRight'][0] * self.sizes[self.coordinates[b]['page']-1][0],
                                    self.coordinates[b]['bottomRight'][1] * self.sizes[self.coordinates[b]['page']-1][1]),
                        QtCore.QPoint(self.coordinates[b]['bottomLeft'][0] * self.sizes[self.coordinates[b]['page']-1][0],
                                    self.coordinates[b]['bottomLeft'][1] * self.sizes[self.coordinates[b]['page']-1][1])
                        ])

        if len(self.time_new_bar) != len(self.points):
            print('Different numbers of bars and areas!')
            print('Nb bars:', len(self.time_new_bar))
            print('Nb areas:', len(self.points))
            sys.exit()

        # Variables for plotting
        self.position = 0
        self.index_page = 0
        self.index_bar = 0

        # Load initial picture
        self.image = self.pics[self.index_page]
        self.update()

    def pictureUpdate(self, position):
        self.position = position
        if self.index_page < len(self.time_new_page)-1 and self.position >= self.time_new_page[self.index_page+1]:
            self.index_page += 1
            self.image = self.pics[self.index_page]
        self.update()

    def areaUpdate(self, position):
        self.position = position
        if self.index_bar < len(self.time_new_bar)-1 and self.position >= self.time_new_bar[self.index_bar+1]:
            self.index_bar += 1
        self.update()

    def paintEvent(self, event):
        condition_0 = (self.position == 0 and self.index_page == 0 and self.index_bar == 0)
        condition_1 = self.index_bar < len(self.time_new_bar) and self.position >= self.time_new_bar[self.index_bar]
        if condition_0 or condition_1:

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
        self.qlabel.setText('Select a part, a subtitle language, and click on Start.')

        # Init subtitling variables
        self.list_languages = dataExtraction.list_languages()
        self.nb_languages = len(self.list_languages.languages)
        self.idx_language = self.list_languages.languages.index('English')

    def load_subtitles(self, target_printed='Don Giovanni: 1.01 Ouvertura', target_language='English'):
        self.target_printed = target_printed
        self.data_names = dataExtraction.list_names2areas()
        for i in range(len(self.data_names.printed_names)):
            if self.data_names.printed_names[i] == self.target_printed:
                self.target_part = self.data_names.names[i]

        # Load subtitles file
        self.subtitles_file = pd.read_excel('./lyrics/DonGiovanni/WSO_Don_Giovanni.xlsx', header=None).to_numpy()

        # Load target language column
        self.target_language = target_language
        self.idx_language = self.list_languages.languages.index(self.target_language)

        # Load annotations
        for file in os.listdir('./lyrics/DonGiovanni/'):
            if self.target_part in file:
                self.annot_file = file
        self.lyrics_annot = pd.read_csv('./lyrics/DonGiovanni/' + self.annot_file, sep='\t', names=['time', 'line']).to_numpy()

        # Load target subtitles
        self.subtitles_labels = [[] for l in range(self.nb_languages)]
        self.subtitles_times = []
        # Add empty subtitle at start
        if self.lyrics_annot.shape[0]==0 or self.lyrics_annot[0, 0] != 0:
            self.subtitles_times.append(0)
            for l in range(self.nb_languages):
                self.subtitles_labels[l].append('')

        if self.lyrics_annot.shape[0] !=0: # for non-instrumental parts
            for t in range(self.lyrics_annot.shape[0]):
                if t==0 or int(self.lyrics_annot[t, 1]!=self.lyrics_annot[t-1, 1]+1): # Reduce annotations to paragraphs
                    self.subtitles_times.append(round(self.lyrics_annot[t, 0]*100))
                    for l in range(self.nb_languages):
                        self.label = str(self.subtitles_file[int(self.lyrics_annot[t, 1])-1, l])
                        self.label_counter = 0
                        self.label_next = str(self.subtitles_file[int(self.lyrics_annot[t, 1]+self.label_counter), l])
                        while not self.label_next.isspace() and self.label_next != 'nan':
                            self.label += '\n' + str(self.subtitles_file[int(self.lyrics_annot[t, 1]+self.label_counter), l])
                            self.label_counter += 1
                            self.label_next = str(self.subtitles_file[int(self.lyrics_annot[t, 1]+self.label_counter), l])
                        self.subtitles_labels[l].append(self.label)

        # Variable for plotting
        self.position = 0
        self.index_subtitle = 0

        # Load initial subtitle
        self.qlabel.setText(self.subtitles_labels[self.idx_language][self.index_subtitle])

    def subtitleUpdate(self, position):
        self.position = position
        if self.index_subtitle < len(self.subtitles_times)-1 and self.position >= self.subtitles_times[self.index_subtitle+1]:
            self.index_subtitle += 1
        self.qlabel.setText(self.subtitles_labels[self.idx_language][self.index_subtitle])
        self.update()

    def languageUpdate(self, new_language):
        self.new_language = new_language
        self.new_idx_language = self.list_languages.languages.index(self.new_language)
        if self.new_idx_language != self.idx_language:
            self.idx_language = self.new_idx_language
        self.update()



class ComboBox_Parts(QtWidgets.QComboBox):

    def __init__(self):
        QtWidgets.QComboBox.__init__(self)
        self.setStyleSheet("max-width: 310px;")

        # List of names
        self.data_names = dataExtraction.list_names2areas()
        for i in range(len(self.data_names.printed_names)):
            self.addItem(self.data_names.printed_names[i])


class ComboBox_Languages(QtWidgets.QComboBox):

    def __init__(self, subtitle):
        QtWidgets.QComboBox.__init__(self)

        # List of languages
        self.data_languages = dataExtraction.list_languages()
        for i in range(len(self.data_languages.languages)):
            self.addItem(self.data_languages.languages[i])

        # Connect selected language
        self.currentIndexChanged.connect(self.update)

        # Add class
        self.subtitle = subtitle

    def update(self):
        self.subtitle.languageUpdate(self.currentText())


class ComboBox_Features(QtWidgets.QComboBox):

    def __init__(self):
        QtWidgets.QComboBox.__init__(self)

        # List of features
        self.addItem('MFCC')
        self.addItem('Posteriogram')


class PushButton(QtWidgets.QPushButton):

    def __init__(self, menu, score, language, subtitle, feature, align):
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
        self.feature = feature
        self.align = align

    def update(self):
        if self.text() == 'Start':
            self.score.load_pictures(target_printed=self.menu.currentText())
            self.subtitle.load_subtitles(target_printed=self.menu.currentText(), target_language=self.language.currentText())
            self.align.load_song(target_printed=self.menu.currentText(), target_feature=self.feature.currentText())

            # Clear buffer
            while np.linalg.norm(self.align.MFCC_audio_stream.next(), ord=2) > 0.0001 or np.linalg.norm(self.align.POST_audio_stream.next(), ord=2) > 0.0001: # when using internal sound
            # while np.linalg.norm(self.align.audio_stream.next(), ord=2) > 1: # when using microphone (ambiant noise)
                print('clear:', np.linalg.norm(self.align.MFCC_audio_stream.next(), ord=2))
                time.sleep(0.01)

            self.align.timer_align.start()
            self.setText('Stop')

        else:
            self.align.timer_align.stop()
            self.setText('Start')


def main():

    app = QtWidgets.QApplication([])
    print("Main application thread is : ", int(app.thread().currentThreadId()))
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setWindowTitle("Opera Tracker")
    MainWindow.resize(600, 1000) # Size of the principal window
    MainWindow.move(737, 0)
    centralWidget = QtWidgets.QWidget(MainWindow)

    # Vertical layout to divide Score and Buttons
    verticalLayout = QtWidgets.QVBoxLayout(centralWidget)

    # Add image widget
    score = ImageWidget()
    score.load_pictures()
    # score.show()
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

    # Add dropdown button with part names, languages, and start click
    menu_horizontalLayout = QtWidgets.QHBoxLayout()

    # Drop menu with part names
    menu = ComboBox_Parts()
    menu_horizontalLayout.addWidget(menu)

    # Drop menu with languages
    language = ComboBox_Languages(subtitle=subtitle)
    menu_horizontalLayout.addWidget(language)

    # Drop menu with features
    feature = ComboBox_Features()
    menu_horizontalLayout.addWidget(feature)

    # Start/Stop button
    push = PushButton(menu=menu, score=score, language=language, subtitle=subtitle, feature=feature, align=align)
    menu_horizontalLayout.addWidget(push)

    verticalLayout.addLayout(menu_horizontalLayout)

    MainWindow.setCentralWidget(centralWidget)
    MainWindow.show()
    app.exec_()                  




if __name__ == '__main__':
    main()



