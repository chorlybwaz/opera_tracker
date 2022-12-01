import os

import numpy as np
import scipy.spatial

import fast_dtw


class audio2audio_alignment(object):
    def __init__(self, target, feature):
        self.target = target
        self.feature = feature

        # Load reference file
        if self.feature == 'MFCC':
            self.feature_folder = './audio/DonGiovanni/Karajan/audio_features/{}/'.format(self.feature.lower())
            for file in os.listdir(self.feature_folder):
                if self.target + '_{}.npy'.format(self.feature.lower()) in file:
                    self.reference = './audio/DonGiovanni/Karajan/audio_features/{}/'.format(self.feature.lower()) + file
                    self.spec_reference = np.load(self.reference)
        elif self.feature == 'Posteriogram':
            self.feature_folder = './audio/DonGiovanni/Karajan/posteriograms/5lang/'
            for file in os.listdir(self.feature_folder):
                if self.target + '.npy' in file:
                    self.reference = './audio/DonGiovanni/Karajan/posteriograms/5lang/' + file
                    self.spec_reference = np.load(self.reference).astype(np.float32)
                    self.spec_reference = np.exp(self.spec_reference[:, :-1]) + 1e-10

        # Hyperparameters for DTW
        self.size_vec = 8000 # corresponds to 5s in the scores
        self.neighbours = 1500
        self.actual_position = 1
        self.actual_time = 1

        # prepare vector for DTW
        self.gamma = np.ones((2, len(self.spec_reference) + 1)).astype(np.float64) * np.inf
        self.gamma[0, 0] = 0


    def dist_calc(self, matrix, vector, type):
        """
        Compute the cosine distances between each row of matrix and vector.
        """
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(matrix, v, type).reshape(-1)


    def local_DTW(self, audio_frame):
        # inf and sup indexes of interval
        self.middle = max(self.actual_position, self.size_vec)
        self.middle_inf = max(0, self.middle - self.size_vec)
        self.middle_sup = min(self.middle + self.size_vec, len(self.spec_reference))

        # Reseting position variables
        self.pos = 0
        self.cost_pos = np.inf

        # Distance
        if self.feature == 'MFCC':
            self.dist = fast_dtw.dist_VecToMat(audio_frame, self.spec_reference[self.middle_inf:self.middle_sup, :].T)
        elif self.feature == 'Posteriogram':
            self.dist = self.dist_calc(self.spec_reference[self.middle_inf:self.middle_sup, :], audio_frame, 'cosine') # delay:280ms

        # Cumulative distance and cost
        self.gamma, self.cost_vec = fast_dtw.costVec(self.middle_inf, self.middle_sup, self.gamma, self.dist, self.actual_time)

        # Reduce the scope to find the score position
        self.pos_inf = max(0, self.actual_position - self.neighbours)
        self.pos_sup = min(len(self.spec_reference), self.actual_position + self.neighbours)
        self.pos = self.pos_inf + np.argmin(self.cost_vec[self.pos_inf:self.pos_sup])

        # Find the new score position
        self.actual_position = min(max(self.actual_position, self.pos), self.actual_position + 5)
        self.actual_time += 1




class list_names2areas(object):
    def __init__(self):
        self.names = []
        self.printed_names = []
        self.AlignmentBeatLevelIDs = []

        self.names.append('Don-Giovanni_Act-1_Ouvertura_')
        self.printed_names.append('Don Giovanni: 1.01 Ouvertura')
        self.AlignmentBeatLevelIDs.append(805)

        self.names.append('Don-Giovanni_Act-1_Scene-1_N1-Introduzione')
        self.printed_names.append('Don Giovanni: 1.02 Notte e giorno faticar.')
        self.AlignmentBeatLevelIDs.append(807)

        self.names.append('Don-Giovanni_Act-1_Scene-2_Recitativo')
        self.printed_names.append('Don Giovanni: 1.03 Leporello, ove sei?')
        self.AlignmentBeatLevelIDs.append(809)

        self.names.append('Don-Giovanni_Act-1_Scene-3_Recitativo')
        self.printed_names.append('Don Giovanni: 1.04 Ah del padre in periglio.')
        self.AlignmentBeatLevelIDs.append(811)

        self.names.append('Don-Giovanni_Act-1_Scene-3_N2-Recitativo')
        self.printed_names.append('Don Giovanni: 1.05 Ma qual mai s’offre, oh Dei / Fuggi, crudele, fuggi!')
        self.AlignmentBeatLevelIDs.append(813)

        self.names.append('Don-Giovanni_Act-1_Scene-4_Recitativo')
        self.printed_names.append('Don Giovanni: 1.06 Orsù, spicciati presto ...')
        self.AlignmentBeatLevelIDs.append(815)

        self.names.append('Don-Giovanni_Act-1_Scene-5_N3-Aria')
        self.printed_names.append('Don Giovanni: 1.07 Ah chi mi dice mai.')
        self.AlignmentBeatLevelIDs.append(817)

        self.names.append('Don-Giovanni_Act-1_Scene-5_Recitativo')
        self.printed_names.append('Don Giovanni: 1.08 Chi è la? Stelle! che vedo!')
        self.AlignmentBeatLevelIDs.append(819)

        self.names.append('Don-Giovanni_Act-1_Scene-5_N4-Aria')
        self.printed_names.append('Don Giovanni: 1.09 Madamina, il catalogo è questo.')
        self.AlignmentBeatLevelIDs.append(796)

        self.names.append('Don-Giovanni_Act-1_Scene-6_Recitativo')
        self.printed_names.append('Don Giovanni: 1.10 In questa forma dunque.')
        self.AlignmentBeatLevelIDs.append(1078)

        self.names.append('Don-Giovanni_Act-1_Scene-7_N5-Coro')
        self.printed_names.append('Don Giovanni: 1.11 Giovinette che fate all’amore.')
        self.AlignmentBeatLevelIDs.append(821)

        self.names.append('Don-Giovanni_Act-1_Scene-8_Recitativo')
        self.printed_names.append('Don Giovanni: 1.12 Manco male è partita.')
        self.AlignmentBeatLevelIDs.append(823)

        self.names.append('Don-Giovanni_Act-1_Scene-8_N6-Aria')
        self.printed_names.append('Don Giovanni: 1.13 Ho capito, signor sì')
        self.AlignmentBeatLevelIDs.append(825)

        self.names.append('Don-Giovanni_Act-1_Scene-9_Recitativo')
        self.printed_names.append('Don Giovanni: 1.14 Alfin siam liberati.')
        self.AlignmentBeatLevelIDs.append(827)

        self.names.append('Don-Giovanni_Act-1_Scene-9_N7-Duetto')
        self.printed_names.append('Don Giovanni: 1.15 Là ci darem la mano.')
        self.AlignmentBeatLevelIDs.append(794)

        self.names.append('Don-Giovanni_Act-1_Scene-10_Recitativo')
        self.printed_names.append('Don Giovanni: 1.16 Fermati scellerato.')
        self.AlignmentBeatLevelIDs.append(829)

        self.names.append('Don-Giovanni_Act-1_Scene-10_N8-Aria')
        self.printed_names.append('Don Giovanni: 1.17 Ah fuggi il traditor.')
        self.AlignmentBeatLevelIDs.append(831)

        self.names.append('Don-Giovanni_Act-1_Scene-11_Recitativo')
        self.printed_names.append('Don Giovanni: 1.18 Mi par ch’oggi il demonio si diverta')
        self.AlignmentBeatLevelIDs.append(833)

        self.names.append('Don-Giovanni_Act-1_Scene-12_')
        self.printed_names.append('Don Giovanni: 1.19 Ah ti ritrovo ancor.')
        self.AlignmentBeatLevelIDs.append(835)

        self.names.append('Don-Giovanni_Act-1_Scene-12_N9-Quartetto')
        self.printed_names.append('Don Giovanni: 1.20 Non ti fidar, o misera.')
        self.AlignmentBeatLevelIDs.append(837)

        self.names.append('Don-Giovanni_Act-1_Scene-12_Recitativo')
        self.printed_names.append('Don Giovanni: 1.21 Povera sventurata!')
        self.AlignmentBeatLevelIDs.append(839)

        self.names.append('Don-Giovanni_Act-1_Scene-13_N10-Recitativo')
        self.printed_names.append('Don Giovanni: 1.22 Don Ottavio, son morta! / Or sai chi l’onore.')
        self.AlignmentBeatLevelIDs.append(841)

        self.names.append('Don-Giovanni_Act-1_Scene-14_Recitativo')
        self.printed_names.append('Don Giovanni: 1.23 Come mai creder deggio.')
        self.AlignmentBeatLevelIDs.append(843)

        self.names.append('Don-Giovanni_Act-1_Scene-14_N10a-Aria')
        self.printed_names.append('Don Giovanni: 1.24 Dalla sua pace.')
        self.AlignmentBeatLevelIDs.append(845)

        self.names.append('Don-Giovanni_Act-1_Scene-15_Recitativo')
        self.printed_names.append('Don Giovanni: 1.25 Io deggio ad ogni patto.')
        self.AlignmentBeatLevelIDs.append(847)

        self.names.append('Don-Giovanni_Act-1_Scene-15_N11-Aria')
        self.printed_names.append('Don Giovanni: 1.26 Fin ch’han dal vino.')
        self.AlignmentBeatLevelIDs.append(798)

        self.names.append('Don-Giovanni_Act-1_Scenes-16-and-17_Recitativo-1')
        self.printed_names.append('Don Giovanni: 1.27 Masetto: senti un po’!')
        self.AlignmentBeatLevelIDs.append(849)

        self.names.append('Don-Giovanni_Act-1_Scenes-16-and-17_N12-Aria')
        self.printed_names.append('Don Giovanni: 1.28 Batti, batti, o bel Masetto / Pace, pace, o vita mia')
        self.AlignmentBeatLevelIDs.append(851)

        self.names.append('Don-Giovanni_Act-1_Scenes-16-and-17_Recitativo-2')
        self.printed_names.append('Don Giovanni: 1.29 Guarda un po’come seppe.')
        self.AlignmentBeatLevelIDs.append(853)

        self.names.append('Don-Giovanni_Act-1_Scenes-16-and-17_N13-Finale-+-Scena-17')
        self.printed_names.append('Don Giovanni: 1.30 Presto presto pria ch’ei venga.')
        self.AlignmentBeatLevelIDs.append(855)

        self.names.append('Don-Giovanni_Act-1_Scenes-18-and-19_')
        self.printed_names.append('Don Giovanni: 1.31 Tra quest’ arbori celata. / Adesso fate core, fate core:. / Signor guardate un poco. / Protegga il giusto cielo.')
        self.AlignmentBeatLevelIDs.append(857)

        self.names.append('Don-Giovanni_Act-1_Scene-20_Allegro')
        self.printed_names.append('Don Giovanni: 1.32 Riposate, vezzose ragazze. / Venite pure avanti. / Da bravi, via, ballate! / Soccorriamo, l’innocente! / Ecco il birbo. / Trema, trema, o scellerato! / Sul tuo capo in questo giorno.')
        self.AlignmentBeatLevelIDs.append(859)

        self.names.append('Don-Giovanni_Act-2_Scene-1_N14-Duetto')
        self.printed_names.append('Don Giovanni: 2.01 Eh via buffone.')
        self.AlignmentBeatLevelIDs.append(889)

        self.names.append('Don-Giovanni_Act-2_Scene-1_Recitativo')
        self.printed_names.append('Don Giovanni: 2.02 Leporello – Signore – Vien qui.')
        self.AlignmentBeatLevelIDs.append(892)

        self.names.append('Don-Giovanni_Act-2_Scene-2_N15-Terzetto')
        self.printed_names.append('Don Giovanni: 2.03 Ah taci, ingiusto core.')
        self.AlignmentBeatLevelIDs.append(895)

        self.names.append('Don-Giovanni_Act-2_Scene-2_Recitativo')
        self.printed_names.append('Don Giovanni: 2.04 Amico, che ti par?')
        self.AlignmentBeatLevelIDs.append(898)

        self.names.append('Don-Giovanni_Act-2_Scene-3_Recitativo-1')
        self.printed_names.append('Don Giovanni: 2.05 Eccomi a voi!')
        self.AlignmentBeatLevelIDs.append(901)

        self.names.append('Don-Giovanni_Act-2_Scene-3_N16-Canzonetta')
        self.printed_names.append('Don Giovanni: 2.06 Deh vieni alla finestra.')
        self.AlignmentBeatLevelIDs.append(800)

        self.names.append('Don-Giovanni_Act-2_Scene-3_Recitativo-2')
        self.printed_names.append('Don Giovanni: 2.07 V’è gente alla finestra!')
        self.AlignmentBeatLevelIDs.append(904)

        self.names.append('Don-Giovanni_Act-2_Scene-4_')
        self.printed_names.append('Don Giovanni: 2.08 Non ci stanchiamo.')
        self.AlignmentBeatLevelIDs.append(907)

        self.names.append('Don-Giovanni_Act-2_Scene-4_N17-Aria')
        self.printed_names.append('Don Giovanni: 2.09 Metà di voi qua vadano.')
        self.AlignmentBeatLevelIDs.append(910)

        self.names.append('Don-Giovanni_Act-2_Scene-5_Recitativo')
        self.printed_names.append('Don Giovanni: 2.10 Zitto! lascia ch’io senta.')
        self.AlignmentBeatLevelIDs.append(913)

        self.names.append('Don-Giovanni_Act-2_Scene-6_Recitativo')
        self.printed_names.append('Don Giovanni: 2.11 Ahi ahi! la testa mia!')
        self.AlignmentBeatLevelIDs.append(916)

        self.names.append('Don-Giovanni_Act-2_Scene-6_N18-Aria')
        self.printed_names.append('Don Giovanni: 2.12 Vedrai carino.')
        self.AlignmentBeatLevelIDs.append(919)

        self.names.append('Don-Giovanni_Act-2_Scene-7_Recitativo')
        self.printed_names.append('Don Giovanni: 2.13 Di molte faci il lume.')
        self.AlignmentBeatLevelIDs.append(922)

        self.names.append('Don-Giovanni_Act-2_Scene-7_N19-Sestetto')
        self.printed_names.append('Don Giovanni: 2.14 Sola sola in buio loco.')
        self.AlignmentBeatLevelIDs.append(925)

        self.names.append('Don-Giovanni_Act-2_Scene-8_')
        self.printed_names.append('Don Giovanni: 2.15 Ferma, bricone, dove ten vai?')
        self.AlignmentBeatLevelIDs.append(928)

        self.names.append('Don-Giovanni_Act-2_Scene-9_Recitativo')
        self.printed_names.append('Don Giovanni: 2.16 Dunque quello sei tu.')
        self.AlignmentBeatLevelIDs.append(931)

        self.names.append('Don-Giovanni_Act-2_Scene-9_N20-Aria')
        self.printed_names.append('Don Giovanni: 2.17 Ah, pietà, signori miei.')
        self.AlignmentBeatLevelIDs.append(934)

        self.names.append('Don-Giovanni_Act-2_Scene-10_Recitativo')
        self.printed_names.append('Don Giovanni: 2.18 Ferma, perfido, ferma...')
        self.AlignmentBeatLevelIDs.append(937)

        self.names.append('Don-Giovanni_Act-2_Scene-10_N21-Aria')
        self.printed_names.append('Don Giovanni: 2.19 Il mio tesoro intanto.')
        self.AlignmentBeatLevelIDs.append(802)

        self.names.append('Don-Giovanni_Act-2_Scene-10_N21-Recitativo')
        self.printed_names.append('Don Giovanni: 2.20 In quali eccessi, o Numi.')
        self.AlignmentBeatLevelIDs.append(940)

        self.names.append('Don-Giovanni_Act-2_Scene-10_Aria')
        self.printed_names.append('Don Giovanni: 2.21 Mi tradì quell’alma ingrata.')
        self.AlignmentBeatLevelIDs.append(943)

        self.names.append('Don-Giovanni_Act-2_Scene-11_Recitativo')
        self.printed_names.append('Don Giovanni: 2.22 Ah ah ah ah, questa è buona. / Di rider finirai pria dell’aurora. / Chi ha parlato? / Ribaldo, audace. / Sarà qualcun di fuori che si burla.')
        self.AlignmentBeatLevelIDs.append(946)

        self.names.append('Don-Giovanni_Act-2_Scene-11_N22-Duetto')
        self.printed_names.append('Don Giovanni: 2.23 O statua gentilissima.')
        self.AlignmentBeatLevelIDs.append(949)

        self.names.append('Don-Giovanni_Act-2_Scene-12_Recitativo-1-+-N23-Recitativo')
        self.printed_names.append('Don Giovanni: 2.24 Calmatevi, idol mio. / Crudele! – Ah no, mio bene! / Non mi dir, bell’idol mio.')
        self.AlignmentBeatLevelIDs.append(952)

        self.names.append('Don-Giovanni_Act-2_Scene-12_Recitativo-2')
        self.printed_names.append('Don Giovanni: 2.25 Ah, si segua il suo passo.')
        self.AlignmentBeatLevelIDs.append(1080)

        self.names.append('Don-Giovanni_Act-2_Scenes-13-and-14_N24-Finale-+-Scena-14')
        self.printed_names.append('Don Giovanni: 2.26 Già la mensa è preparata. / L’ultima prova dell’amor mio. / Ah signor... per carità!')
        self.AlignmentBeatLevelIDs.append(955)

        self.names.append('Don-Giovanni_Act-2_Scene-15_')
        self.printed_names.append('Don Giovanni: 2.27 Don Giovanni, a cenar teco. / Oimè! che gelo è questo mai? / Da qual tremore insolito...')
        self.AlignmentBeatLevelIDs.append(958)

        self.names.append('Don-Giovanni_Act-2_Final-Scene_')
        self.printed_names.append('Don Giovanni: 2.28 Ah dove è il perfido. / Or che tutti, o mio tesoro. / Questo è il fin di chi fa mal.')
        self.AlignmentBeatLevelIDs.append(961)


class list_languages(object):
    def __init__(self):
        self.languages = []
        self.languages.append('Deutsch')
        self.languages.append('English')
        self.languages.append('Italiano')
        self.languages.append('Français')
        self.languages.append('日本語')
        self.languages.append('Русский язык')
        self.languages.append('普通话')
        self.languages.append('Español')
