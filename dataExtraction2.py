import os
import sys

from PyQt5 import QtGui, QtCore, QtWidgets

import json
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
import scipy.spatial
import yaml

import fast_dtw


class audio2audio_alignment(object):
    def __init__(self, Y, lengths_Y, Y_LR, lengths_Y_LR):
        # Hyperparameters for OLTW
        self.Y = Y
        self.size_vec = 2000 # corresponds to 2s in the scores
        self.neighbours = 1500
        self.middle_inf = 0
        self.middle_sup = 2 * self.size_vec
        self.actual_position = 1
        self.actual_time = 1
        self.gamma = np.ones((2, self.Y.shape[0] + 1)).astype(np.float64) * np.inf
        self.gamma[0, 0] = 0

        # Hyperparameters for JOLTW
        self.lengths_Y = lengths_Y
        self.s_list = np.array(self.lengths_Y[:-1], dtype=np.int32)
        self.t_list = np.array(self.lengths_Y[1:]-1, dtype=np.int32)
        self.penalty = 1/100
        self.checking_parts = 4
        self.state_idx = 0
        self.prev_norm = 0
        self.jump = True
        self.dist_jump_list = []
        for s in self.s_list:
            self.dist_jump_list.append(np.ones([self.size_vec])*np.inf)
        self.dist_jump_list = np.array(self.dist_jump_list)

        # Hyperparameters for LR
        self.Y_LR = Y_LR
        self.saved_audio_frames = np.zeros((60, self.Y.shape[1])).astype('float32')
        self.win_LR = np.hanning(60).astype('float32')
        self.actual_time_LR = 1
        self.gamma_LR = np.ones((2, self.Y_LR.shape[0]+1)) * np.inf
        self.gamma_LR[0, 0] = 0
        self.lengths_Y_LR = lengths_Y_LR
        self.LR_margin = 2000
        self.len_LRDiag = 30
        self.C_LR = np.zeros((self.len_LRDiag, Y_LR.shape[0])) # LR cost matrix
        self.D_LR = np.ones((self.C_LR.shape[0], self.C_LR.shape[1])) * np.inf # Cumulative distance matrix
        self.s_LR_list = np.array(lengths_Y_LR[:-1], dtype=np.int32)
        self.t_LR_list = np.array(lengths_Y_LR[1:]-1, dtype=np.int32)
        self.len_LRsure = 30
        self.lastpos_LR = np.ones(self.len_LRsure)
        self.lastprime_LR = np.ones(self.len_LRsure)
        self.RE = False
        self.var = 8000
        self.rf = 1

        # Hyperparameters for detectors
        self.gamma_save_applause = np.ones((1, Y.shape[1]+1)) * np.inf


    def dist_calc(self, matrix, vector, type):
        """
        Compute the cosine distances between each row of matrix and vector.
        """
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(matrix, v, type).reshape(-1)


    def local_OLTW(self, audio_frame):
        # inf and sup indexes of interval
        self.middle = max(self.actual_position, self.size_vec)
        self.middle_inf = max(0, self.middle - self.size_vec)
        self.middle_sup = min(self.middle + self.size_vec, len(self.Y))

        # Distance
        self.dist = fast_dtw.dist_VecToMat(audio_frame, self.Y[self.middle_inf:self.middle_sup, :].T)

        # Cumulative distance and cost
        self.gamma, self.cost_vec = fast_dtw.costVec(self.middle_inf, self.middle_sup, self.gamma, self.dist, self.actual_time)

        # Reduce the scope to find the score position
        self.pos_inf = max(0, self.actual_position - self.neighbours)
        self.pos_sup = min(len(self.Y), self.actual_position + self.neighbours)
        self.pos = self.pos_inf + np.argmin(self.cost_vec[self.pos_inf:self.pos_sup])

        # Find the new score position
        self.actual_position = min(max(self.actual_position, self.pos), self.actual_position + 5)
        self.actual_time += 1


    def local_JOLTW(self, audio_frame):
        # Current state interval
        if not self.jump:
            self.middle = self.actual_position
        elif self.jump:
            if self.middle_inf <= self.actual_position <= self.middle_sup:
                self.middle = self.actual_position
            # else: nothing, we keep previous middle_inf and middle_sup

        # inf and sup indexes of interval
        self.middle_inf = max(self.s_list[self.state_idx], self.middle - self.size_vec)
        self.middle_sup = min(self.middle + self.size_vec, self.t_list[self.state_idx]+1)

        #####################
        # Compute distances #
        #####################
        # Classical OLTW
        if not (self.middle_inf <= self.t_list[self.state_idx] < self.middle_sup):
            self.jump = False
            self.dist_state = fast_dtw.dist_VecToMat_nonorm(audio_frame, self.Y[self.middle_inf:self.middle_sup, :].T)

        # JumpOLTW: if we are close to the end of the section (but not the last one)
        elif  self.middle_inf <= self.t_list[self.state_idx] < self.middle_sup and self.state_idx != len(self.s_list)-1:
            self.jump = True
            self.dist_state = fast_dtw.dist_VecToMat_nonorm(audio_frame, self.Y[self.middle_inf:self.middle_sup, :].T)

            # for all beginnings
            for ind_s in range(self.state_idx, min(len(self.s_list)-1, self.state_idx+self.checking_parts)):
                if self.middle_inf in [self.s_list[ind_s], self.s_list[ind_s]+2]:
                    pass
                elif self.middle_inf > self.s_list[ind_s]+2:
                    self.dist_jump_list[ind_s][:min(self.size_vec, self.t_list[ind_s]+1-self.s_list[ind_s], self.middle_inf-self.s_list[ind_s]-1)] = fast_dtw.dist_VecToMat_nonorm(audio_frame, self.Y[self.s_list[ind_s]:min(self.s_list[ind_s]+self.size_vec, self.t_list[ind_s]+1, self.middle_inf-1), :].T)
                else:
                    self.dist_jump_list[ind_s][:min(self.size_vec, self.t_list[ind_s]+1-self.s_list[ind_s])] = fast_dtw.dist_VecToMat_nonorm(audio_frame, self.Y[self.s_list[ind_s]:min(self.s_list[ind_s] + self.size_vec, self.t_list[ind_s]+1), :].T)  

        # Compute cumulative distances
        # self.gamma, self.cost_vec = fast_dtw.costVec_jump2(self.middle_inf, self.middle_sup, self.gamma, self.dist_state, self.actual_time, self.jump, self.dist_jump_list, self.s_list, self.t_list, self.state_idx, self.prev_norm, self.size_vec, self.penalty)
        self.gamma, self.cost_vec = fast_dtw.costVec_jump2test(self.middle_inf, self.middle_sup, self.gamma, self.dist_state, self.checking_parts, self.actual_time, self.jump, self.dist_jump_list, self.s_list, self.t_list, self.state_idx, self.prev_norm, self.size_vec, self.penalty)
        # Reduce the scope to find the score position
        self.pos = np.argmin(self.cost_vec) - 1

        # if jump is activated and score position not at the end of the current song or at a beginning of another song: we have jumped!
        if self.jump and not (self.t_list[self.state_idx] - 2*self.size_vec <= self.pos <= self.t_list[self.state_idx] or any([s <= self.pos < s + 500 for s in self.s_list])):
            self.jump = False
            self.prev_norm += self.t_list[self.state_idx] - self.s_list[self.state_idx] # add length of previous state_idx
            self.state_idx = len(self.s_list[self.s_list<self.pos]) - 1 # new state: the one where pos is slightly superior
            self.gamma[0, :max(1, self.s_list[self.state_idx]+1)] = self.gamma[0, self.t_list[self.state_idx]+1:] = np.inf # shift of one index for gamma!
            print('New part tracked!')

        # Reinitializing distances
        for s in range(len(self.s_list)):
            self.dist_jump_list[s] = np.ones([self.size_vec]) * np.inf

        # Find the new score position
        if not self.jump or self.pos < self.middle_sup:
            self.actual_position = min(max(self.actual_position, self.pos), self.actual_position + 5)
        else:
            self.actual_position = self.pos

        self.actual_time += 1


    def local_LR(self, audio_frame_LR):
        self.actual_time_LR = int(self.actual_time/30)

        # Diagonal matching
        self.C_LR[self.actual_time_LR%self.len_LRDiag, :] = fast_dtw.dist_VecToMat(audio_frame_LR, self.Y_LR.T)
        for i_X in range(self.len_LRDiag):
            if i_X == 0:
                self.D_LR[i_X, :] = self.C_LR[(self.actual_time_LR+1)%self.len_LRDiag, :]
            else:
                self.D_LR[i_X, 0] = self.D_LR[i_X-1, 0] + self.C_LR[(self.actual_time_LR+1+i_X)%self.len_LRDiag, 0]
                if i_X<self.len_LRDiag-1:
                    self.D_LR[i_X:i_X+2, :] = fast_dtw.diagonal_matching_jump(1, self.Y_LR.shape[0], self.D_LR[i_X-1:i_X+1, :], self.C_LR[(self.actual_time_LR+1+i_X)%self.len_LRDiag, :], self.s_LR_list, self.t_LR_list)
                else:
                    gamma_final = fast_dtw.diagonal_matching(1, self.Y_LR.shape[0], self.D_LR[i_X-1:i_X+1], self.C_LR[(self.actual_time_LR+1+i_X)%self.len_LRDiag, :])
                    self.D_LR[i_X, :] = gamma_final[0, :]

        # Normalizing
        self.D_LR_normed = self.D_LR[i_X, :]
        self.D_LR_normed = self.D_LR_normed - min(self.D_LR_normed)
        max_val = max(self.D_LR_normed[self.D_LR_normed!=np.inf])
        self.D_LR_normed[self.D_LR_normed==np.inf] = max_val
        self.D_LR_normed = self.D_LR_normed / max_val

        # JOLTW with self.D_LR_normed as cost
        self.gamma_LR = fast_dtw.cumulative_matrix_jump(0, self.Y_LR.shape[0], self.gamma_LR, self.D_LR_normed, self.s_LR_list, self.t_LR_list)
        self.pos_LR = np.argmin(self.gamma_LR[0, :])

        # Reliability factor
        self.lastpos_LR = np.concatenate((self.lastpos_LR[1:], [self.pos_LR]))
        self.lastprime_LR = np.concatenate((self.lastprime_LR[1:], [self.lastpos_LR[-1]-self.lastpos_LR[0]]))
        if (15<self.lastprime_LR).all() and (self.lastprime_LR<45).all():
            self.rf = 1
        else:
            self.rf = 0

    def local_JOLTWLR(self, audio_frame):
        # Current state interval
        if not self.jump:
            self.middle = self.actual_position
        elif self.jump:
            if self.middle_inf <= self.actual_position <= self.middle_sup:
                self.middle = self.actual_position
            # else: nothing, we keep previous middle_inf and middle_sup

        # inf and sup indexes of interval
        self.middle_inf = max(self.s_list[self.state_idx], self.middle - self.size_vec)
        self.middle_sup = min(self.middle + self.size_vec, self.t_list[self.state_idx]+1)

        #####################
        # Compute distances #
        #####################
        # Classical OLTW
        if not (self.middle_inf <= self.t_list[self.state_idx] < self.middle_sup):
            self.jump = False
            self.dist_state = fast_dtw.dist_VecToMat_nonorm(audio_frame, self.Y[self.middle_inf:self.middle_sup, :].T)

        # JumpOLTW: if we are close to the end of the section (but not the last one)
        elif  self.middle_inf <= self.t_list[self.state_idx] < self.middle_sup and self.state_idx != len(self.s_list)-1:
            self.jump = True
            self.dist_state = fast_dtw.dist_VecToMat_nonorm(audio_frame, self.Y[self.middle_inf:self.middle_sup, :].T)

            # for all beginnings
            for ind_s in range(self.state_idx, min(len(self.s_list)-1, self.state_idx+self.checking_parts)):
                if self.middle_inf in [self.s_list[ind_s], self.s_list[ind_s]+2]:
                    pass
                elif self.middle_inf > self.s_list[ind_s]+2:
                    self.dist_jump_list[ind_s][:min(self.size_vec, self.t_list[ind_s]+1-self.s_list[ind_s], self.middle_inf-self.s_list[ind_s]-1)] = fast_dtw.dist_VecToMat_nonorm(audio_frame, self.Y[self.s_list[ind_s]:min(self.s_list[ind_s]+self.size_vec, self.t_list[ind_s]+1, self.middle_inf-1), :].T)
                else:
                    self.dist_jump_list[ind_s][:min(self.size_vec, self.t_list[ind_s]+1-self.s_list[ind_s])] = fast_dtw.dist_VecToMat_nonorm(audio_frame, self.Y[self.s_list[ind_s]:min(self.s_list[ind_s] + self.size_vec, self.t_list[ind_s]+1), :].T)  

        # Compute cumulative distances
        # self.gamma, self.cost_vec = fast_dtw.costVec_jump2(self.middle_inf, self.middle_sup, self.gamma, self.dist_state, self.actual_time, self.jump, self.dist_jump_list, self.s_list, self.t_list, self.state_idx, self.prev_norm, self.size_vec, self.penalty)
        self.gamma, self.cost_vec = fast_dtw.costVec_jump2test(self.middle_inf, self.middle_sup, self.gamma, self.dist_state, self.checking_parts, self.actual_time, self.jump, self.dist_jump_list, self.s_list, self.t_list, self.state_idx, self.prev_norm, self.size_vec, self.penalty)
        # Reduce the scope to find the score position
        self.pos = np.argmin(self.cost_vec) - 1

        # Compute LR
        self.saved_audio_frames = np.vstack((self.saved_audio_frames[1:, :], audio_frame))
        if self.actual_time > 500 and self.actual_time%30 == 0:
            self.audio_frame_LR = np.dot(self.win_LR, self.saved_audio_frames)
            self.local_LR(self.audio_frame_LR)
            if self.rf == 1:
                state_idx_LR = len(self.s_LR_list[self.s_LR_list<self.pos_LR]) - 1 # correct state
                relative_pos_LR = self.pos_LR - self.s_LR_list[state_idx_LR] # relative position
                pos_HR = self.s_list[state_idx_LR] + relative_pos_LR * 30 # High Resolution position
                ## PART CHANGE ##
                if self.pos_LR - self.s_LR_list[state_idx_LR] > 10 and state_idx_LR != self.state_idx: # but the song we are curently tracking is different from the rough position song
                    print('Tracker lost!')
                    self.state_idx = state_idx_LR
                    self.gamma[0, :] = np.inf
                    self.gamma[0, pos_HR+1:pos_HR+1+30] = self.D_LR_normed[self.pos_LR] * self.var
                    self.prev_norm = self.s_list[self.state_idx] - self.actual_time - pos_HR  + 4000
                    self.jump = False
                    self.RE = True
                    self.pos = pos_HR
                ## LOCAL CORRECTIONS ##
                elif self.pos_LR - self.s_LR_list[state_idx_LR] > 100 and self.t_LR_list[state_idx_LR] - self.pos_LR > 100 and state_idx_LR == self.state_idx and not (pos_HR - self.LR_margin < self.pos < pos_HR + self.LR_margin):
                    print('Correcting tracking from pos=', self.pos, 'to pos=', pos_HR)
                    self.gamma[0, :] = np.inf
                    self.gamma[0, pos_HR+1:pos_HR+1+30] = self.D_LR_normed[self.pos_LR] * self.var
                    self.prev_norm = self.s_list[self.state_idx] - self.actual_time - pos_HR  + 4000
                    self.jump = False
                    self.RE = True
                    self.pos = pos_HR+1+30
                else:
                    self.RE = False
            else:
                self.RE = False

        # Find the new score position
        if self.jump == False and self.RE == False:
            self.pos = min(max(self.actual_position, self.pos), self.actual_position + 5)

        # if jump is activated and score position not at the end of the current song or at a beginning of another song: we have jumped!
        if self.jump and not (self.t_list[self.state_idx] - 2*self.size_vec <= self.pos <= self.t_list[self.state_idx] or any([s <= self.pos < s + 500 for s in self.s_list])):
            self.jump = False
            self.prev_norm += self.t_list[self.state_idx] - self.s_list[self.state_idx] # add length of previous state_idx
            self.state_idx = len(self.s_list[self.s_list<self.pos]) - 1 # new state: the one where pos is slightly superior
            self.gamma[0, :max(1, self.s_list[self.state_idx]+1)] = self.gamma[0, self.t_list[self.state_idx]+1:] = np.inf # shift of one index for gamma!
            print('New part tracked!')

        # Reinitializing distances
        for s in range(len(self.s_list)):
            self.dist_jump_list[s] = np.ones([self.size_vec]) * np.inf

        # Update position
        # if not self.jump or self.pos < self.middle_sup:
        #     self.actual_position = min(max(self.actual_position, self.pos), self.actual_position + 5)
        # else:
        #     self.actual_position = self.pos
        self.actual_position = self.pos
        self.actual_time += 1


class data_loading(object):
    def __init__(self, target, feature='MFCC'):
        self.target = target
        self.feature = feature

        #############
        #   Names   #
        #############

        # Parts in reference
        if 'Act 1' in self.target:
            self.parts_Y = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44']
        elif 'Act 2' in self.target:
            self.parts_Y = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

        # Loading file names and annotations
        self.Y_names = [] # Reference file names
        self.AlignmentBeatLevelIDs = [] # Reference annotations
        if 'Act 1' in self.target:
            # Don-Giovanni_Act-1_Ouvertura_
            self.Y_names.append('Don-Giovanni_Act-1_Ouvertura_')
            self.AlignmentBeatLevelIDs.append(805)
            # Don-Giovanni_Act-1_Scene-1_N1-Introduzione
            self.Y_names.append('Don-Giovanni_Act-1_Scene-1_N1-Introduzione')
            self.AlignmentBeatLevelIDs.append(807)
            # Don-Giovanni_Act-1_Scene-2_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-2_Recitativo')
            self.AlignmentBeatLevelIDs.append(809)
            # Don-Giovanni_Act-1_Scene-3_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-3_Recitativo')
            self.AlignmentBeatLevelIDs.append(811)
            # Don-Giovanni_Act-1_Scene-3_N2-Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-3_N2-Recitativo')
            self.AlignmentBeatLevelIDs.append(813)
            # Don-Giovanni_Act-1_Scene-4_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-4_Recitativo')
            self.AlignmentBeatLevelIDs.append(815)
            # Don-Giovanni_Act-1_Scene-5_N3-Aria
            self.Y_names.append('Don-Giovanni_Act-1_Scene-5_N3-Aria')
            self.AlignmentBeatLevelIDs.append(817)
            # Don-Giovanni_Act-1_Scene-5_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-5_Recitativo')
            self.AlignmentBeatLevelIDs.append(819)
            # Don-Giovanni_Act-1_Scene-5_N4-Aria
            self.Y_names.append('Don-Giovanni_Act-1_Scene-5_N4-Aria')
            self.AlignmentBeatLevelIDs.append(796)
            # Don-Giovanni_Act-1_Scene-6_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-6_Recitativo')
            self.AlignmentBeatLevelIDs.append(1078)
            # Don-Giovanni_Act-1_Scene-7_N5-Coro
            self.Y_names.append('Don-Giovanni_Act-1_Scene-7_N5-Coro')
            self.AlignmentBeatLevelIDs.append(821)
            # Don-Giovanni_Act-1_Scene-8_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-8_Recitativo')
            self.AlignmentBeatLevelIDs.append(823)
            # Don-Giovanni_Act-1_Scene-8_N6-Aria
            self.Y_names.append('Don-Giovanni_Act-1_Scene-8_N6-Aria')
            self.AlignmentBeatLevelIDs.append(825)
            # Don-Giovanni_Act-1_Scene-9_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-9_Recitativo')
            self.AlignmentBeatLevelIDs.append(827)
            # Don-Giovanni_Act-1_Scene-9_N7-Duetto
            self.Y_names.append('Don-Giovanni_Act-1_Scene-9_N7-Duetto')
            self.AlignmentBeatLevelIDs.append(794)
            # Don-Giovanni_Act-1_Scene-10_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-10_Recitativo')
            self.AlignmentBeatLevelIDs.append(829)
            # Don-Giovanni_Act-1_Scene-10_N8-Aria
            self.Y_names.append('Don-Giovanni_Act-1_Scene-10_N8-Aria')
            self.AlignmentBeatLevelIDs.append(831)
            # Don-Giovanni_Act-1_Scene-11_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-11_Recitativo')
            self.AlignmentBeatLevelIDs.append(833)
            # Don-Giovanni_Act-1_Scene-12_
            self.Y_names.append('Don-Giovanni_Act-1_Scene-12_')
            self.AlignmentBeatLevelIDs.append(835)
            # Don-Giovanni_Act-1_Scene-12_N9-Quartetto
            self.Y_names.append('Don-Giovanni_Act-1_Scene-12_N9-Quartetto')
            self.AlignmentBeatLevelIDs.append(837)
            # Don-Giovanni_Act-1_Scene-12_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-12_Recitativo')
            self.AlignmentBeatLevelIDs.append(839)
            # Don-Giovanni_Act-1_Scene-13_N10-Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-13_N10-Recitativo')
            self.AlignmentBeatLevelIDs.append(841)
            # Don-Giovanni_Act-1_Scene-14_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-14_Recitativo')
            self.AlignmentBeatLevelIDs.append(843)
            # Don-Giovanni_Act-1_Scene-14_N10a-Aria
            self.Y_names.append('Don-Giovanni_Act-1_Scene-14_N10a-Aria')
            self.AlignmentBeatLevelIDs.append(845)
            # Don-Giovanni_Act-1_Scene-15_Recitativo
            self.Y_names.append('Don-Giovanni_Act-1_Scene-15_Recitativo')
            self.AlignmentBeatLevelIDs.append(847)
            # Don-Giovanni_Act-1_Scene-15_N11-Aria
            self.Y_names.append('Don-Giovanni_Act-1_Scene-15_N11-Aria')
            self.AlignmentBeatLevelIDs.append(798)
            # Don-Giovanni_Act-1_Scenes-16-and-17_Recitativo-1
            self.Y_names.append('Don-Giovanni_Act-1_Scenes-16-and-17_Recitativo-1')
            self.AlignmentBeatLevelIDs.append(849)
            # Don-Giovanni_Act-1_Scenes-16-and-17_N12-Aria
            self.Y_names.append('Don-Giovanni_Act-1_Scenes-16-and-17_N12-Aria')
            self.AlignmentBeatLevelIDs.append(851)
            # Don-Giovanni_Act-1_Scenes-16-and-17_Recitativo-2
            self.Y_names.append('Don-Giovanni_Act-1_Scenes-16-and-17_Recitativo-2')
            self.AlignmentBeatLevelIDs.append(853)
            # Don-Giovanni_Act-1_Scenes-16-and-17_N13-Finale-+-Scena-17
            self.Y_names.append('Don-Giovanni_Act-1_Scenes-16-and-17_N13-Finale-+-Scena-17')
            self.AlignmentBeatLevelIDs.append(855)
            # Don-Giovanni_Act-1_Scenes-18-and-19_
            self.Y_names.append('Don-Giovanni_Act-1_Scenes-18-and-19_')
            self.AlignmentBeatLevelIDs.append(857)
            # Don-Giovanni_Act-1_Scene-20_Allegro
            self.Y_names.append('Don-Giovanni_Act-1_Scene-20_Allegro')
            self.AlignmentBeatLevelIDs.append(859)
        elif 'Act 2' in self.target:
            # Don-Giovanni_Act-2_Scene-1_N14-Duetto
            self.Y_names.append('Don-Giovanni_Act-2_Scene-1_N14-Duetto')
            self.AlignmentBeatLevelIDs.append(889)
            # Don-Giovanni_Act-2_Scene-1_Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-1_Recitativo')
            self.AlignmentBeatLevelIDs.append(892)
            # Don-Giovanni_Act-2_Scene-2_N15-Terzetto
            self.Y_names.append('Don-Giovanni_Act-2_Scene-2_N15-Terzetto')
            self.AlignmentBeatLevelIDs.append(895)
            # Don-Giovanni_Act-2_Scene-2_Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-2_Recitativo')
            self.AlignmentBeatLevelIDs.append(898)
            # Don-Giovanni_Act-2_Scene-3_Recitativo-1
            self.Y_names.append('Don-Giovanni_Act-2_Scene-3_Recitativo-1')
            self.AlignmentBeatLevelIDs.append(901)
            # Don-Giovanni_Act-2_Scene-3_N16-Canzonetta
            self.Y_names.append('Don-Giovanni_Act-2_Scene-3_N16-Canzonetta')
            self.AlignmentBeatLevelIDs.append(800)
            # Don-Giovanni_Act-2_Scene-3_Recitativo-2
            self.Y_names.append('Don-Giovanni_Act-2_Scene-3_Recitativo-2')
            self.AlignmentBeatLevelIDs.append(904)
            # Don-Giovanni_Act-2_Scene-4_
            self.Y_names.append('Don-Giovanni_Act-2_Scene-4_')
            self.AlignmentBeatLevelIDs.append(907)
            # Don-Giovanni_Act-2_Scene-4_N17-Aria
            self.Y_names.append('Don-Giovanni_Act-2_Scene-4_N17-Aria')
            self.AlignmentBeatLevelIDs.append(910)
            # Don-Giovanni_Act-2_Scene-5_Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-5_Recitativo')
            self.AlignmentBeatLevelIDs.append(913)
            # Don-Giovanni_Act-2_Scene-6_Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-6_Recitativo')
            self.AlignmentBeatLevelIDs.append(916)
            # Don-Giovanni_Act-2_Scene-6_N18-Aria
            self.Y_names.append('Don-Giovanni_Act-2_Scene-6_N18-Aria')
            self.AlignmentBeatLevelIDs.append(919)
            # Don-Giovanni_Act-2_Scene-7_Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-7_Recitativo')
            self.AlignmentBeatLevelIDs.append(922)
            # Don-Giovanni_Act-2_Scene-7_N19-Sestetto
            self.Y_names.append('Don-Giovanni_Act-2_Scene-7_N19-Sestetto')
            self.AlignmentBeatLevelIDs.append(925)
            # Don-Giovanni_Act-2_Scene-8_
            self.Y_names.append('Don-Giovanni_Act-2_Scene-8_')
            self.AlignmentBeatLevelIDs.append(928)
            # Don-Giovanni_Act-2_Scene-9_Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-9_Recitativo')
            self.AlignmentBeatLevelIDs.append(931)
            # Don-Giovanni_Act-2_Scene-9_N20-Aria
            self.Y_names.append('Don-Giovanni_Act-2_Scene-9_N20-Aria')
            self.AlignmentBeatLevelIDs.append(934)
            # Don-Giovanni_Act-2_Scene-10_Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-10_Recitativo')
            self.AlignmentBeatLevelIDs.append(937)
            # Don-Giovanni_Act-2_Scene-10_N21-Aria
            self.Y_names.append('Don-Giovanni_Act-2_Scene-10_N21-Aria')
            self.AlignmentBeatLevelIDs.append(802)
            # Don-Giovanni_Act-2_Scene-10_N21-Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-10_N21-Recitativo')
            self.AlignmentBeatLevelIDs.append(940)
            # Don-Giovanni_Act-2_Scene-10_Aria
            self.Y_names.append('Don-Giovanni_Act-2_Scene-10_Aria')
            self.AlignmentBeatLevelIDs.append(943)
            # Don-Giovanni_Act-2_Scene-11_Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-11_Recitativo')
            self.AlignmentBeatLevelIDs.append(946)
            # Don-Giovanni_Act-2_Scene-11_N22-Duetto
            self.Y_names.append('Don-Giovanni_Act-2_Scene-11_N22-Duetto')
            self.AlignmentBeatLevelIDs.append(949)
            # Don-Giovanni_Act-2_Scene-12_Recitativo-1-+-N23-Recitativo
            self.Y_names.append('Don-Giovanni_Act-2_Scene-12_Recitativo-1-+-N23-Recitativo')
            self.AlignmentBeatLevelIDs.append(952)
            # Don-Giovanni_Act-2_Scene-12_Recitativo-2
            self.Y_names.append('Don-Giovanni_Act-2_Scene-12_Recitativo-2')
            self.AlignmentBeatLevelIDs.append(1080)
            # Don-Giovanni_Act-2_Scenes-13-and-14_N24-Finale-+-Scena-14
            self.Y_names.append('Don-Giovanni_Act-2_Scenes-13-and-14_N24-Finale-+-Scena-14')
            self.AlignmentBeatLevelIDs.append(955)
            # Don-Giovanni_Act-2_Scene-15_
            self.Y_names.append('Don-Giovanni_Act-2_Scene-15_')
            self.AlignmentBeatLevelIDs.append(958)
            # Don-Giovanni_Act-2_Final-Scene_
            self.Y_names.append('Don-Giovanni_Act-2_Final-Scene_')
            self.AlignmentBeatLevelIDs.append(961)

        #############
        #   Audio   #
        #############

        # Loading feature files and start indexes of songs
        self.Y_dir = './audio/DonGiovanni/Karajan/audio_features/{}/'.format(self.feature.lower())
        self.Y = np.empty((0, 100), dtype='float32')
        self.lengths_Y = [0]
        for Y_name in self.Y_names:
            for file in os.listdir(self.Y_dir):
                if '{}_{}.npy'.format(Y_name, self.feature.lower()) in file:
                    Y_song = np.load(self.Y_dir + file).astype(np.float32)
            self.Y = np.concatenate((self.Y, Y_song), axis=0)
            self.lengths_Y.append(self.lengths_Y[-1] + Y_song.shape[0]) # Cumulative lengths
        self.Y += 1e-10

        # Loading annotation indexes, pages, and bar areas
        self.time_new_bar = [] # bar annotations
        self.time_new_page = [] # page times
        self.points = [] # area coordinates
        self.pics = [] # pdf pages

        # Load subtitles file, languages
        self.subtitles_file = pd.read_excel('./lyrics/DonGiovanni/WSO_Don_Giovanni.xlsx', header=None).to_numpy()
        self.list_languages = list_languages()
        self.nb_languages = len(self.list_languages)
        self.subtitles_labels = [[] for l in range(self.nb_languages)] # labels
        self.subtitles_times = [] # times

        for i in range(len(self.AlignmentBeatLevelIDs)):
            ##############
            #  Pictures  #
            ##############

            # List PDF Score pages of the target
            self.pdf_scores = []
            for file in os.listdir('./scores/DonGiovanni/'):
                if self.Y_names[i] + '-' in file:
                    self.pdf_scores.append('./scores/DonGiovanni/' + file)
            self.pdf_scores.sort()

            # Load pictures and extract sizes
            self.pics_local = []
            self.sizes = []
            for score in self.pdf_scores:
                pic = QtGui.QImage(score).scaled(600, 800, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                self.pics_local.append(pic)
                self.sizes.append([pic.width(), pic.height()])

            # Load area file containing coordinates
            self.ymlfile = './areas/{}.yml'.format(self.AlignmentBeatLevelIDs[i])
            with open(self.ymlfile, 'r') as f:
                self.coordinates = yaml.load(f, Loader=yaml.FullLoader)

            # Load reference bar times
            self.time_new_bar_local = np.load('./annotations/{}_times.npy'.format(self.AlignmentBeatLevelIDs[i]))
            if self.Y_names[i] == 'Don-Giovanni_Act-1_Scene-5_Recitativo': # Special case
                self.time_new_bar_local = np.concatenate(([0], self.time_new_bar_local))
            if i == 0:
                self.time_new_bar_local[0] = 0 # Init first song

            # Load reference page times
            self.time_new_page_local = np.zeros(self.coordinates[-1]['page'], dtype='int')
            # create a counter for bars which require 2 ares
            count=0
            for c in range(len(self.coordinates)):
                # Check weird areas
                if self.coordinates[c]['beats'][0] == self.coordinates[c]['beats'][1]:
                    count += 1
                # Adding time of a new page
                if c == 0:
                    self.time_new_page_local[0] = self.time_new_bar_local[c]
                elif self.coordinates[c]['page'] != self.coordinates[c-1]['page']:
                    self.time_new_page_local[self.coordinates[c]['page'] - 1] = self.time_new_bar_local[c-count]

            # Load points from coordinates
            self.points_local = []
            for c in range(len(self.coordinates)):
                if c != 0 and self.coordinates[c-1]['beats'][0] == self.coordinates[c-1]['beats'][1]:
                    new_points = [
                            QtCore.QPoint(self.coordinates[c]['topLeft'][0] * self.sizes[self.coordinates[c]['page']-1][0],
                                        self.coordinates[c]['topLeft'][1] * self.sizes[self.coordinates[c]['page']-1][1]),
                            QtCore.QPoint(self.coordinates[c]['topRight'][0] * self.sizes[self.coordinates[c]['page']-1][0],
                                        self.coordinates[c]['topRight'][1] * self.sizes[self.coordinates[c]['page']-1][1]),
                            QtCore.QPoint(self.coordinates[c]['bottomRight'][0] * self.sizes[self.coordinates[c]['page']-1][0],
                                        self.coordinates[c]['bottomRight'][1] * self.sizes[self.coordinates[c]['page']-1][1]),
                            QtCore.QPoint(self.coordinates[c]['bottomLeft'][0] * self.sizes[self.coordinates[c]['page']-1][0],
                                        self.coordinates[c]['bottomLeft'][1] * self.sizes[self.coordinates[c]['page']-1][1])
                            ]
                    self.points_local[-1] = [self.points_local[-1], new_points]
                else:
                    self.points_local.append([
                            QtCore.QPoint(self.coordinates[c]['topLeft'][0] * self.sizes[self.coordinates[c]['page']-1][0],
                                        self.coordinates[c]['topLeft'][1] * self.sizes[self.coordinates[c]['page']-1][1]),
                            QtCore.QPoint(self.coordinates[c]['topRight'][0] * self.sizes[self.coordinates[c]['page']-1][0],
                                        self.coordinates[c]['topRight'][1] * self.sizes[self.coordinates[c]['page']-1][1]),
                            QtCore.QPoint(self.coordinates[c]['bottomRight'][0] * self.sizes[self.coordinates[c]['page']-1][0],
                                        self.coordinates[c]['bottomRight'][1] * self.sizes[self.coordinates[c]['page']-1][1]),
                            QtCore.QPoint(self.coordinates[c]['bottomLeft'][0] * self.sizes[self.coordinates[c]['page']-1][0],
                                        self.coordinates[c]['bottomLeft'][1] * self.sizes[self.coordinates[c]['page']-1][1])
                            ])

            # Check if number of areas corresponds to numbers of bar annotations
            if len(self.time_new_bar_local) != len(self.points_local):
                print('Different numbers of bars and areas!')
                print('Nb bars:', len(self.time_new_bar_local))
                print('Nb areas:', len(self.points_local))
                sys.exit()

            # Check if number of pages corresponds to numbers of page annotations
            if len(self.time_new_page_local) != len(self.pics_local):
                print('Different numbers of pages and annotations!')
                print('Nb pages:', len(self.pics_local))
                print('Nb areas:', len(self.time_new_page_local))
                sys.exit()

            # Update bar, page, area, and pdf lists
            self.time_new_bar = np.concatenate((self.time_new_bar, self.time_new_bar_local + self.lengths_Y[i]))
            self.time_new_page = np.concatenate((self.time_new_page, self.time_new_page_local + self.lengths_Y[i]))
            self.points = self.points + self.points_local
            self.pics = self.pics + self.pics_local


            #############
            # Subtitles #
            #############
            # Load annotations
            for file in os.listdir('./lyrics/DonGiovanni/'):
                if self.Y_names[i] + '.lab' in file:
                    self.annot_file = file
            self.lyrics_annot = pd.read_csv('./lyrics/DonGiovanni/' + self.annot_file, sep='\t', names=['time', 'line']).to_numpy()

            # Add empty subtitle at start of each part
            if (self.lyrics_annot.shape[0]==0 or self.lyrics_annot[0, 0] != 0):
                self.subtitles_times.append(self.lengths_Y[i])
                for l in range(self.nb_languages):
                    self.subtitles_labels[l].append('')

            if self.lyrics_annot.shape[0] !=0: # for non-instrumental parts
                for t in range(self.lyrics_annot.shape[0]):
                    if t==0 or int(self.lyrics_annot[t, 1]!=self.lyrics_annot[t-1, 1]+1): # Reduce annotations to paragraphs
                        self.subtitles_times.append(round(self.lyrics_annot[t, 0]*100) + self.lengths_Y[i])
                        for l in range(self.nb_languages):
                            self.label = str(self.subtitles_file[int(self.lyrics_annot[t, 1])-1, l])
                            self.label_counter = 0
                            self.label_next = str(self.subtitles_file[int(self.lyrics_annot[t, 1]+self.label_counter), l])
                            while not self.label_next.isspace() and self.label_next != 'nan':
                                self.label += '\n' + str(self.subtitles_file[int(self.lyrics_annot[t, 1]+self.label_counter), l])
                                self.label_counter += 1
                                self.label_next = str(self.subtitles_file[int(self.lyrics_annot[t, 1]+self.label_counter), l])
                            self.subtitles_labels[l].append(self.label)


        ######################
        # Part segmentation  #
        ######################

        # Extracting part index changes
        self.partFile_Y = './annotations/parts_DonGiovanni_Karajan2.json'
        with open(self.partFile_Y, encoding='utf-8', errors='ignore') as json_data:
            if 'Act 1' in self.target:
                self.partData_Y = json.load(json_data, strict=False)['Audio']['Act1']
            elif 'Act 2' in self.target:
                self.partData_Y = json.load(json_data, strict=False)['Audio']['Act2']
        self.part_names_Y = []
        for n in self.partData_Y.items():
            self.part_names_Y.append(n[0])
        self.lengths_Yr = np.zeros((1), dtype='int')

        # Adding part indexes
        for idx_part in self.parts_Y:
            # Retrieving global start and end indexes
            global_framestart, global_frameend = self.part_retrieval(self.part_names_Y, idx_part, self.Y_names, self.partData_Y, self.lengths_Y)
            if idx_part == self.parts_Y[0]:
                global_framestart = 0
            elif idx_part == self.parts_Y[-1]:
                global_frameend = self.Y.shape[0]
            self.lengths_Yr = np.concatenate((self.lengths_Yr, [self.lengths_Yr[-1] + global_frameend - global_framestart]))
        self.lengths_Y = self.lengths_Yr

        # Low resolution (computed by concatenation of HR variables)
        self.win_LR = np.expand_dims(np.hanning(60).astype('float32'), axis=0)
        self.Y_LR = convolve(self.Y.T, self.win_LR)[:, ::30]
        self.Y_LR = self.Y_LR.T
        self.lengths_Y_LR = self.lengths_Y/30
        self.lengths_Y_LR = self.lengths_Y_LR.astype(int)
        self.lengths_Y_LR[-1] = self.Y_LR.shape[0]


    # Given a part id, retrieving global start and end indexes
    def part_retrieval(self, part_names, idx_part, names, partData, cumul_length):
        # Part name given an idx_part
        part_name = [item for item in part_names if item.startswith(idx_part + '_')][0]
        # Corresponding part index in part_names list
        part_index = part_names.index(part_name)
        # Audio file name
        part_audioname = partData[part_name]['audio_file']
        # Name and idx in names
        names_name = [item for item in names if item + '.wav' in part_audioname][0]
        names_idx = names.index(names_name)

        # Frame start
        part_timestart = partData[part_name]['t_start']
        part_secstart = sum(x * float(t) for x, t in zip([1, 60, 3600], reversed(part_timestart.split(":"))))
        part_framestart = round(part_secstart * 100) # hop size: 10ms
        global_framestart = cumul_length[names_idx] + part_framestart

        # Frame end
        if part_name != part_names[-1]: # if we are NOT dealing with the last part
            # Next part name, part audio name, and corresponding name and idx in names
            next_part_name = part_names[part_index+1]
            next_part_audioname = partData[next_part_name]['audio_file']
            next_names_name = [item for item in names if item + '.wav' in next_part_audioname][0]
            next_names_idx = names.index(next_names_name)
            part_timeend = partData[next_part_name]['t_start']
            part_secend = sum(x * float(t) for x, t in zip([1, 60, 3600], reversed(part_timeend.split(":"))))
            part_frameend = round(part_secend * 100) # hop size: 10ms
            global_frameend = cumul_length[next_names_idx] + part_frameend
        else: # if we are dealing with the last part
            part_frameend = cumul_length[-1]
            global_frameend = cumul_length[-1]

        return global_framestart, global_frameend



def list_languages():
    languages = []
    languages.append('Deutsch')
    languages.append('English')
    languages.append('Italiano')
    languages.append('Français')
    languages.append('日本語')
    languages.append('Русский язык')
    languages.append('普通话')
    languages.append('Español')
    return languages
