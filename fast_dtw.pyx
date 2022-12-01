# python fast_dtw_setup.py build_ext --inplace

import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sqrt, acos, INFINITY, abs


# Calculate distance between a vector X and a list of vectors Y
def dist_VecToMat(np.ndarray[float, ndim=1] X,
        np.ndarray[float, ndim=2] Y,
        ):

    cdef int i
    cdef int j

    cdef int Y_x = Y.shape[0]
    cdef int Y_y = Y.shape[1]

    cdef float num
    cdef float normY
    cdef float normX = 0
    cdef float normDist = 0

    cdef np.ndarray[double, ndim=1] dist = np.empty(Y_y, dtype=np.float64)

    for j in range(Y_x):
        normX += X[j] * X[j]

    for i in range(Y_y):
        num = 0
        normY = 0
        for j in range(Y_x):
            num += Y[j, i] * X[j]
            normY += Y[j, i] * Y[j, i]
        dist[i] = acos(num / sqrt(normX * normY))
        normDist += dist[i] * dist[i]

    for i in range(Y_y):
        dist[i] /= sqrt(normDist)

    return dist


def costVec(int middle_inf,
                int middle_sup,
                np.ndarray[double, ndim=2] gamma,
                np.ndarray[double, ndim=1] dist,
                int i,
                ):

    cdef int idx
    cdef int j

    cdef float norm_cost
    cdef float cost_pos = INFINITY

    cdef int gamma_x = gamma.shape[0]
    cdef int gamma_y = gamma.shape[1]

    cdef np.ndarray[double, ndim=1] cost_vec = np.ones(gamma_y, dtype=np.float64) * INFINITY


    for idx, j in enumerate(range(middle_inf, middle_sup)):
        gamma[1,j+1] = dist[idx] + min(gamma[0, j], gamma[0, j+1], gamma[1, j])

        cost_vec[j] = gamma[1, j+1] / (i+j+1)

        if cost_vec[j] < cost_pos:
            cost_pos = cost_vec[j]

    gamma[0, :] = gamma[1, :]
    gamma[1, :] = (gamma[1, :] + 1e-10) * INFINITY

    return gamma, cost_vec


def dist_VecToMat_nonorm(np.ndarray[float, ndim=1] X,
        np.ndarray[float, ndim=2] Y,
        ):

    cdef int i
    cdef int j

    cdef int Y_x = Y.shape[0]
    cdef int Y_y = Y.shape[1]

    cdef float num
    cdef float normY
    cdef float normX = 0
    cdef float normDist = 0

    cdef np.ndarray[double, ndim=1] dist = np.empty(Y_y, dtype=np.float64)

    for j in range(Y_x):
        normX += X[j] * X[j]

    for i in range(Y_y):
        num = 0
        normY = 0
        for j in range(Y_x):
            num += Y[j, i] * X[j]
            normY += Y[j, i] * Y[j, i]
        dist[i] = acos(num / sqrt(normX * normY))
        normDist += dist[i] * dist[i]

    # for i in range(Y_y):
    #     dist[i] /= sqrt(normDist)

    return dist


def costVec_jump2(int middle_inf,
                int middle_sup,
                np.ndarray[double, ndim=2] gamma,
                np.ndarray[double, ndim=1] dist_state,
                int i,
                bint jump, # check if jump
                np.ndarray[double, ndim=2] dist_jump_list, # vector of dists
                np.ndarray[int, ndim=1] s_list, # vector of start indexes
                np.ndarray[int, ndim=1] t_list, # vector of end indexes
                int state_idx, # current state
                int prev_norm, # cumulative norm
                int size_vec, # size_vec
                float penalty
                ):

    cdef int idx
    cdef int ids
    cdef int idj

    cdef int j
    cdef int s 

    cdef int gamma_x = gamma.shape[0]
    cdef int gamma_y = gamma.shape[1]

    cdef np.ndarray[double, ndim=1] cost_vec = np.ones(gamma_y, dtype=np.float64) * INFINITY

    for idx, j in enumerate(range(middle_inf, middle_sup)):
        gamma[1,j+1] = dist_state[idx] + min(gamma[0, j], gamma[0, j+1], gamma[1, j])
        if i+j+1 - s_list[state_idx] + prev_norm == 0:
            cost_vec[j+1] = INFINITY
        else:
            cost_vec[j+1] = gamma[1, j+1] / (i+j+1 - s_list[state_idx] + prev_norm)

    if jump:
        # for all beginnings
        for ids, s in enumerate(s_list):
            # if the beginning is not in [middle_inf, middle_sup[
            if not (middle_inf <= s < middle_sup):
                for idj, j in enumerate(range(s, s + size_vec)):
                    if idj == 0:
                        # gamma[1,j+1] = dist_jump_list[ids][idj] + gamma[0, t_list[state_idx]] + 0.5 * t_list[state_idx] / 100
                        gamma[1,j+1] = dist_jump_list[ids][idj] + gamma[0, t_list[state_idx]] + 0.5 * (prev_norm+t_list[state_idx]-s_list[state_idx]) * penalty
                    else:
                        gamma[1,j+1] = dist_jump_list[ids][idj] + min(gamma[0, j], gamma[0, j+1], gamma[1, j])
                    if i+j+1 - s + (t_list[state_idx] - s_list[state_idx]) + prev_norm == 0:
                        cost_vec[j+1] = INFINITY
                    else:
                        cost_vec[j+1] = gamma[1, j+1] / (i+j+1 - s + (t_list[state_idx] - s_list[state_idx]) + prev_norm)


    gamma[0, :] = gamma[1, :]
    gamma[1, :] = (gamma[1, :] + 1e-10) * INFINITY

    return gamma, cost_vec


def costVec_jump2test(int middle_inf,
                int middle_sup,
                np.ndarray[double, ndim=2] gamma,
                np.ndarray[double, ndim=1] dist_state,
                int checking_parts,
                int i,
                bint jump, # check if jump
                np.ndarray[double, ndim=2] dist_jump_list, # vector of dists
                np.ndarray[int, ndim=1] s_list, # vector of start indexes
                np.ndarray[int, ndim=1] t_list, # vector of end indexes
                int state_idx, # current state
                int prev_norm, # cumulative norm
                int size_vec, # size_vec
                float penalty
                ):

    cdef int idx
    cdef int ids
    cdef int idj

    cdef int j
    cdef int s 

    cdef int gamma_x = gamma.shape[0]
    cdef int gamma_y = gamma.shape[1]

    cdef np.ndarray[double, ndim=1] cost_vec = np.ones(gamma_y, dtype=np.float64) * INFINITY

    for idx, j in enumerate(range(middle_inf, middle_sup)):
        gamma[1,j+1] = dist_state[idx] + min(gamma[0, j], gamma[0, j+1], gamma[1, j])
        if i+j+1 - s_list[state_idx] + prev_norm == 0:
            cost_vec[j+1] = INFINITY
        else:
            cost_vec[j+1] = gamma[1, j+1] / (i+j+1 - s_list[state_idx] + prev_norm)

    if jump:
        # for all beginnings
        for ids in range(state_idx, min(len(s_list)-1, state_idx+checking_parts)):
            # if the beginning is not in [middle_inf, middle_sup[
            if not (middle_inf <= s_list[ids] < middle_sup):
                for idj, j in enumerate(range(s_list[ids], s_list[ids] + size_vec)):
                    if idj == 0:
                        # gamma[1,j+1] = dist_jump_list[ids][idj] + gamma[0, t_list[state_idx]] + 0.5 * t_list[state_idx] / 100
                        gamma[1,j+1] = dist_jump_list[ids][idj] + gamma[0, t_list[state_idx]] + 0.5 * (prev_norm+t_list[state_idx]-s_list[state_idx]) * penalty
                    else:
                        gamma[1,j+1] = dist_jump_list[ids][idj] + min(gamma[0, j], gamma[0, j+1], gamma[1, j])
                    if i+j+1 - s_list[ids] + (t_list[state_idx] - s_list[state_idx]) + prev_norm == 0:
                        cost_vec[j+1] = INFINITY
                    else:
                        cost_vec[j+1] = gamma[1, j+1] / (i+j+1 - s_list[ids] + (t_list[state_idx] - s_list[state_idx]) + prev_norm)


    gamma[0, :] = gamma[1, :]
    gamma[1, :] = (gamma[1, :] + 1e-10) * INFINITY

    return gamma, cost_vec


def diagonal_matching(int middle_inf,
                int middle_sup,
                np.ndarray[double, ndim=2] gamma,
                np.ndarray[double, ndim=1] dist,
                ):

    cdef int j

    for j in range(middle_inf, middle_sup):
        gamma[1,j] = dist[j] + gamma[0, j-1]

    gamma[0, :] = gamma[1, :]
    gamma[1, :] = (gamma[1, :] + 1e-10) * INFINITY

    return gamma


def diagonal_matching_jump(int middle_inf,
                int middle_sup,
                np.ndarray[double, ndim=2] gamma,
                np.ndarray[double, ndim=1] dist,
                np.ndarray[int, ndim=1] s_list, # vector of start indexes
                np.ndarray[int, ndim=1] t_list, # vector of end indexes
                ):

    cdef int start = middle_inf
    cdef int j
    cdef int e
    cdef int t

    cdef int len_t_list = t_list.shape[0]
    cdef float gamma_end_min = INFINITY

    # Cut into segments in between the t_list
    # Way faster than: if j not in t_list
    for e in range(len_t_list):
        # Normal DTW
        for j in range(start, t_list[e]+1):
            gamma[1,j] = dist[j] + gamma[0, j-1]

        # Jump DTW
        j = t_list[e]+1
        if j < middle_sup:
            # Jump DTW
            gamma_end_min = INFINITY
            for t in range(len_t_list):
                if gamma[0, t_list[t]] < gamma_end_min:
                    gamma_end_min = gamma[0, t_list[t]]
            gamma[1,j] = dist[j] + min(gamma[0, j-1], gamma_end_min)
            start = j + 1


    gamma[0, :] = gamma[1, :]
    gamma[1, :] = (gamma[1, :] + 1e-10) * INFINITY

    return gamma


def cumulative_matrix_jump(int middle_inf,
                int middle_sup,
                np.ndarray[double, ndim=2] gamma,
                np.ndarray[double, ndim=1] dist,
                np.ndarray[int, ndim=1] s_list, # vector of start indexes
                np.ndarray[int, ndim=1] t_list, # vector of end indexes
                ):

    cdef int start = middle_inf
    cdef int j
    cdef int e
    cdef int t

    cdef int len_t_list = t_list.shape[0]
    cdef float gamma_end_min = INFINITY

    # Cut into segments in between the t_list
    # Way faster than: if j not in t_list
    for e in range(len_t_list):
        # Normal DTW
        for j in range(start, t_list[e]+1):
            gamma[1,j] = dist[j] + min(gamma[0, j-1], gamma[0, j], gamma[1, j-1])

        # Jump DTW
        j = t_list[e]+1
        if j < middle_sup:
            # Jump DTW
            gamma_end_min = INFINITY
            for t in range(len_t_list):
                if gamma[0, t_list[t]] < gamma_end_min:
                    gamma_end_min = gamma[0, t_list[t]]
            gamma[1,j] = dist[j] + min(gamma[0, j-1], gamma[0, j], gamma[1, j-1], gamma_end_min)
            start = j + 1


    gamma[0, :] = gamma[1, :]
    gamma[1, :] = (gamma[1, :] + 1e-10) * INFINITY

    return gamma


    