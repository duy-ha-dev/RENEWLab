#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:19:21 2021

@author: red
"""

from pyargus.directionEstimation import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks



d = .5 # Inter element spacing [lambda]
#d = .03968 # Inter element spacing [lambda]

# Number of antenna elements
M = 8

# Generate ULA scanning vectors
array_alignment = np.arange(0, M, 1)* d
incident_angles= np.arange(0,181,.001)
scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)


# =============================================================================
# Add my code
# =============================================================================

#Samps
#complex_h = pd.read_csv('pole_exp_1_clean/goldline/h5/samps_complex_first_d.csv', index_col=[0])
#complex_h = pd.read_csv('pole_exp_1_clean/ten_yard/h5/samps_complex_first_d.csv', index_col=[0])
#complex_h = pd.read_csv('pole_exp_1_clean/twenty_yard/h5/samps_complex_first_d.csv', index_col=[0])

#complex_h = pd.read_csv('pole_exp_1_clean/goldline/h5/samps_complex_first.csv', index_col=[0])
#complex_h = pd.read_csv('pole_exp_1_clean/ten_yard/h5/samps_complex_first.csv', index_col=[0])
#complex_h = pd.read_csv('pole_exp_1_clean/twenty_yard/h5/samps_complex_first.csv', index_col=[0])

complex_h = pd.read_csv('sean_10_yard.csv', index_col=[0])

#H or CSI
#complex_h = pd.read_csv('sean_ten.csv', index_col=[0],skipinitialspace=True)

complex_h = complex_h.values
#complex_h = complex_h.T


# =============================================================================
# Even or odd
# =============================================================================
complex_h_chain_a = complex_h[::2]
complex_h_chain_b = complex_h[1::2]


def f(complex_h_chain_b):
    return np.complex(complex_h_chain_b)
f2 = np.vectorize(f)


# Taking chain A or B
rec_signal = complex_h_chain_b.astype(complex)
#rec_signal = complex_h_chain_a.astype(complex)
#rec_signal = complex_h.astype(complex)


# Polarization
Amp = abs(rec_signal)
# # +45 for chain A
# degrees1 = np.angle(rec_signal, deg=True)
# degrees = degrees1+45


# -45 for chain B
degrees1 = np.angle(rec_signal, deg=True)
degrees = degrees1-45

x1 = Amp*np.cos(degrees)
y1 = Amp*np.sin(degrees)




# First row
rec_signal = rec_signal[:8,:]


# # 45 shift
# z = complex(1,1)
# z = z/np.sqrt(2)
# rec_signal = z * rec_signal


# =============================================================================
# DOA estimation module
# =============================================================================

# Estimating the spatial correlation matrix
R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")

# Estimate DOA 
Bartlett = DOA_Bartlett(R, scanning_vectors)
Capon = DOA_Capon(R, scanning_vectors)
MEM = DOA_MEM(R, scanning_vectors, column_select = 1) 
LPM = DOA_LPM(R, scanning_vectors, element_select = 0)
MUSIC = DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)


peaks, _ = find_peaks(Bartlett, height=0)
print('Angles for Bartlett: ', peaks/1000)

peaks, _ = find_peaks(MUSIC, height=0)
print('Angles for Music: ', peaks/1000)

peaks, _ = find_peaks(LPM, height=0)
print('Angles for LPM: ', peaks/1000)

peaks, _ = find_peaks(MEM, height=0)
print('Angles for MEM: ', peaks/1000)

peaks, _ = find_peaks(Capon, height=0)
print('Angles for Capon: ', peaks/1000)


# Get matplotlib axes object
axes = plt.axes()

# # Plot results on the same fiugre
DOA_plot(Bartlett, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(Capon, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(MEM, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(LPM, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(MUSIC, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)

axes.legend(("Bartlett","Capon","MEM","LPM","MUSIC"))
plt.show()

