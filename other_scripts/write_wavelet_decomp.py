#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:44:40 2023

@author: apratimdey
"""

import numpy as np
from scipy.io import loadmat
import json
import pywt
import os


def write_wavelet_decomp(original_image, spectrumwise_wavelet_type, slicewise_wavelet_type, wavelet_level, filename):
    
    spectrum_wave_dat = np.zeros_like(original_image, dtype = float)
    
    for i in range(original_image.shape[1]):
      for j in range(original_image.shape[2]):
        cA, cD = pywt.dwt(original_image[:, i, j], wavelet = spectrumwise_wavelet_type, mode = 'periodization')
        spectrum_wave_dat[:, i, j] = np.concatenate((cA, cD))
    
    wave_dat = list()

    for slice_val in range(spectrum_wave_dat.shape[0]):
      spectrum_wave_slice = spectrum_wave_dat[slice_val, :, :]
      coeffs = pywt.wavedec2(spectrum_wave_slice, slicewise_wavelet_type, level = wavelet_level, mode = "periodization")
      wave_dat = wave_dat + [coeffs]
      
    wavelet_dict = {'approx': np.array([wave_dat[slice_val][0] for slice_val in range(len(wave_dat))]).tolist()}
    
    for band in range(1, wavelet_level + 1):
        for subband in range(3):
            pooled_coeffs = np.array([wave_dat[slice_val][band][subband] for slice_val in range(len(wave_dat))])
            wavelet_dict[f'band_{band}_subband_{subband}'] = pooled_coeffs.tolist()
            
    if not os.path.exists('hyperspectral_data/wavelet_decomp/'):
        os.makedirs('hyperspectral_data/wavelet_decomp/')
    
    with open(f'hyperspectral_data/wavelet_decomp/{filename}', 'w') as file:
        json.dump(wavelet_dict, file)
        
        
if __name__ == '__main__':
    wavelet_decomp_level = 3
    data = loadmat('hyperspectral_data/datasets/Indian_pines.mat')['indian_pines']
    original_image = np.transpose(data, (2, 0, 1))
    write_wavelet_decomp(original_image,
                         'haar',
                         'db2',
                         wavelet_decomp_level,
                         f'indian_pines_level_{wavelet_decomp_level}.json')

