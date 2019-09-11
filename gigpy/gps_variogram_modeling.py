#! /usr/bin/env python
#################################################################
###  This program is part of GigPy  v1.0                      ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################

import numpy as np
import os
import sys  
import subprocess
import getopt
import time
import glob
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import random

import pykrige
from pykrige import OrdinaryKriging
from pykrige import variogram_models
from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr

from gigpy import _utils as ut
#import matlab.engine
#######################################################
#residual_variogram_dict = {'linear': variogram_models.linear_variogram_model_residual,
#                      'power': variogram_models.power_variogram_model_residual,
#                      'gaussian': variogram_models.gaussian_variogram_model_residual,
#                      'spherical': variogram_models.spherical_variogram_model_residual,
#                      'exponential': variogram_models.exponential_variogram_model_residual,
#                      'hole-effect': variogram_models.hole_effect_variogram_model_residual}


variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

def unit_length(STR0,length=5):
    STR = STR0
    if len(STR0) > 5:
        STR = STR0[0:5]
    elif len(STR0) ==1:
        STR = STR0 + '.000'
    elif len(STR0) ==2:
        STR = STR0 + '.00'  
    elif len(STR0) ==3:
        STR = STR0 + '00' 
    elif len(STR0) ==4:
        STR = STR0 + '0'
    
    return STR


def unit_length0(STR0):
    if float(STR0)==0:
        STR = STR0 + '0'
    else:
        STR = unit_length4(STR0)
    
    return STR

def unit_length4(STR0):
    STR = STR0
    if len(STR0) > 4:
        STR = STR0[0:4]
    elif len(STR0) ==3:
        STR = STR0 + '0'
    
    return STR

def read_hdf5(fname, datasetName=None, box=None):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        data = f[datasetName][:]
        atr = dict(f.attrs)
        
    return data, atr

def remove_outlier_variogram(semivariance, lag):

    semi0 = np.asarray(semivariance,dtype = np.float32)
    lag0 = np.asarray(lag,dtype = np.float32)
    N = len(semi0)
    
    remove = []
    for i in range(N-3):
        k0 = semi0[i]
        c0 = semi0[(i+1):(i+4)]
        #print(c0)
        c00 = np.mean(c0)

        if k0 > c00:
            remove.append(i)
    
    semi00 = []
    lag00 = []
    for i in range(N):
        if i not in remove:
            semi00.append(semi0[i])
            lag00.append(lag0[i])

    semi00 = np.asarray(semi00,dtype=np.float32)
    lag00 = np.asarray(lag00,dtype=np.float32)
    return semi00, lag00

def write_gps_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):
    #output = 'variogramStack.h5'
    'lags                  1 x N '
    'semivariance          M x N '
    'sills                 M x 1 '
    'ranges                M x 1 '
    'nuggets               M x 1 '
    
    if os.path.isfile(out_file):
        print('delete exsited file: {}'.format(out_file))
        os.remove(out_file)

    print('create HDF5 file: {} with w mode'.format(out_file))
    dt = h5py.special_dtype(vlen=np.dtype('float64'))

    
    with h5py.File(out_file, 'w') as f:
        for dsName in datasetDict.keys():
            data = datasetDict[dsName]
            ds = f.create_dataset(dsName,
                              data=data,
                              compression=compression)
        
        for key, value in metadata.items():
            f.attrs[key] = str(value)
            #print(key + ': ' +  value)
    print('finished writing to {}'.format(out_file))
        
    return out_file  
 
#########################################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   Variogram model estimation of the GPS tropospheric measurements.
'''

EXAMPLE = '''
    Usage:
            gps_variogram_modeling.py gps_aps_variogram.h5 
            gps_variogram_modeling.py gps_aps_variogram.h5  --model gaussian
            gps_variogram_modeling.py gps_pwv_variogram.h5  --max-length 150 --model spherical

##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('input_file',help='input file name (e.g., gps_aps_variogram.h5).')
    parser.add_argument('-m','--model', dest='model', default='spherical',
                      help='variogram model used to fit the variance samples')
    parser.add_argument('--max-length', dest='max_length',type=float, metavar='NUM',
                      help='used bin ratio for mdeling the structure model.')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',
                      help='name of the output file')

    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    FILE = inps.input_file
    
    date,meta = read_hdf5(FILE, datasetName='date')
    variance_tzd = read_hdf5(FILE, datasetName='Semivariance')[0]
        
    semivariance = read_hdf5(FILE, datasetName='Semivariance')[0]
    
    Lags = read_hdf5(FILE, datasetName='Lags')[0]
    
    datasetNames = ut.get_dataNames(FILE)
    if 'Semivariance_wzd' in datasetNames:
        variance_wzd = read_hdf5(FILE, datasetName='Semivariance_wzd')[0]
        
    if inps.max_length:
        max_lag = inps.max_length
    else:
        max_lag = max(Lags[0,:]) + 0.001
    meta['max_length'] = max_lag
    r0 = np.asarray(1/2*max_lag)
    range0 = r0.tolist()
    
    datasetDict = dict()
    datasetDict['Lags'] = Lags
    
    if inps.out_file: OUT = os.path.out_file
    else: OUT = 'gps_delay_variogramModel.h5'

    #eng = matlab.engine.start_matlab()
    
    row,col = variance_tzd.shape
    model_parameters = np.zeros((row,4),dtype='float32')   # sill, range, nugget, Rs
    model_parameters_wzd = np.zeros((row,4),dtype='float32')   # sill, range, nugget, Rs
    
    def resi_func(m,d,y):
        variogram_function =variogram_dict[inps.model] 
        return  y - variogram_function(m,d)
    
    print('')
    print('Variogram model: %s' % inps.model)
    print('Maximum length: %s km' % inps.max_length)
    
    print('')
    print('  Date     Sill(cm^2)     Range(km)     Nugget(cm^2)      Corr')
    for i in range(row):
        
        lag = Lags[i,:]
        #print(Lags.shape)
        LL0 = lag[(lag >0) & (lag < max_lag)]
        #print(variance_tzd.shape)
        S0 = variance_tzd[i,:]
        SS0 = S0[(lag >0) & (lag < max_lag)]
        
        #print(LL0)
        #print(SS0)
        
        sill0 = max(SS0)
        sill0 = sill0.tolist()
        
        p0 = [sill0, range0, 0.0001]   
        #resi_func = residual_variogram_dict[inps.model]
        vari_func = variogram_dict[inps.model]
        
        SS01 = SS0.copy()
        LL01 = LL0.copy()
        #SS01, LL01 = remove_outlier_variogram(SS0, LL0)
        #print(LL01)
        #print(SS01)
        tt, _ = leastsq(resi_func,p0,args = (LL01,SS01))   
        corr, _ = pearsonr(SS01, vari_func(tt,LL01))
        if tt[2] < 0:
            tt[2] =0
        #LLm = matlab.double(LL0.tolist())
        #SSm = matlab.double(SS0.tolist())
        model_parameters[i,0:3] = tt
        #print(tt)
        model_parameters[i,3] = corr   
        date0 = date[i].astype(str)

        #LLm = matlab.double(LL01.tolist())
        #SSm = matlab.double(SS01.tolist())
       
        #tt = eng.variogramfit(LLm,SSm,range0,sill0,[],'nugget',0.00001,'model',inps.model)
        #model_parameters[i,:] = np.asarray(tt)
        #print(tt)
        print(date0 + '     ' + unit_length4(str(round(tt[0]*10000*2*100)/100)) + '           ' + unit_length(str(round(tt[1]*100)/100)) + '           ' + unit_length0(str(round(tt[2]*10000*2*100)/100)) + '          ' + unit_length(str(round(corr*1000)/1000)))
        
        if 'Semivariance_wzd' in datasetNames:
            S0 = variance_wzd[i,:]
            SS0 = S0[lag < max_lag]
            sill0 = max(SS0)

            p0 = [sill0, range0, 0.0001]   
            vari_func = variogram_dict[inps.model]
        
            tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))   
            corr, _ = pearsonr(SS0, vari_func(tt,LL0))

            model_parameters_wzd[i,0:3] = tt
            model_parameters_wzd[i,3] = corr

        
    meta['variogram_model'] = inps.model
    #meta['elevation_model'] = meta['elevation_model']    
    #del meta['model']
    
    
    datasetDict = dict()
    for dataName in datasetNames:
        datasetDict[dataName] = read_hdf5(FILE,datasetName=dataName)[0]
    
    datasetDict['tzd_variogram_parameter'] = model_parameters  
    if 'Semivariance_wzd' in datasetNames:
        datasetDict['wzd_variogram_parameter'] = model_parameters_wzd  
    #eng.quit()
    ut.write_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None) 
    
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
