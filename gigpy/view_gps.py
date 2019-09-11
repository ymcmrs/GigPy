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
import matplotlib.pyplot as plt
import h5py


from pykrige import variogram_models

#######################################################
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 12,
        }

def remove_outlier_variogram(semivariance, lag):
    
    semi0 = semivariance.copy()
    lag0 = lag.copy()
    N = len(semi0)
    
    remove = []
    for i in range(N-3):
        k0 = semivariance[i]
        c0 = semivariance[(i+1):(i+4)]
        #print(c0)
        c00 = np.mean(c0)
        
        if k0 > c00:
            remove.append(i)
        
    #print(len(remove))
    #print(remove)
    
    semi00 = []
    lag00 = []
    for i in range(N):
        if i not in remove:
            semi00.append(semi0[i])
            lag00.append(lag0[i])

    
    return semi00, lag00
    
def read_hdf5(fname, datasetName=None, box=None):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        data = f[datasetName][:]
        atr = dict(f.attrs)
        
    return data, atr

def adjust_aps_lat_lon(gps_aps_h5,epoch = 0):
    
    FILE = gps_aps_h5
    
    gps_hei = read_hdf5(FILE,datasetName='gps_height')[0]
    gps_lat = read_hdf5(FILE,datasetName='gps_lat')[0]
    gps_lon = read_hdf5(FILE,datasetName='gps_lon')[0]
    gps_nm = read_hdf5(FILE,datasetName='gps_name')[0]
    gps_nm = list(gps_nm)

    date = read_hdf5(FILE,datasetName='date')[0]
    station = read_hdf5(FILE,datasetName='station')[0]
    wzd = read_hdf5(FILE,datasetName='wzd_turb')[0]
    tzd = read_hdf5(FILE,datasetName='tzd_turb')[0]
    
    station= list(station[epoch])
    wzd= list(wzd[epoch])
    tzd= list(tzd[epoch])
    
    k0 =9999
    for i in range(len(station)):
        if station[i].decode("utf-8")=='0.0':
            if i < k0:
                k0 =i
    station = station[0:k0]
    wzd = wzd[0:k0]     
    tzd = tzd[0:k0]
    NN = len(station)
    
    hei = np.zeros((NN,))
    lat = np.zeros((NN,))
    lon = np.zeros((NN,))
    for i in range(NN):
        hei[i] = gps_hei[gps_nm.index(station[i])]
        lat[i] = gps_lat[gps_nm.index(station[i])]
        lon[i] = gps_lon[gps_nm.index(station[i])]
    tzd_turb = tzd
    wzd_turb = wzd
    
    return tzd_turb, wzd_turb, lat, lon


INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   View (i.e., plot) the GPS tropospheric products and estimations.
'''

EXAMPLE = '''
    Usage:
            view_gps.py gps_aps_variogram.h5 -n 2
            
    Examples:
            view_gps.py variogramStack.h5 -n 20 --compare
            view_gps.py variogramStack.h5 -n 20 --max-length 100.0 --compare
            view_gps.py variogramStack.h5 -n 20 --max-length 100.0 --data sar
##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='View the variogram of atmospheric delays in InSAR/TS-InSAR or SAR.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('input_file',help='name of the input file')
    parser.add_argument('-n','--dset-num', dest='dset_num',type=int,default=0, metavar='NUM',
                      help='number of dset to show')
    parser.add_argument('--max-length', dest='max_length',type=float,metavar='NUM',
                      help='maximum length to show')
    parser.add_argument('--compare', dest='compare_noweight', action='store_true',
                     help='compare the weighted results and the non-weighted results')
    parser.add_argument('--plot_model', dest='plot_model', action='store_true',
                     help='plot the models of the variogram')
    parser.add_argument('--data', dest='data',default='insar', choices={'insar','sar'},
                      help='selection of the APS variogram for InSAR or SAR')
    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    FILE = inps.input_file
    
    if inps.dset_num:
        Num = inps.dset_num
    else:
        Num = 0  
    
    datelist,meta = read_hdf5(FILE, datasetName='date')
    Date0 = datelist[Num]
    k0 = 1
    if meta['UNIT']=='m':
        k0 =100**2
    
    semivariance = read_hdf5(FILE, datasetName='Semivariance')[0]
    #print(semivariance[0,:]*k0)
    semivariance_trend = read_hdf5(FILE, datasetName='Semivariance_trend')[0]
    Lag = read_hdf5(FILE, datasetName='Lags')[0]
    y0 = semivariance[Num,:]*k0
    y0_trend = semivariance_trend[Num,:]*k0
    x0 = Lag[Num,:]
    
    semi0, lag0 = remove_outlier_variogram(y0, x0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot(x0, y0, 'ro',label='WNVCE-based variances',fillstyle='none')
    #print(x0)
    #print(y0)
    ax.plot(x0, y0, 'ro')
    ax.plot(x0, y0_trend, 'bo')
    #ax.plot(lag0, semi0, 'ko')
    ax.yaxis.grid()
        
    label = datelist[Num]    
    plt.title(label)
    plt.xlabel('Lag (km)', fontdict=font)
    plt.ylabel('Semi-variance (cm$^2$)', fontdict=font)
    plt.show()

    
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
