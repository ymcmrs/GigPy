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
from pykrige import OrdinaryKriging
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

#######################################################
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
    
    wzd_trend = read_hdf5(FILE,datasetName='wzd_turb_trend')[0]
    tzd_trend = read_hdf5(FILE,datasetName='tzd_turb_trend')[0]
    
    station= list(station[epoch])
    wzd= list(wzd[epoch])
    tzd= list(tzd[epoch])
    
    wzd_trend= list(wzd_trend[epoch])
    tzd_trend= list(tzd_trend[epoch])
    
    k0 =9999
    for i in range(len(station)):
        if station[i].decode("utf-8")=='0.0':
            if i < k0:
                k0 =i
    station = station[0:k0]
    wzd = wzd[0:k0]     
    tzd = tzd[0:k0]
    
    wzd_trend = wzd_trend[0:k0]     
    tzd_trend = tzd_trend[0:k0]
    
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
    
    return tzd_turb, wzd_turb, tzd_trend, wzd_trend, lat, lon

def remove_numb(x,y,z,numb=0):
    
    z = np.asarray(z,dtype=np.float32)
    sort_z = sorted(list(np.abs(z)))
    k0 = sort_z[len(z)-numb-1] + 0.0001
    
    fg = np.where(abs(z)<k0)
    fg = np.asarray(fg,dtype=int)
    
    x0 = x[fg]
    y0 = y[fg]
    z0 = z[fg]
    
    return x0, y0, z0


def latlon2dis(lat1,lon1,lat2,lon2,R=6371):
    lat1 = lat1/180*np.pi
    lat2 = lat2/180*np.pi
    lon1 = lon1/180*np.pi
    lon2 = lon2/180*np.pi
    dist = 6371.01 * np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lon1 - lon2))
    return dist


#########################################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   Estimate the variance components of the turbulent tropospheric measurements.
'''

EXAMPLE = '''
    Usage:
            variogram_gps.py gps_aps_HgtCor.h5
            variogram_gps.py gps_pwv_HgtCor.h5 --remove_numb 5 --bin_numb 60
            variogram_gps.py gps_pwv_HgtCor.h5 --remove_numb 5             
            variogram_gps.py gps_aps_HgtCor.h5 --bin_numb 60

##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('gps_file',help='input gps file name (ifgramStack.h5 or timeseires.h5).')
    parser.add_argument('--bin_numb', dest='bin_numb',type=int,default=30, metavar='NUM',
                      help='number of bins used to fit the variogram model')
    parser.add_argument('--remove_numb', dest='remove_numb',type=int,default=0, metavar='NUM',
                      help='remove the largest data for variogram estimation.')
    parser.add_argument('-o','--out_file', dest='out', metavar='FILE',
                      help='name of the output file')

    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    gps_h5 = inps.gps_file
    FILE = gps_h5
    R = 6371     # Radius of the Earth (km)
    if 'pwv' in gps_h5:
        OUT = 'gps_pwv_variogram.h5'
    elif 'aps' in gps_h5:
        OUT = 'gps_aps_variogram.h5'
    else:
        OUT = 'gps_trop_variogram.h5'
    if inps.out: OUT = inps.out
    BIN_NUMB = inps.bin_numb
    
    date_list, meta =  read_hdf5(FILE,datasetName='date')
    tzd_turb =  read_hdf5(FILE,datasetName='tzd_turb')[0]
    wzd_turb =  read_hdf5(FILE,datasetName='wzd_turb')[0]
    
    tzd_turb_trend =  read_hdf5(FILE,datasetName='tzd_turb_trend')[0]
    wzd_turb_trend =  read_hdf5(FILE,datasetName='wzd_turb_trend')[0]
    
    Lag = np.zeros((len(date_list),BIN_NUMB),dtype = np.float32)
    Variance = np.zeros((len(date_list),BIN_NUMB),dtype = np.float32)
    Variance_trend = np.zeros((len(date_list),BIN_NUMB),dtype = np.float32)
    
    Variance_wzd = np.zeros((len(date_list),BIN_NUMB),dtype = np.float32)
    Variance_wzd_trend = np.zeros((len(date_list),BIN_NUMB),dtype = np.float32)
    
    for i in range(len(date_list)):
        tzd_turb, wzd_turb, tzd_turb_trend, wzd_turb_trend, lat, lon = adjust_aps_lat_lon(FILE,epoch = i)
        lat = np.asarray(lat,dtype = np.float32)
        lon = np.asarray(lon,dtype = np.float32)

        lat0, lon0, tzd_turb0 = remove_numb(lat,lon,tzd_turb,numb=inps.remove_numb)
        #uk = OrdinaryKriging(lon, lat, tzd_turb, coordinates_type = 'geographic', nlags=BIN_NUMB)
        uk = OrdinaryKriging(lon0, lat0, tzd_turb0, coordinates_type = 'geographic', nlags=BIN_NUMB)
        #print(len((uk.lags)))
        Lags = (uk.lags)/180*np.pi*R
        Semivariance = 2*(uk.semivariance)
        
        lat0, lon0, tzd_turb_trend0 = remove_numb(lat,lon,tzd_turb_trend,numb=inps.remove_numb)
        uk = OrdinaryKriging(lon0, lat0, tzd_turb_trend0, coordinates_type = 'geographic', nlags=BIN_NUMB)
        Semivariance_trend = 2*(uk.semivariance)
        
        Lag[i,:] = Lags
        Variance[i,:] = Semivariance
        Variance_trend[i,:] = Semivariance_trend
        
        
        lat0, lon0, wzd_turb0 = remove_numb(lat,lon,wzd_turb,numb=inps.remove_numb)
        uk = OrdinaryKriging(lon0, lat0, wzd_turb0, coordinates_type = 'geographic', nlags=BIN_NUMB)
        Semivariance_wzd = 2*(uk.semivariance)
        
        lat0, lon0, wzd_turb_trend0 = remove_numb(lat,lon,wzd_turb_trend,numb=inps.remove_numb)
        uk = OrdinaryKriging(lon0, lat0, wzd_turb_trend0, coordinates_type = 'geographic', nlags=BIN_NUMB)
        Semivariance_wzd_trend = 2*(uk.semivariance)
        
        Variance_wzd[i,:] = Semivariance
        Variance_wzd_trend[i,:] = Semivariance_trend

        
    datasetNames = ['date','gps_name','gps_lat','gps_lon','gps_height','hzd','wzd','tzd','wzd_turb_trend','tzd_turb_trend','wzd_turb','tzd_turb','station','tzd_elevation_parameter', 'wzd_elevation_parameter','tzd_trend_parameter', 'wzd_trend_parameter']
    
    datasetDict = dict()
    meta['remove_numb'] = inps.remove_numb
    
    for dataName in datasetNames:
        datasetDict[dataName] = read_hdf5(FILE,datasetName=dataName)[0]
    
    datasetDict['Lags'] = Lag
    datasetDict['Semivariance'] = Variance
    datasetDict['Semivariance_trend'] = Variance_trend
    
    datasetDict['Semivariance_wzd'] = Variance_wzd
    datasetDict['Semivariance_wzd_trend'] = Variance_wzd_trend
    
    write_gps_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)    
    
    
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
