#! /usr/bin/env python
#################################################################
###  This program is part of GigPy  v1.0                      ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################


import numpy as np
import getopt
import sys
import os
import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr

from gigpy import elevation_models
#### define dicts for model/residual/initial values ###############

model_dict = {'linear': elevation_models.linear_elevation_model,
                      'onn': elevation_models.onn_elevation_model,
                      'onn_linear': elevation_models.onn_linear_elevation_model,
                      'exp': elevation_models.exp_elevation_model,
                      'exp_linear': elevation_models.exp_linear_elevation_model}

residual_dict = {'linear': elevation_models.residuals_linear,
                      'onn': elevation_models.residuals_onn,
                      'onn_linear': elevation_models.residuals_onn_linear,
                      'exp': elevation_models.residuals_exp,
                      'exp_linear': elevation_models.residuals_exp_linear}

initial_dict = {'linear': elevation_models.initial_linear,
                      'onn': elevation_models.initial_onn,
                      'onn_linear': elevation_models.initial_onn_linear,
                      'exp': elevation_models.initial_exp,
                      'exp_linear': elevation_models.initial_exp_linear}

para_numb_dict = {'linear': 2,
                  'onn' : 3,
                  'onn_linear':4,
                  'exp':2,
                  'exp_linear':3}


def adjust_aps_lat_lon(gps_aps_h5,epoch = 0):
    
    FILE = gps_aps_h5
    
    gps_hei = read_hdf5(FILE,datasetName='gps_height')[0]
    gps_lat = read_hdf5(FILE,datasetName='gps_lat')[0]
    gps_lon = read_hdf5(FILE,datasetName='gps_lon')[0]
    gps_nm = read_hdf5(FILE,datasetName='gps_name')[0]
    gps_nm = list(gps_nm)

    date = read_hdf5(FILE,datasetName='date')[0]
    station = read_hdf5(FILE,datasetName='station')[0]
    
    station= list(station[epoch])

    
    k0 =9999
    for i in range(len(station)):
        if station[i].decode("utf-8")=='0.0':
            if i < k0:
                k0 =i
    station = station[0:k0]
    NN = len(station)
    
    hei = np.zeros((NN,))
    lat = np.zeros((NN,))
    lon = np.zeros((NN,))
    for i in range(NN):
        hei[i] = gps_hei[gps_nm.index(station[i])]
        lat[i] = gps_lat[gps_nm.index(station[i])]
        lon[i] = gps_lon[gps_nm.index(station[i])]
    
    return hei, lat, lon

def func_trend(lat,lon,p):
    a0,b0,c0,d0 = p
    return a0 + b0*lat + c0*lon +d0*lat*lon

def residual_trend(p,lat,lon,y0):
    a0,b0,c0,d0 = p 
    return y0 - func_trend(lat,lon,p)


def remove_ramp(lat,lon,data):
    # mod = a*x + b*y + c*x*y
    lat = lat/180*np.pi
    lon = lon/180*np.pi  
    lon = lon*np.cos(lat) # to get isometrics coordinates
    
    p0 = [0.0001,0.0001,0.0001,0.0000001]
    plsq = leastsq(residual_trend,p0,args = (lat,lon,data))
    para = plsq[0]
    data_trend = data - func_trend(lat,lon,para)
    corr, _ = pearsonr(data, func_trend(lat,lon,para))
    return data_trend, para, corr
    
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


def adjust_pwv(gps_pwv_h5,epoch = 0):
    
    FILE = gps_pwv_h5
    
    gps_hei = read_hdf5(FILE,datasetName='gps_height')[0]
    gps_nm = read_hdf5(FILE,datasetName='gps_name')[0]
    gps_nm = list(gps_nm)

    date = read_hdf5(FILE,datasetName='date')[0]
    station = read_hdf5(FILE,datasetName='station')[0]
    zwd = read_hdf5(FILE,datasetName='wzd')[0]
    tzd = read_hdf5(FILE,datasetName='tzd')[0]
    pwv = read_hdf5(FILE,datasetName='pwv')[0]
    station= list(station[epoch])
    zwd= list(zwd[epoch])
    pwv= list(pwv[epoch])
    tzd= list(tzd[epoch])
    
    k0 =9999
    for i in range(len(station)):
        if station[i].decode("utf-8")=='0.0':
            if i < k0:
                k0 ==i
    station = station[0:k0]
    
    zwd = zwd[0:k0]
    pwv = pwv[0:k0]       
    tzd = tzd[0:k0]
    
    hei = np.zeros((len(station),))
    for i in range(len(station)):
         hei[i] = gps_hei[gps_nm.index(station[i])]
            
    return station, hei, zwd, tzd, pwv

def adjust_aps(gps_aps_h5,epoch = 0):
    
    FILE = gps_aps_h5
    
    gps_hei = read_hdf5(FILE,datasetName='gps_height')[0]
    gps_nm = read_hdf5(FILE,datasetName='gps_name')[0]
    gps_nm = list(gps_nm)

    date = read_hdf5(FILE,datasetName='date')[0]
    station = read_hdf5(FILE,datasetName='station')[0]
    wzd = read_hdf5(FILE,datasetName='wzd')[0]
    tzd = read_hdf5(FILE,datasetName='tzd')[0]
    
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
    for i in range(NN):
         hei[i] = gps_hei[gps_nm.index(station[i])]
            
    return hei, wzd, tzd
            
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Download GPS data over SAR coverage.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('gps_file',help='gps_h5 file.')
    parser.add_argument('-m','--model', dest='model',default = 'onn_linear',choices= ['linear','onn','onn_linear','exp','exp_linear'],help='elevation models')
    parser.add_argument('-o', dest='out', help='output file.')
    
    inps = parser.parse_args()
    
    return inps


INTRODUCTION = '''
################################################################################################
    Modeling for the elevation-correlated tropospheric products: tropospheric delays, atmospehric water vapor ...    
    Supported models: Linear, Onn, Onn-linear, Exponential, Exponential-linear.
    
    See alse:   search_gps.py, download_gps_atm.py, extract_sar_atm.py, 
'''


EXAMPLE = '''EXAMPLES:
    elevation_correlation.py gps_aps.h5 -m onn
    elevation_correlation.py gps_pwv.h5 -m onn_linear 
    elevation_correlation.py gps_pwv.h5 -m exp -o gps_pwv_HgtCor.h5
    elevation_correlation.py gps_aps.h5 -m onn -o gps_aps_HgtCor.h5
################################################################################################
'''


def main(argv):
    
    inps = cmdLineParse()
    gps_h5 = inps.gps_file
    FILE = gps_h5
    
    if 'pwv' in gps_h5:
        OUT = 'gps_pwv_HgtCor.h5'
    elif 'aps' in gps_h5:
        OUT = 'gps_aps_HgtCor.h5'
    else:
        OUT = 'gps_trop_HgtCor.h5'
    
    if inps.out: OUT = inps.out
        
    model = inps.model
    date_list, meta =  read_hdf5(FILE,datasetName='date')
    tzd =  read_hdf5(FILE,datasetName='tzd')[0]
    
    wzd_model_parameters = np.zeros((len(date_list),para_numb_dict[model]),dtype = np.float32)
    tzd_model_parameters = np.zeros((len(date_list),para_numb_dict[model]),dtype = np.float32)
    
    wzd_trend_parameters = np.zeros((len(date_list),4),dtype = np.float32)
    tzd_trend_parameters = np.zeros((len(date_list),4),dtype = np.float32)
    
    tzd_turb_trend = np.zeros((np.shape(tzd)),dtype = np.float32)
    wzd_turb_trend = np.zeros((np.shape(tzd)),dtype = np.float32)
    
    tzd_turb = np.zeros((np.shape(tzd)),dtype = np.float32)
    wzd_turb = np.zeros((np.shape(tzd)),dtype = np.float32)
    
    initial_function = initial_dict[model]
    elevation_function = model_dict[model]
    residual_function = residual_dict[model]
    
    for i in range(len(date_list)):
        hei, wzd, tzd = adjust_aps(FILE,epoch = i)
        hei,lat,lon = adjust_aps_lat_lon(FILE,epoch = i)
        p0 = initial_function(hei,wzd)
        plsq = leastsq(residual_function,p0,args = (hei,wzd))
        wzd_model_parameters[i,:] = plsq[0]
        wzd_turb_trend[i,0:len(wzd)] = wzd - elevation_function(plsq[0],hei)
        corr_wzd, _ = pearsonr(wzd, elevation_function(plsq[0],hei))
        
        wzd_turb0,para0,corr_trend0 = remove_ramp(lat,lon,wzd_turb_trend[i,0:len(wzd)])
        wzd_turb[i,0:len(wzd)] = wzd_turb0
        wzd_trend_parameters[i,:] = para0
        
        p0 = initial_function(hei,tzd)
        plsq = leastsq(residual_function,p0,args = (hei,tzd))
        tzd_model_parameters[i,:] = plsq[0]
        tzd_turb_trend[i,0:len(wzd)] = tzd - elevation_function(plsq[0],hei)
        corr_tzd, _ = pearsonr(tzd, elevation_function(plsq[0],hei))
        
        tzd_turb0,para0,corr0 = remove_ramp(lat,lon,tzd_turb_trend[i,0:len(wzd)])
        tzd_turb[i,0:len(wzd)] = tzd_turb0
        tzd_trend_parameters[i,:] = para0
        
        print(str(date_list[i].decode("utf-8")) + ' : ' + str(round(corr_wzd*1000)/1000) + ' ' + str(round(corr_tzd*10000)/10000) + ' ' + str(round(corr_trend0*1000)/1000) + ' ' + str(round(corr0*1000)/1000))
        

    datasetNames =['date','gps_name','gps_lat','gps_lon','gps_height','hzd','wzd','tzd','station']
    datasetDict = dict()
    
  
    for dataName in datasetNames:
        datasetDict[dataName] = read_hdf5(FILE,datasetName=dataName)[0]
    
    datasetDict['tzd_turb_trend'] = tzd_turb_trend
    datasetDict['wzd_turb_trend'] = wzd_turb_trend    
        
    datasetDict['tzd_turb'] = tzd_turb
    datasetDict['wzd_turb'] = wzd_turb
    
    datasetDict['tzd_elevation_parameter'] = tzd_model_parameters
    datasetDict['wzd_elevation_parameter'] = wzd_model_parameters
    
    datasetDict['tzd_trend_parameter'] = tzd_trend_parameters
    datasetDict['wzd_trend_parameter'] = wzd_trend_parameters
    
    meta['elevation_model'] = model
    
    write_gps_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)
            
if __name__ == '__main__':
    main(sys.argv[1:])