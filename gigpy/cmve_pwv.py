#! /usr/bin/env python
#################################################################
###  This program is part of GigPy  v1.0                      ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : Central South University (CSU) & KAUST.          ###   
#################################################################

import random
import matplotlib.pyplot as plt
import sys
import os
import re
import subprocess
import argparse
import numpy as np
import h5py
from gigpy import elevation_models

from numpy.linalg import inv
from scipy import linalg

from pykrige import variogram_models
from pykrige import OrdinaryKriging

from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from gigpy import _utils as ut0

from mintpy.utils import (ptime,
                          readfile,
                          utils as ut,
                          plot as pp)


from mintpy.objects import (
    datasetUnitDict,
    geometry,
    geometryDatasetNames,
    giantIfgramStack,
    giantTimeseries,
    ifgramDatasetNames,
    ifgramStack,
    timeseriesDatasetNames,
    timeseries,
    HDFEOS
)
###############################################################


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


variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

def geo2sar(lat0, lon0, latitude_sar, longitude_sar):
    
    m0 = np.nanmean(longitude_sar)
    if (lon0 - m0) > 180.0:
        longitude_sar = longitude_sar + 360.0
    
    dd = np.sqrt((latitude_sar-float(lat0))**2+(longitude_sar-float(lon0))**2)
    where_are_NaNs = np.isnan(dd)
    dd[where_are_NaNs] = 9999999.0
    row0, col0 = np.where(dd==dd.min())
    
    return row0[0], col0[0]

def get_sarCoord(lat_list,lon_list,latitude_sar,longitude_sar):
    
    row_sar = []
    col_sar = []
    
    for i in range(len(lat_list)):
        row0,col0 = geo2sar(lat_list[i], lon_list[i], latitude_sar, longitude_sar)
        row_sar.append(row0)
        col_sar.append(col0)
    
    #row_sar = np.array(row_sar)
    #row_sar = row_sar.flatten()
    
    #col_sar = np.array(col_sar)
    #col_sar = col_sar.flatten()
    return row_sar, col_sar

def get_common_station(station1, lat1, lon1, hei1, wzd1, station2, lat2, lon2, hei2, wzd2):
    
    station2 = np.array(station2)
    station2 = list(station2.flatten())
    #print(station2)
    N1 = len(station1)
    
    station = []
    lat = []
    lon = []
    hei = []
    wzd = []
    wzdS = []
    
    fg1 = []
    fg2 = []
    
    for i in range(N1):
        if station1[i] in station2:
            fg1.append(i)
            fg2.append(station2.index(station1[i]))
            station.append(station1[i])
            lat.append(lat1[i])
            lon.append(lon1[i])
            hei.append(hei1[i])
            wzd.append(wzd1[i])
            wzdS.append(wzd2[station2.index(station1[i])])
    
    fg1 = np.array(fg1)
    fg2 = np.array(fg2)
    fg1 = fg1.flatten()
    fg2 = fg2.flatten()
    
    return station, lat, lon, hei, wzd, wzdS, fg1, fg2
    

def spherical(dist,para):
    mm = para[2]+para[0]*(dist/para[1]*3/2 -1/2*((dist/para[1])**3))
    mm[dist>para[1]] = para[0] + para[2]
    
    return mm
    
    
def dry_orb_sar(lat,lon,hei,para):
    # mod = a*x + b*y + c*x*y
    lat0 = lat/180.0*np.pi
    lon00 = lon/180.0*np.pi  
    lon0 = lon00*np.cos(lat0) # to get isometrics coordinates
    
    a0,b0,c0,d0 = para
    dry_orb_sar0 = a0 + b0*lat0 + c0*lon0 + d0*hei
    
    return dry_orb_sar0


def residual_trend(para,lat,lon,hei,y0):
    a0,b0,c0,d0 = para 
    return y0 - dry_orb_sar(lat,lon,hei, para)

def model_dry_orb(lat_gps, lon_gps, hei_sar, diff_insar_dgps):
    
    lat_gps = np.array(lat_gps)
    lon_gps = np.array(lon_gps)
    hei_sar = np.array(hei_sar)
    diff_insar_dgps = np.array(diff_insar_dgps)
    
    
    p0 = [0.0001,0.0001,0.0001,0.0001]
    plsq = leastsq(residual_trend,p0,args = (lat_gps, lon_gps, hei_sar, diff_insar_dgps))
    para = plsq[0]
    plsq = leastsq(residual_trend, para, args = (lat_gps, lon_gps, hei_sar, diff_insar_dgps))
    para = plsq[0]
    plsq = leastsq(residual_trend, para, args = (lat_gps, lon_gps, hei_sar, diff_insar_dgps))
    para = plsq[0]
    dryOrb_sar = dry_orb_sar(lat_gps, lon_gps, hei_sar, para)
    corr, _ = pearsonr(diff_insar_dgps, dry_orb_sar(lat_gps,lon_gps,hei_sar,para))
    
    return dryOrb_sar, para, corr
    

def dry_orb_sar_new(X,a0, b0, c0, d0):
    lat,lon,hei = X
    # mod = a*x + b*y + c*x*y    
    dry_orb_sar0 = a0 + b0*lat + c0*lon + d0*hei
    
    return dry_orb_sar0    
    
def model_dry_orb_new(lat_gps, lon_gps, hei_sar, diff_insar_dgps):
    
    lat_gps = np.array(lat_gps)
    lon_gps = np.array(lon_gps)
    hei_sar = np.array(hei_sar)
    diff_insar_dgps = np.array(diff_insar_dgps)
    
    # mod = a*x + b*y + c*x*y
    lat0 = lat_gps/180.0*np.pi
    lon00 = lon_gps/180.0*np.pi  
    lon0 = lon00*np.cos(lat0) # to get isometrics coordinates
    
    
    lat_gps = lat0
    lon_gps = lon0
    
    X = (lat_gps, lon_gps, hei_sar)
    diff_insar_dgps = np.array(diff_insar_dgps)
    
    
    p0 = [0.1, 0.01, 0.01, 0.01]
    
    popt, pcov = curve_fit(dry_orb_sar_new, (lat_gps,lon_gps,hei_sar), diff_insar_dgps, p0)
    #print(popt)
    
    a0, b0, c0, d0 = popt
    #plsq = leastsq(residual_trend_new,p0,args = (lat_gps, lon_gps, hei_sar, diff_insar_dgps))
    #para = plsq[0]
    ##print(para)
    #plsq = leastsq(residual_trend_new, para, args = (lat_gps, lon_gps, hei_sar, diff_insar_dgps))
    #para = plsq[0]
    #print(para)
    #plsq = leastsq(residual_trend_new, para, args = (lat_gps, lon_gps, hei_sar, diff_insar_dgps))
    #para = plsq[0]
    #print(para)
    dryOrb_sar = dry_orb_sar_new(X, a0, b0, c0, d0)
    corr, _ = pearsonr(diff_insar_dgps, dryOrb_sar)
    
    return dryOrb_sar, popt, corr    
    

def remove_numb(x,y,z,numb=0):
    
    z = np.asarray(z,dtype=np.float32)
    sort_z = sorted(list(np.abs(z)))
    k0 = sort_z[len(z)-numb-1] + 0.0001
    
    fg = np.where(abs(z)<k0)
    fg = np.asarray(fg,dtype=int)
    
    x0 = x[fg]
    y0 = y[fg]
    z0 = z[fg]
    
    return x0, y0, z0, fg

def get_dataNames(FILE):
    with h5py.File(FILE, 'r') as f:
        dataNames = []
        for k0 in f.keys():
            dataNames.append(k0)
    return dataNames


# read_hdf5_file is from mintpy
def read_hdf5_file(fname, datasetName=None, box=None):
    """
    Parameters: fname : str, name of HDF5 file to read
                datasetName : str or list of str, dataset name in root level with/without date info
                    'timeseries'
                    'timeseries-20150215'
                    'unwrapPhase'
                    'unwrapPhase-20150215_20150227'
                    'HDFEOS/GRIDS/timeseries/observation/displacement'
                    'recons'
                    'recons-20150215'
                    ['recons-20150215', 'recons-20150227', ...]
                    '20150215'
                    'cmask'
                    'igram-20150215_20150227'
                    ...
                box : 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
    Returns:    data : 2D/3D array
                atr : dict, metadata
    """
    # File Info: list of slice / dataset / dataset2d / dataset3d
    slice_list = get_slice_list(fname)
    ds_list = []
    for i in [i.split('-')[0] for i in slice_list]:
        if i not in ds_list:
            ds_list.append(i)
    ds_2d_list = [i for i in slice_list if '-' not in i]
    ds_3d_list = [i for i in ds_list if i not in ds_2d_list]

    # Input Argument: convert input datasetName into list of slice
    if not datasetName:
        datasetName = [ds_list[0]]
    elif isinstance(datasetName, str):
        datasetName = [datasetName]
    if all(i.isdigit() for i in datasetName):
        datasetName = ['{}-{}'.format(ds_3d_list[0], i) for i in datasetName]
    # Input Argument: decompose slice list into dsFamily and inputDateList
    dsFamily = datasetName[0].split('-')[0]
    inputDateList = [i.replace(dsFamily,'').replace('-','') for i in datasetName]

    # read hdf5
    with h5py.File(fname, 'r') as f:
        # get dataset object
        dsNames = [i for i in [datasetName[0], dsFamily] if i in f.keys()]
        dsNamesOld = [i for i in slice_list if '/{}'.format(datasetName[0]) in i] # support for old mintpy files
        if len(dsNames) > 0:
            ds = f[dsNames[0]]
        elif len(dsNamesOld) > 0:
            ds = f[dsNamesOld[0]]
        else:
            raise ValueError('input dataset {} not found in file {}'.format(datasetName, fname))
        
        # 2D dataset        
        if ds.ndim == 2:
            #data = ds[box[1]:box[3], box[0]:box[2]]
            data = ds
 
        # 3D dataset
        elif ds.ndim == 3:
            # define flag matrix for index in time domain
            slice_flag = np.zeros((ds.shape[0]), dtype=np.bool_)
            if not inputDateList or inputDateList == ['']:
                slice_flag[:] = True
            else:
                date_list = [i.split('-')[1] for i in 
                             [j for j in slice_list if j.startswith(dsFamily)]]
                for d in inputDateList:
                    slice_flag[date_list.index(d)] = True

            # read data
            #data = ds[slice_flag, box[1]:box[3], box[0]:box[2]]
            data = ds[slice_flag,:,:]
            data = np.squeeze(data)
        
        elif ds.ndim==1:
            data = ds[:]
        
    return data

# get_slice_list is from mintpy
def get_slice_list(fname):
    """Get list of 2D slice existed in file (for display)"""
    fbase, fext = os.path.splitext(os.path.basename(fname))
    fext = fext.lower()
    atr = read_attribute(fname)
    k = atr['FILE_TYPE']

    global slice_list
    # HDF5 Files
    if fext in ['.h5', '.he5']:
        with h5py.File(fname, 'r') as f:
            d1_list = [i for i in f.keys() if isinstance(f[i], h5py.Dataset)]
        if k == 'timeseries' and k in d1_list:
            obj = timeseries(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['geometry'] and k not in d1_list:
            obj = geometry(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['ifgramStack']:
            obj = ifgramStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['HDFEOS']:
            obj = HDFEOS(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['giantTimeseries']:
            obj = giantTimeseries(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['giantIfgramStack']:
            obj = giantIfgramStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        else:
            ## Find slice by walking through the file structure
            length, width = int(atr['LENGTH']), int(atr['WIDTH'])
            def get_hdf5_2d_dataset(name, obj):
                global slice_list
                if isinstance(obj, h5py.Dataset) and obj.shape[-2:] == (length, width):
                    if obj.ndim == 2:
                        slice_list.append(name)
                    else:
                        warnings.warn('file has un-defined {}D dataset: {}'.format(obj.ndim, name))
            slice_list = []
            with h5py.File(fname, 'r') as f:
                f.visititems(get_hdf5_2d_dataset)

    # Binary Files
    else:
        if fext.lower() in ['.trans', '.utm_to_rdc']:
            slice_list = ['rangeCoord', 'azimuthCoord']
        elif fbase.startswith('los'):
            slice_list = ['incidenceAngle', 'azimuthAngle']
        elif atr.get('number_bands', '1') == '2' and 'unw' not in k:
            slice_list = ['band1', 'band2']
        else:
            slice_list = ['']
    return slice_list

def read_hdf5(fname, datasetName=None, box=None):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        data = f[datasetName][:]
        atr = dict(f.attrs)
        
    return data, atr

def read_attr(fname):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        atr = dict(f.attrs)
        
    return atr

def read_attribute(fname):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        atr = dict(f.attrs)
        
    return atr

def read_gps_unavco(gps_file,date):
    
    FILE = gps_file
    gps_hei = read_hdf5(FILE,datasetName='gps_height')[0]
    gps_lat = read_hdf5(FILE,datasetName='gps_lat')[0]
    gps_lon = read_hdf5(FILE,datasetName='gps_lon')[0]
    gps_nm = read_hdf5(FILE,datasetName='gps_name')[0]
    gps_nm = list(gps_nm)

    date_list = read_hdf5(FILE,datasetName='date')[0]
    
    for i in range(len(date_list)):
        if date_list[i].decode("utf-8")==date:
            k_flag = i
    
    station = read_hdf5(FILE,datasetName='station')[0]
    station= list(station[k_flag])
    
    k0 =9999
    for i in range(len(station)):
        if station[i].decode("utf-8")=='0.0':
            if i < k0:
                k0 =i
    
    wzd = read_hdf5(FILE,datasetName='wzd')[0]
    wzd_turb = read_hdf5(FILE,datasetName='wzd_turb')[0]
    
    station = station[0:k0]
    wzd = wzd[:,0:k0]
    wzd_turb = wzd_turb[:,0:k0]
    
    NN = len(station)
    
    hei = np.zeros((NN,))
    lat = np.zeros((NN,))
    lon = np.zeros((NN,))
    
    for i in range(NN):
        hei[i] = gps_hei[gps_nm.index(station[i])]
        lat[i] = gps_lat[gps_nm.index(station[i])]
        lon[i] = gps_lon[gps_nm.index(station[i])]
    
    
    elevation_para0 = read_hdf5(FILE,datasetName='wzd_elevation_parameter')[0]
    trend_para0 = read_hdf5(FILE,datasetName='wzd_trend_parameter')[0]
    variogram_para0 = read_hdf5(FILE,datasetName='wzd_variogram_parameter')[0]
    
    wzd = wzd[k_flag,:]
    wzd_turb = wzd_turb[k_flag,:]
    elevation_para = elevation_para0[k_flag,:]
    trend_para = trend_para0[k_flag,:]
    variogram_para = variogram_para0[k_flag,:]
    
    return wzd, wzd_turb, elevation_para, trend_para, variogram_para, station, lat, lon, hei


def split_list(list_length, processors = 4):
    
    nn = list_length
    dn = round(nn/int(processors))
    
    idx = []
    for i in range(processors):
        a0 = i*dn
        b0 = (i+1)*dn
        
        if i == (processors - 1):
            b0 = nn
        
        if not a0 > b0:
            idx0 = np.arange(a0,b0)
            #print(idx0)
            idx.append(idx0)
            
    return idx

def latlon2dis(lat1,lon1,lat2,lon2,R=6371):
    
    #lat1 = lat1.reshape((1,-1))
    #lon1 = lon1.reshape((1,-1))
    
    lat2 = lat2.reshape((-1,1))
    lon2 = lon2.reshape((-1,1))
    
    lat1 = np.array(lat1)*np.pi/180.0
    lat2 = np.array(lat2)*np.pi/180.0
    dlon = (lon1-lon2)*np.pi/180.0

    # Evaluate trigonometric functions that need to be evaluated more
    # than once:
    c1 = np.cos(lat1)
    s1 = np.sin(lat1)
    c2 = np.cos(lat2)
    s2 = np.sin(lat2)
    cd = np.cos(dlon)

    # This uses the arctan version of the great-circle distance function
    # from en.wikipedia.org/wiki/Great-circle_distance for increased
    # numerical stability.
    # Formula can be obtained from [2] combining eqns. (14)-(16)
    # for spherical geometry (f=0).

    dist =  R*np.arctan2(np.sqrt((c2*np.sin(dlon))**2 + (c1*s2-s1*c2*cd)**2), s1*s2+c1*c2*cd)
    s = dist
    kk = sorted(range(len(s)), key=lambda k: s[k])
    lat22 = lat2[kk]
    lon22 = lon2[kk]
    
    return dist, kk

def latlon2dis_matrix(lat1,lon1,lat2,lon2,R=6371):
    
    lat1 = lat1.reshape((1,-1))
    lon1 = lon1.reshape((1,-1))
    
    lat2 = lat2.reshape((-1,1))
    lon2 = lon2.reshape((-1,1))
    
    lat1 = np.array(lat1)*np.pi/180.0
    lat2 = np.array(lat2)*np.pi/180.0
    dlon = (lon1-lon2)*np.pi/180.0
    #print(dlon)

    # Evaluate trigonometric functions that need to be evaluated more
    # than once:
    c1 = np.cos(lat1)
    s1 = np.sin(lat1)
    c2 = np.cos(lat2)
    s2 = np.sin(lat2)
    cd = np.cos(dlon)

    # This uses the arctan version of the great-circle distance function
    # from en.wikipedia.org/wiki/Great-circle_distance for increased
    # numerical stability.
    # Formula can be obtained from [2] combining eqns. (14)-(16)
    # for spherical geometry (f=0).

    dist =  R*np.arctan2(np.sqrt((c2*np.sin(dlon))**2 + (c1*s2-s1*c2*cd)**2), s1*s2+c1*c2*cd)
    #s = dist
    #kk = sorted(range(len(s)), key=lambda k: s[k])
    
    return dist

def func_ramp(X,a0,b0,c0,d0):
    lat, lon, lalo = X
    return a0 + b0*lat + c0*lon +d0*lalo

def residual_ramp(p,lat,lon,y0):
    a0,b0,c0,d0 = p 
    return y0 - func_ramp(lat,lon,p)

def remove_ramp(lat,lon,data):
    # mod = a*x + b*y + c*x*y    
    lat = lat/180*np.pi
    lon = lon/180*np.pi  
    lon = lon*np.cos(lat) # to get isometrics coordinates
    
    lat = np.array(lat)
    lon = np.array(lon)
    
    X = (lat, lon, np.array(lat*lon))
    
    data = np.array(data)
    
    p0 = [0.1, 0.01, 0.01, 0.01]
    popt, pcov = curve_fit(func_ramp, (lat, lon, lat*lon), data, p0)
    #print(popt)
    
    a0, b0, c0, d0 = popt
    data_trend = data - func_ramp(X,a0,b0,c0,d0)
    corr, _ = pearsonr(data, func_ramp(X,a0,b0,c0,d0))
    return data_trend, popt, corr


def space_elevation_model(lat, lon, hei, wzd, elevation_model ='onn_linear'):
    
        
    lat = np.array(lat)
    lon = np.array(lon)
    hei = np.array(hei)
    wzd = np.array(wzd)
    
    initial_function = initial_dict[elevation_model]
    elevation_function = model_dict[elevation_model]
    residual_function = residual_dict[elevation_model]
    
    p0 = initial_function(hei,wzd)
    plsq = leastsq(residual_function,p0,args = (hei,wzd))
    wzd_turb_trend = wzd - elevation_function(plsq[0],hei)
    corr_wzd, _ = pearsonr(wzd, elevation_function(plsq[0],hei))
    
    wzd_turb_trend = np.array(wzd_turb_trend)
    lat = lat.flatten()
    lon = lon.flatten()
    
    wzd_turb0,para0,corr_trend0 = remove_ramp(lat,lon,wzd_turb_trend)
    
    wzd_turb = wzd_turb0
    elevation_model_para = plsq[0]
    ramp_model_para = para0
    corr_elevation = corr_wzd
    corr_ramp = corr_trend0
    
    return wzd_turb, elevation_model_para, ramp_model_para, corr_elevation, corr_ramp

def space_variogram_model(lat, lon, wzd_turb,variogram_model = 'spherical', max_length = 150, bin_numb = 30, remove_outlier = 0):
    
    R = 6371
    lat = np.array(lat)
    lon = np.array(lon)
    wzd_turb = np.array(wzd_turb)
    
    lat0, lon0, wzd_turb0, fg0 = remove_numb(lat, lon, wzd_turb, numb = remove_outlier)
    uk = OrdinaryKriging(lat0, lon0, wzd_turb0, coordinates_type = 'geographic', nlags=bin_numb)
    #print(len((uk.lags)))
    Lags = (uk.lags)/180*np.pi*R
    Semivariance = 2*(uk.semivariance)
    
    LL0 = Lags[(Lags >0) & (Lags < max_length)]
    S0 = Semivariance
    SS0 = S0[(Lags >0) & (Lags < max_length)]
    
    r0 = np.asarray(1/2*max_length)
    sill0 = max(SS0)
    p0 = [sill0, r0, 0.0001]   
    
    vari_func = variogram_dict[variogram_model]
    
    def resi_func(m,d,y): 
        return  y - vari_func(m,d)
    
    tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))
    p0 = tt[0:3]
    tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))
    p0 = tt[0:3]
    tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))

    corr, _ = pearsonr(SS0, vari_func(tt,LL0))

    if tt[2] < 0:
        tt[2] =0
    
    para = tt[0:3]
    
    return Lags, Semivariance , para, corr

    
def pre_model_dry_orb(stationx, wzdxA, wzdxB, latx, lonx, heix, row_sar, col_sar, lat_sar, lon_sar, insar_hei, insar_obs, insar_coh, row, col, coh_threshold = '0.85'):
    
    latx1 = []
    lonx1 = []
    heix1 = []
    row_sar1 = []
    col_sar1 = []
    lat_sar1 = []
    lon_sar1 = []
    insar_hei1 = []
    insar_obs1 = []
    insar_coh1 = []
    wzdxA1 = []
    wzdxB1 = []
    stationx1 = []
    for i in range(len(latx)):
        if (row_sar[i] !=0 and col_sar[i]!=0 and row_sar[i]!= (row-1) and col_sar!= (col-1)) and (insar_coh[i] > float(coh_threshold)):
            latx1.append(latx[i])
            lonx1.append(lonx[i])
            heix1.append(heix[i])
            row_sar1.append(row_sar[i])
            col_sar1.append(col_sar[i])
            lat_sar1.append(lat_sar[i])
            lon_sar1.append(lon_sar[i])
            insar_hei1.append(insar_hei[i])
            insar_obs1.append(insar_obs[i])
            insar_coh1.append(insar_coh[i])
            wzdxA1.append(wzdxA[i])
            wzdxB1.append(wzdxB[i])
            stationx1.append(stationx[i])
      
    return stationx1, wzdxA1, wzdxB1, latx1, lonx1, heix1, row_sar1, col_sar1, lat_sar1, lon_sar1, insar_hei1, insar_obs1, insar_coh1


def kriging_interp(lat1, lon1, wzd_turb1, lat0, lon0, variogram_para, vari_func, nearestNumb):
    
    N1 = len(lat1)
    rr = N1+1;
    
    dist1, kk1 = latlon2dis(lat0,lon0, lat1, lon1)
    kk10 = kk1[0:int(nearestNumb)]
    
    N1 = len(kk10)
    rr = N1+1;
    
    dist11 = dist1[kk10]
    lat11 = lat1[kk10]
    lon11 = lon1[kk10]
    wzd_turb11 = wzd_turb1[kk10] 
    dist_matrix11 = latlon2dis_matrix(lat11,lon11,lat11,lon11,R=6371)
    
    V11 = spherical(dist_matrix11,variogram_para[0:3])
    VV1 = np.ones((1+N1,1+N1))
    VV1[0:N1,0:N1] = V11
    VV1[N1,N1] = 0
    
    L1 = spherical(dist11,variogram_para[0:3])
    LL = np.ones((rr,1))
    LL[0:N1] = L1
    
    xx = linalg.lstsq(VV1, LL, cond=1e-15)[0]
    w1 = xx[0:N1];
    
    w1 = w1.flatten()
    wzd_turb11 = wzd_turb11.flatten()
    
    wzd_x1 = sum(w1*wzd_turb11)
    
    return wzd_x1
     
    
def cmve(wzd_turb1, lat1, lon1, variogram_para1, wzd_turb2, lat2, lon2, variogram_para2, vari_fun, lat0, lon0, insar_wzd_turb0, nearest):
    
    N1 = len(lat1)
    N2 = len(lat2)
    rr = N1+1+N2+1+1;
    
    dist1, kk1 = latlon2dis(lat0,lon0, lat1, lon1)
    dist2, kk2 = latlon2dis(lat0,lon0, lat2, lon2)
    
    #if int(nearest) > len(kk1):
    #    kk10 = kk1
    #else:
    #    kk10 = kk1[0:int(nearest)]
    #    
    #if int(nearest) > len(kk2):
    #    kk20 = kk2
    #else:
    #    kk20 = kk2[0:int(nearest)]
    kk10 = kk1[0:int(nearest)]
    kk20 = kk2[0:int(nearest)]
        
    N1 = len(kk10)
    N2 = len(kk20)
    rr = N1+1+N2+1+1;
    
    #kk10 = kk1[0:int(nearest)]
    #kk20 = kk2[0:int(nearest)]
    
    dist11 = dist1[kk10]
    lat11 = lat1[kk10]
    lon11 = lon1[kk10]
    #print(wzd_turb1.shape)
    wzd_turb11 = wzd_turb1[kk10]
    
    dist22 = dist2[kk20]
    lat22 = lat2[kk20]
    lon22 = lon2[kk20]
    wzd_turb22 = wzd_turb2[kk20]
    
    dist_matrix11 = latlon2dis_matrix(lat11,lon11,lat11,lon11,R=6371)
    dist_matrix22 = latlon2dis_matrix(lat22,lon22,lat22,lon22,R=6371)
    
    #print(dist_matrix11)
    #print(variogram_para1)
    
    #print(float(variogram_para1[0]))
    V11 = spherical(dist_matrix11,variogram_para1[0:3])
    V22 = spherical(dist_matrix22,variogram_para2[0:3])
    #print(V11)
    
    VV1 = np.ones((1+N1,1+N1))
    VV2 = np.ones((1+N2,1+N2))
    
    AA = np.zeros((rr,rr))
    
    VV1[0:N1,0:N1] = V11
    VV2[0:N2,0:N2] = V22
   
    VV1[N1,N1] = 0
    VV2[N1,N1] = 0
    
    AA[0:(1+N1),0:(N1+1)] = VV1
    AA[(N1+1):(rr-1),(N1+1):(rr-1)] = VV2
    
    AA[rr-1,0:N1] = wzd_turb11
    AA[rr-1,(N1+1):(N1+N2+1)] = -wzd_turb22
    AA[0:(rr-1),(rr-1)]=1
    AA[0:N1,rr-1] = wzd_turb11
    AA[(N1+1):(N1+N2+1),rr-1] = -wzd_turb22
    
    
    L1 = spherical(dist11,variogram_para1[0:3])
    L2 = spherical(dist22,variogram_para2[0:3])
    LL = np.ones((rr,1))
    LL[0:N1] = L1
    LL[(N1+1):(rr-2)] = L2
    LL[rr-2] = 1
    LL[rr-1] = insar_wzd_turb0
    BB = inv(AA)
    xx = linalg.lstsq(AA, LL, cond=1e-15)[0]
    #print(xx)
    w1 = xx[0:N1];
    w2 = xx[(N1+1):(N1+1+N2)]
    
    w1 = w1.flatten()
    wzd_turb11 = wzd_turb11.flatten()
    
    w2 = w2.flatten()
    wzd_turb22 = wzd_turb22.flatten()
    
    wzd_x1 = sum(w1*wzd_turb11)
    wzd_x2 = sum(w2*wzd_turb22)

    return wzd_x1, wzd_x2

def kriging_matrix(lat1,lon1,wzd_turb1, variogram_para1,lat2,lon2,wzd_turb2, variogram_para2,vari_fun):
    
    N1 = len(lat1)
    N2 = len(lat2)
    rr = N1+1+N2+1+1;
    
    dist_matrix11 = latlon2dis_matrix(lat1,lon1,lat1,lon1,R=6371)
    dist_matrix22 = latlon2dis_matrix(lat2,lon2,lat2,lon2,R=6371)

    #print(dist_matrix11)
    #print(variogram_para1)
    
    #print(float(variogram_para1[0]))
    V11 = spherical(dist_matrix11,variogram_para1[0:3])
    V22 = spherical(dist_matrix22,variogram_para2[0:3])
    #print(V11)
    
    VV1 = np.ones((1+N1,1+N1))
    VV2 = np.ones((1+N2,1+N2))
    
    AA = np.zeros((rr,rr))
    
    VV1[0:N1,0:N1] = V11
    VV2[0:N2,0:N2] = V22
   
    VV1[N1,N1] = 0
    VV2[N2,N2] = 0
    
    AA[0:(1+N1),0:(N1+1)] = VV1
    AA[(N1+1):(rr-1),(N1+1):(rr-1)] = VV2
    
    AA[rr-1,0:N1] = wzd_turb1
    AA[rr-1,(N1+1):(N1+N2+1)] = -wzd_turb2
    AA[0:(rr-1),(rr-1)]=1
    AA[0:N1,rr-1] = wzd_turb1
    AA[(N1+1):(N1+N2+1),rr-1] = -wzd_turb2
    
    return AA
    

def cmve_all(AA, wzd_turb1, lat1, lon1, variogram_para1, wzd_turb2, lat2, lon2, variogram_para2, vari_fun, lat0, lon0, insar_wzd_turb0):
    
    N1 = len(lat1)
    N2 = len(lat2)
    rr = N1+1+N2+1+1;
    
    dist1, kk1 = latlon2dis(lat0,lon0, lat1, lon1)
    dist2, kk2 = latlon2dis(lat0,lon0, lat2, lon2)
    
    L1 = spherical(dist1,variogram_para1[0:3])
    L2 = spherical(dist2,variogram_para2[0:3])
    LL = np.ones((rr,1))
    LL[0:N1] = L1
    LL[(N1+1):(rr-2)] = L2
    LL[rr-2] = 1
    LL[rr-1] = insar_wzd_turb0
    BB = inv(AA)
    xx = linalg.lstsq(AA, LL, cond=1e-15)[0]
    #print(xx)
    w1 = xx[0:N1];
    w2 = xx[(N1+1):(N1+1+N2)]
    
    w1 = w1.flatten()
    wzd_turb1 = wzd_turb1.flatten()
    
    w2 = w2.flatten()
    wzd_turb2 = wzd_turb2.flatten()
    
    wzd_x1 = sum(w1*wzd_turb1)
    wzd_x2 = sum(w2*wzd_turb2)

    return wzd_x1, wzd_x2

def cmve_list(AA, wzd_turb1, lat1, lon1, variogram_para1, wzd_turb2, lat2, lon2, variogram_para2, vari_fun, lat_list0, lon_list0, insar_wzd_turb_list0, nearest):
    
    sar1 = []
    sar2 = []
    for i in range(len(lat_list0)):
        if nearest == 'all':
            ff1,ff2 = cmve_all(AA, wzd_turb1, lat1, lon1, variogram_para1, wzd_turb2, lat2, lon2, variogram_para2, vari_fun, lat_list0[i], lon_list0[i], insar_wzd_turb_list0[i])
        else:
            ff1,ff2 = cmve(wzd_turb1, lat1, lon1, variogram_para1, wzd_turb2, lat2, lon2, variogram_para2, vari_fun, lat_list0[i], lon_list0[i], insar_wzd_turb_list0[i], nearest)
            
        sar1.append(ff1)
        sar2.append(ff2)
    
    return sar1, sar2


def cmve_para(data0):
    
    AA, wzd_turb1, lat1, lon1, variogram_para1, wzd_turb2, lat2, lon2, variogram_para2, vari_fun, lat_list0, lon_list0, insar_wzd_turb_list0, nearest_numb = data0
    sar1,sar2 = cmve_list(AA, wzd_turb1, lat1, lon1, variogram_para1, wzd_turb2, lat2, lon2, variogram_para2, vari_fun, lat_list0, lon_list0, insar_wzd_turb_list0, nearest_numb)
    
    return sar1, sar2
    
    
    

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Interpolate high-resolution tropospheric product map.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('ifgramStack', help='Unwrapped interferograms loaded in mintpy format.')
    parser.add_argument('gps_file',help='input file name (e.g., gps_aps_variogramModel.h5).')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')    
    parser.add_argument('ifgNumb',help='The interferogram number to be processed.')
    parser.add_argument('--exclude-stations', dest='excludeStation', help='Text of stations to be excluded.')
    parser.add_argument('--elevation-model', dest='elevation_model',default = 'onn_linear',choices= ['linear','onn','onn_linear','exp','exp_linear'],help='elevation models. [default: onn_linear]')
    parser.add_argument('--variogram-model', dest='variogram_model',default = 'spherical',choices= ['spherical','gaussian','exponential','hole-effect','power','linear'],help='variogram models. [default: spherical]')
    parser.add_argument('--max-length', dest='max_length',type=float, default=150, metavar='NUM',
                      help='used maximum distance for mdeling the structure model.')
    parser.add_argument('--bin_numb', dest='bin_numb',type=int,default=30, metavar='NUM',
                      help='number of bins used to fit the variogram model')
    parser.add_argument('--coh-threshold', dest='coh_threshold',type=float,default=0.85, metavar='NUM',
                      help='minimum coherence used for threshold.')
    parser.add_argument('--remove_numb', dest='remove_numb',type=int,default=0, metavar='NUM',
                      help='remove the largest data for variogram estimation.')
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, help='Enable parallel processing and Specify the number of processors.')
    parser.add_argument('--cross-validation-numb', dest='cross_validation_numb', type=int, default=0, help='Cross validation of CMVE using the GPS observation.')
    parser.add_argument('--nearest-numb', dest='nearestNumb', default= 'all', help='Number of the closest points used for Kriging interpolation. [default: all]')
       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   Generate high-resolution atmospheric water vapor maps by fusing GPS and InSAR measurements.
'''

EXAMPLE = """Example:
  
  cmve_pwv.py ifgramStack.h5 gps_delay_variogramModel.h5 geometryRadar.h5 10
  cmve_pwv.py ifgramStack.h5 gps_delay_variogramModel.h5 geometryRadar.h5 15 --parallel 8
  cmve_pwv.py ifgramStack.h5 gps_delay_variogramModel.h5 geometryRadar.h5 10 --nearest-numb 15
  cmve_pwv.py ifgramStack.h5 gps_delay_variogramModel.h5 geometryRadar.h5 10 --exclude-stations exclude_station.txt

  

###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse()
    ifgram_file = inps.ifgramStack
    gps_file = inps.gps_file
    geom_file = inps.geo_file
    ifgNumb = inps.ifgNumb
    
    elevation_model = inps.elevation_model
    variogram_model = inps.variogram_model
    bin_numb = inps.bin_numb
    rn = inps.remove_numb
    
    dem = read_hdf5(geom_file,datasetName = 'height', box=None)[0]
    latitude_sar = read_hdf5(geom_file,datasetName = 'latitude', box=None)[0]
    longitude_sar = read_hdf5(geom_file,datasetName = 'longitude', box=None)[0]
    
    row_nan,col_nan = np.where(np.isnan(latitude_sar))
    latitude_sar[row_nan,col_nan] = 0
    longitude_sar[row_nan,col_nan] = 0
    
    m0 = np.nanmean(longitude_sar)
    #print(m0)
    if m0 < 0:
        longitude_sar0 = longitude_sar + 360.0
    else:
        longitude_sar0 = longitude_sar
    
    #print(longitude_sar0)
    inc_sar = read_hdf5(geom_file,datasetName = 'incidenceAngle', box=None)[0]

    attr0 = read_attribute(ifgram_file)
    wavelength = float(attr0['WAVELENGTH'])
    
    #row0,col0 = geo2sar(34.0, 242.5, latitude_sar, longitude_sar)
    #print(row0)
    #print(col0)
    
    root_path = os.getcwd()
    gigpy_dir = root_path + '/gigpy'
    cmve_dir = gigpy_dir + '/cmve'
    
    if not os.path.isdir(gigpy_dir):
        os.mkdir(gigpy_dir)
    if not os.path.isdir(cmve_dir):
        os.mkdir(cmve_dir)
    
    slice_list = get_slice_list(ifgram_file)
    unwrap_slice = []
    coherence_slice = []
    
    for k0 in slice_list:
        if 'unwrap' in k0:
            unwrap_slice.append(k0)
        else:
            coherence_slice.append(k0)
    
    unwrap = unwrap_slice[int(ifgNumb)]
    coherence = coherence_slice[int(ifgNumb)]
    pair = unwrap.split('-')[1]
    print('')
    print('Processed interferogram: ' + pair)
    print('')
    
    pair_dir = cmve_dir + '/' + pair
    if not os.path.isdir(pair_dir):
        os.mkdir(pair_dir)

    unwrap_data = read_hdf5_file(ifgram_file,datasetName = unwrap, box=None)
    insar_zenith = unwrap_data/(-4*np.pi)*wavelength*np.cos(inc_sar/180*np.pi) # convert LOS to zenith
    #print(np.cos(inc_sar/180*np.pi))
    row,col = insar_zenith.shape
    
    coherence_data = read_hdf5_file(ifgram_file,datasetName = coherence, box=None)
    
    date1 = unwrap.split('-')[1].split('_')[0]
    date2 = unwrap.split('-')[1].split('_')[1]
    

    
    wzd1, wzd_turb1, elevation_para1, trend_para1, variogram_para1, station1, lat1, lon1, hei1 = read_gps_unavco(gps_file,date1)
    
    wzd2, wzd_turb2, elevation_para2, trend_para2, variogram_para2, station2, lat2, lon2, hei2 = read_gps_unavco(gps_file,date2)
    
    lat2 = lat2.reshape(len(lat2),1)
    lon2 = lon2.reshape(len(lon2),1)
    
    dist, kk = latlon2dis(lat1[3],lon1[3], lat2,lon2)
    dist00 = latlon2dis_matrix(lat2,lon2, lat2,lon2)
    
    stationC, latC, lonC, heiC, wzdC1, wzdC2, fg001, fg002 = get_common_station(station1, lat1, lon1, hei1, wzd1, station2, lat2, lon2, hei2, wzd2)
    
    row_sar, col_sar = get_sarCoord(latC,lonC,latitude_sar,longitude_sar)
    
    lat_sar = latitude_sar[row_sar,col_sar]
    lon_sar = longitude_sar[row_sar,col_sar]
    
    #print(lat_sar)
    #print(latC)
    
    insar_obs = insar_zenith[row_sar,col_sar]
    insar_coh = coherence_data[row_sar,col_sar]
    insar_hei = dem[row_sar,col_sar]
    
    stationxC1, wzdxC1, wzdxC2, latx1, lonx1, heix1, row_sar1, col_sar1, lat_sar1, lon_sar1, insar_hei1, insar_obs1, insar_coh1 = pre_model_dry_orb(stationC, wzdC1, wzdC2, latC, lonC, heiC, row_sar, col_sar, lat_sar, lon_sar, insar_hei, insar_obs, insar_coh, row, col, coh_threshold = '0.85')
    
    heix1 = np.array(heix1)
    insar_hei1 = np.array(insar_hei1)
    dem_diff = heix1.mean() - insar_hei1.mean()
    print('Mean differnce between GPS/SRTM elevations: ' + str(heix1.mean() - insar_hei1.mean()))
    
    fig = plt.figure(figsize=(20.0, 8.0))
    plt.plot(heix1,'ro--',label = 'GPS elevation',markersize= 10)
    plt.plot(insar_hei1,'bo--',label = 'SRTM elevation',markersize= 10)
    plt.xlabel('GPS stations',fontsize = 16,fontweight = 'bold')
    plt.ylabel('Elevations (m)',fontsize = 16,fontweight = 'bold')
    plt.grid(True)
    plt.legend()
    plt.title('Elevation difference between InSAR & GPS',fontsize = 16,fontweight = 'bold')
    #plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    #plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')   
    figure_out = pair_dir + '/' + pair + '_SRTM_GPS_elevation.png'
    fig.savefig(figure_out, dpi=300)
    #print(np.array(lat_sar1) - np.array(latx1))
    #print(np.array(lon_sar1) + 360 - np.array(lonx1))
    #print(row_sar1)
    #print(col_sar1)
    
    dgps = (np.array(wzdxC1) - np.array(wzdxC2))
    diff_insar_dgps = np.array(insar_obs1) - dgps
    #print(dgps)
    #print(np.array(insar_obs1))
    #print(diff_insar_dgps)
    
    
    dryOrb_sar, para_dryOrb, corr_dryOrb = model_dry_orb(latx1, lonx1, insar_hei1, diff_insar_dgps)
    insar_dwzd = np.array(insar_obs1) - dryOrb_sar
    corr0 = pearsonr(np.array(insar_obs1),dgps-dgps.mean())
    corr1 = pearsonr(np.array(insar_dwzd),dgps)

    #print(corr_dryOrb)
    #print(corr0)
    #print(corr1)
    
    
    dryOrb_sar, para_dryOrb, corr_dryOrb = model_dry_orb_new(latx1, lonx1, insar_hei1, diff_insar_dgps)
    insar_dwzd = np.array(insar_obs1) - dryOrb_sar
    corr0 = pearsonr(np.array(insar_obs1),dgps-dgps.mean())
    corr1 = pearsonr(np.array(insar_dwzd),dgps)

    #print(corr_dryOrb)
    #print(corr0)
    #print(corr1)
    
    txt_gps_insar = cmve_dir + '/' + pair + '/'+ pair + '_cmve.txt'
    if os.path.isfile(txt_gps_insar):
        os.remove(txt_gps_insar)
    
    dryOrb_sar0 = dryOrb_sar - dryOrb_sar.mean()
    with open(txt_gps_insar, 'w') as f:
        STR0 = '  Station   Lat        Lon      wzd1     wzd2    dwzd_gps   insar    model  dwzd_insar\n'
        f.write(STR0)
        for i in range(len(latx1)):
            STR0 = '  ' + stationxC1[i].decode("utf-8") + '   ' + str(round(latx1[i]*100000)/100000) + '   ' + str(round(lonx1[i]*100000)/100000) + '   ' + str(round(wzdxC1[i]*10000)/10000) + '   ' + str(round(wzdxC2[i]*10000)/10000) + '   ' + str(round(dgps[i]*10000)/10000) + '   ' + str(round(insar_obs1[i]*10000)/10000) + '   ' + str(round(dryOrb_sar0[i]*10000)/10000) + '   ' + str(round(insar_dwzd[i]*10000)/10000) + '\n'
            f.write(STR0)
    
    
    lat0_map = latitude_sar/180.0*np.pi
    lon00_map = longitude_sar0/180.0*np.pi  
    lon0_map = lon00_map*np.cos(lat0_map) # to get isometrics coordinates
    
    X = lat0_map, lon0_map, dem
    a0, b0, c0, d0 = para_dryOrb
    insar_model_map = dry_orb_sar_new(X,a0, b0, c0, d0)
    insar_dwzd_map = insar_zenith - insar_model_map
          
    fig = plt.figure(figsize=(20.0, 8.0))
    #plt.scatter(lonx1,latx1,c = insar_obs1, cmap='hsv', alpha=0.75)
    #plt.scatter(lonx1,latx1,c = wzdxC1, cmap='hsv', alpha=0.75)
    plt.plot(np.array(insar_obs1),'ko-',label = 'Original InSAR', markersize= 10)
    plt.plot(dgps,'ro-',label = 'DWZD_gps',markersize= 10)
    plt.plot(insar_dwzd,'bo-',label = 'DWZD InSAR',markersize= 10)
    #plt.plot(diff_insar_dgps,'ro-',label = 'Original Diff')
    #plt.plot(dryOrb_sar,'bo-',label = 'Modeled Diff')
    plt.xlabel('GPS stations',fontsize = 16,fontweight = 'bold')
    plt.ylabel('wet zenith delays (m)',fontsize = 16,fontweight = 'bold')
    plt.grid(True)
    plt.legend()
    plt.title(pair+' ( corr:' + str(round(corr1[0]*1000)/1000) + ')',fontsize = 16,fontweight = 'bold')
    #plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    #plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')
    
    figure_out = pair_dir + '/' + pair + '_InSAR_GPS_dwzd.png'
    fig.savefig(figure_out, dpi=300)
    
    
    date1 = pair.split('_')[0]
    date2 = pair.split('_')[1]
    
    print('Start to estimate the spatial wzd models...')
    print('Elevation model: ' + inps.elevation_model)
    print('Variogram model: ' + inps.variogram_model)
    print('Max length: ' + str(inps.max_length))
    print('Bin number: ' + str(inps.bin_numb))
    print('Remove number: ' + str(inps.remove_numb))
    
    
    wzd_turb1, elevation_model_para1, ramp_model_para1, corr_elevation1, corr_ramp1 = space_elevation_model(lat1, lon1, hei1, wzd1, elevation_model =inps.elevation_model)    
    wzd_turb2, elevation_model_para2, ramp_model_para2, corr_elevation2, corr_ramp2 = space_elevation_model(lat2, lon2, hei2, wzd2, elevation_model =inps.elevation_model)
    
    lat11, lon11, wzd11, fg1 = remove_numb(lat1,lon1,wzd_turb1,numb = rn)
    lat22, lon22, wzd22, fg2 = remove_numb(lat2,lon2,wzd_turb2,numb = rn)
    
    lat1 = lat1[fg1].flatten()
    lon1 = lon1[fg1].flatten()
    hei1 = hei1[fg1].flatten()
    #insar_hei1 = np.array(insar_hei1)
    #insar_hei1 = insar_hei1.flatten()
    #insar_hei1 = insar_hei1[fg1].flatten()
    station1 = np.array(station1)
    station1 = station1[fg1].flatten()
    wzd1 = wzd1[fg1].flatten()
    

    lat2 = lat2[fg2].flatten()
    lon2 = lon2[fg2].flatten()
    hei2 = hei2[fg2].flatten()
    #insar_hei2 = np.array(insar_hei2)
    #insar_hei2 = insar_hei2.flatten()
    #insar_hei2 = insar_hei2[fg2].flatten()
    station2 = np.array(station2)
    station2 = station2[fg2].flatten()
    wzd2 = wzd2[fg2].flatten()
    

    ##### calculate dwzd difference between InSAR and GPS #############
    stationC, latC, lonC, heiC, wzdC1, wzdC2, fg001, fg002 = get_common_station(station1, lat1, lon1, hei1, wzd1, station2, lat2, lon2, hei2, wzd2)
    
    row_sar, col_sar = get_sarCoord(latC,lonC,latitude_sar,longitude_sar)
    
    lat_sar = latitude_sar[row_sar,col_sar]
    lon_sar = longitude_sar[row_sar,col_sar]
    
    #print(lat_sar)
    #print(latC)
    
    insar_obs = insar_zenith[row_sar,col_sar]
    insar_coh = coherence_data[row_sar,col_sar]
    insar_hei = dem[row_sar,col_sar]
    
    stationxC1, wzdxC1, wzdxC2, latx1, lonx1, heix1, row_sar1, col_sar1, lat_sar1, lon_sar1, insar_hei1, insar_obs1, insar_coh1 = pre_model_dry_orb(stationC, wzdC1, wzdC2, latC, lonC, heiC, row_sar, col_sar, lat_sar, lon_sar, insar_hei, insar_obs, insar_coh, row, col, coh_threshold = '0.85')
    
    dgps = (np.array(wzdxC1) - np.array(wzdxC2))
    diff_insar_dgps = np.array(insar_obs1) - dgps
    
    dryOrb_sar, para_dryOrb, corr_dryOrb = model_dry_orb(latx1, lonx1, insar_hei1, diff_insar_dgps)
    insar_dwzd = np.array(insar_obs1) - dryOrb_sar
    corr0 = pearsonr(np.array(insar_obs1),dgps-dgps.mean())
    corr1 = pearsonr(np.array(insar_dwzd),dgps)
    
    dryOrb_sar, para_dryOrb, corr_dryOrb = model_dry_orb_new(latx1, lonx1, insar_hei1, diff_insar_dgps)
    insar_dwzd = np.array(insar_obs1) - dryOrb_sar
    corr0 = pearsonr(np.array(insar_obs1),dgps-dgps.mean())
    corr1 = pearsonr(np.array(insar_dwzd),dgps)
    
    lat0_map = latitude_sar/180.0*np.pi
    lon00_map = longitude_sar0/180.0*np.pi  
    lon0_map = lon00_map*np.cos(lat0_map) # to get isometrics coordinates
    
    X = lat0_map, lon0_map, dem
    a0, b0, c0, d0 = para_dryOrb
    insar_model_map = dry_orb_sar_new(X,a0, b0, c0, d0)
    insar_dwzd_map = insar_zenith - insar_model_map
    
    
    txt_gps_insar = cmve_dir + '/' + pair + '/'+ pair + '_cmve.txt'
    if os.path.isfile(txt_gps_insar):
        os.remove(txt_gps_insar)
    
    dryOrb_sar0 = dryOrb_sar - dryOrb_sar.mean()
    with open(txt_gps_insar, 'w') as f:
        STR0 = '  Station   Lat        Lon      wzd1     wzd2    dwzd_gps   insar    model  dwzd_insar\n'
        f.write(STR0)
        for i in range(len(latx1)):
            STR0 = '  ' + stationxC1[i].decode("utf-8") + '   ' + str(round(latx1[i]*100000)/100000) + '   ' + str(round(lonx1[i]*100000)/100000) + '   ' + str(round(wzdxC1[i]*10000)/10000) + '   ' + str(round(wzdxC2[i]*10000)/10000) + '   ' + str(round(dgps[i]*10000)/10000) + '   ' + str(round(insar_obs1[i]*10000)/10000) + '   ' + str(round(dryOrb_sar0[i]*10000)/10000) + '   ' + str(round(insar_dwzd[i]*10000)/10000) + '\n'
            f.write(STR0)
    
    
    fig = plt.figure(figsize=(20.0, 8.0))
    #plt.scatter(lonx1,latx1,c = insar_obs1, cmap='hsv', alpha=0.75)
    #plt.scatter(lonx1,latx1,c = wzdxC1, cmap='hsv', alpha=0.75)
    plt.plot(np.array(insar_obs1),'ko-',label = 'Original InSAR', markersize= 10)
    plt.plot(dgps,'ro-',label = 'DWZD_gps',markersize= 10)
    plt.plot(insar_dwzd,'bo-',label = 'DWZD InSAR',markersize= 10)
    #plt.plot(diff_insar_dgps,'ro-',label = 'Original Diff')
    #plt.plot(dryOrb_sar,'bo-',label = 'Modeled Diff')
    plt.xlabel('GPS stations',fontsize = 16,fontweight = 'bold')
    plt.ylabel('wet zenith delays (m)',fontsize = 16,fontweight = 'bold')
    plt.grid(True)
    plt.legend()
    plt.title(pair+' ( corr:' + str(round(corr1[0]*1000)/1000) + ')',fontsize = 16,fontweight = 'bold')
    #plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    #plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')
    
    figure_out = pair_dir + '/' + pair + '_InSAR_GPS_dwzd_cor.png'
    fig.savefig(figure_out, dpi=300)
      
    ###############################################################
    hei1_cor = hei1 - dem_diff
    hei2_cor = hei2 - dem_diff
    
    wzd_turb1, elevation_model_para1, ramp_model_para1, corr_elevation1, corr_ramp1 = space_elevation_model(lat1, lon1, hei1_cor, wzd1, elevation_model =inps.elevation_model)    
    wzd_turb2, elevation_model_para2, ramp_model_para2, corr_elevation2, corr_ramp2 = space_elevation_model(lat2, lon2, hei2_cor, wzd2, elevation_model =inps.elevation_model)
    
    
    
    Lags1, Semivariance1 , variogram_para1, variogram_corr1 = space_variogram_model(lat1, lon1, wzd_turb1,variogram_model = 'spherical', max_length = inps.max_length, bin_numb = inps.bin_numb, remove_outlier = 0)
    Lags2, Semivariance2, variogram_para2, variogram_corr2 = space_variogram_model(lat2, lon2, wzd_turb2,variogram_model = 'spherical', max_length = inps.max_length, bin_numb = inps.bin_numb, remove_outlier = 0)
    
    print('  ' + date1  +  '      ' + str(len(lat1)) +  '        ' + str(round(corr_elevation1*10000)/10000) + '       '  + str(round(corr_ramp1*1000)/1000) + '        ' + str(round(variogram_corr1*10000)/10000))
    print('  ' + date2  +  '      ' + str(len(lat2)) +  '        ' + str(round(corr_elevation2*10000)/10000) + '       '  + str(round(corr_ramp2*1000)/1000) + '        ' + str(round(variogram_corr2*10000)/10000))
    
    
    vari_func = variogram_dict[inps.variogram_model]
    
    
    fig = plt.figure(figsize=(20, 10.0))
    
    fw = 'normal'
    
    dem0 = dem.flatten()
    xx = np.arange(hei1.min(),dem0.max(),200)
    elevation_function = model_dict[inps.elevation_model]
    yy1 = elevation_function(elevation_model_para1,xx) 
    yy2 = elevation_function(elevation_model_para2,xx)   
    diff_yy = yy1 - yy2
    #yy2_test = elevation_function(elevation_model_para2,xx+30)
    
    ax = plt.subplot(221)
    plt.plot(hei1_cor,  wzd1*100, 'bo', label = 'Total wzd sample',ms = 8)
    plt.plot(xx, yy1*100, 'r-', label = 'Stratified wzd model',linewidth=3.0)
    #plt.plot(hei1, wzd1 - elevation_function(elevation_model_para1,hei1), 'mo', label = 'turbulent wzd with ramp',ms = 8)
    plt.plot(hei1_cor, wzd_turb1*100, 'go', label = 'Turbulent wzd sample',ms = 8)
    plt.title(date1,fontsize = 20, fontweight = 'bold', fontname='Times New Roman')
    plt.xlabel('Elevation (m)',fontsize = 16,fontweight = fw)
    plt.ylabel('Wet zenith delays (cm)',fontsize = 16,fontweight = fw)
    plt.legend(loc = 'upper right', fontsize = 12)
    plt.tick_params(labelsize = 10)
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight= fw)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight= fw)
    
    ax = plt.subplot(222)
    plt.plot(hei2_cor,  wzd2*100, 'bo', label = 'Total wzd sample',ms = 8)
    plt.plot(xx, yy2*100, 'r-', label = 'Stratified wzd model',linewidth=3.0)
    #plt.plot(xx, yy2_test*100, 'k-', label = 'Stratified wzd model',linewidth=3.0)
    #plt.plot(hei2, wzd2 - elevation_function(elevation_model_para2,hei2), 'mo', label = 'turbulent wzd with ramp',ms = 8)
    plt.plot(hei2_cor, wzd_turb2*100, 'go', label = 'Turbulent wzd sample',ms = 8)
    plt.title(date2,fontsize = 20,fontweight = 'bold',fontname='Times New Roman')
    plt.xlabel('Elevation (m)',fontsize = 16,fontweight = fw)
    plt.ylabel('Wet zenith delays (cm)',fontsize = 16,fontweight = fw)
    plt.legend(loc = 'upper right', fontsize = 12)
    plt.tick_params(labelsize = 10)
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight=fw)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight=fw)
    
    #print(variogram_para1)
    ax = plt.subplot(223)
    plt.plot(Lags1[Lags1 < inps.max_length], Semivariance1[Lags1 < inps.max_length]*10000, 'yo', label = 'Variance sample',ms = 12)
    plt.plot(Lags1[Lags1 < inps.max_length], (vari_func(variogram_para1, Lags1[Lags1 < inps.max_length])*10000), '-', color = 'lightgray', label = 'Variogram model',linewidth=3.0)
    #plt.title(date1 + ': Variogram model',fontsize = 16,fontweight = 'bold')
    plt.xlabel('Distance (km)',fontsize = 16,fontweight = fw)
    plt.ylabel('Variance (cm$^2$)',fontsize = 16,fontweight = fw)
    plt.legend(loc = 'lower right', fontsize = 12)
    plt.tick_params(labelsize = 10)
    
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    
    dx = (xmax - xmin)
    dy = (ymax - ymin)
    
    kx = 0.5*(xmax - xmin) + xmin
    ky = 0.5*(ymax - ymin) + ymin -0.1*dy
    
    ax.text(kx, ky + 0.1*dy, 'Sill     : ' + str(round(variogram_para1[0]*1000000)/100) + ' (cm$^2$)', fontsize=14, fontname='Times New Roman')
    ax.text(kx, ky, 'Range : ' + str(round(variogram_para1[1]*10)/10) + ' (km)', fontsize=14, fontname='Times New Roman')
    ax.text(kx, ky - 0.1*dy,'Nugget: ' + str(round(variogram_para1[2]*1000000)/100) + ' (cm$^2$)', fontsize=14, fontname='Times New Roman')
    
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight=fw)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight=fw)
    
    plt.grid(True)
    
    ax = plt.subplot(224)
    plt.plot(Lags2[Lags2 < inps.max_length], Semivariance2[Lags2 < inps.max_length]*10000, 'yo', label = 'Variance sample',ms = 12)
    plt.plot(Lags2[Lags2 < inps.max_length], (vari_func(variogram_para2, Lags2[Lags2 < inps.max_length])*10000), '-', color = 'lightgray', label = 'Variogram model',linewidth=3.0)
    #plt.title(date2 + ': Variogram model',fontsize = 16,fontweight = 'bold')
    plt.xlabel('Distance (km)',fontsize = 16,fontweight = fw)
    plt.ylabel('Variance (cm$^2$)',fontsize = 16,fontweight = fw)
    plt.legend(loc = 'lower right',fontsize = 12)
    plt.tick_params(labelsize = 10)
    
    
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    
    dx = (xmax - xmin)
    dy = (ymax - ymin)
    
    kx = 0.5*(xmax - xmin) + xmin
    ky = 0.5*(ymax - ymin) + ymin -0.1*dy
    
    ax.text(kx, ky + 0.1*dy, 'Sill     : ' + str(round(variogram_para2[0]*1000000)/100) + ' (cm$^2$)', fontsize=14, fontname='Times New Roman')
    ax.text(kx, ky, 'Range : ' + str(round(variogram_para2[1]*10)/10) + ' (km)', fontsize=14, fontname='Times New Roman')
    ax.text(kx, ky - 0.1*dy,'Nugget: ' + str(round(variogram_para2[2]*1000000)/100) + ' (cm$^2$)', fontsize=14, fontname='Times New Roman')
    
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight=fw)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight=fw)
    
    plt.grid(True)
    figure_out = pair_dir + '/' + pair + '_GPS_wzd_models.png'
    fig.savefig(figure_out, dpi = 300)
    
    fig = plt.figure(figsize=(20.0, 8.0))
    plt.plot(xx,yy1*100,'g--',label = 'Stratified wzd model: ' + date1,linewidth= 3)
    plt.plot(xx, yy2*100,'b--',label = 'Stratified wzd model: ' + date2,linewidth= 3)
    plt.plot(xx, diff_yy*100,'r-',label = 'Stratified wzd model: ' + date1 + '-' + date2,linewidth= 3)
    plt.xlabel('Elevations (m)',fontsize = 16,fontweight = fw)
    plt.ylabel('Modeled stratified wzd (cm)',fontsize = 16,fontweight = fw)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.grid(True)
    plt.legend(fontsize = 16)
    #plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    #plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')   
    figure_out = pair_dir + '/' + pair + '_InSAR_stratified_wzd.png'
    fig.savefig(figure_out, dpi=300)
    
    a01, b01, c01, d01 = ramp_model_para1
    a02, b02, c02, d02 = ramp_model_para2
    
    lat_sarX = latitude_sar/180*np.pi
    lon_sarX = longitude_sar0/180*np.pi
    lon_sarX = lon_sarX*np.cos(lat_sarX)
    
    lalo_sarX = lat_sarX*lon_sarX
    XX = (lat_sarX, lon_sarX, lalo_sarX)
    
    insar_stratified_map = elevation_function(elevation_model_para1,dem)  - elevation_function(elevation_model_para2,dem)
    insar_trend_map = func_ramp(XX,a01,b01,c01,d01) - func_ramp(XX,a02,b02,c02,d02)
    insar_turbulent_map = insar_dwzd_map - insar_stratified_map - insar_trend_map
    
    datasetDict = dict()  
    datasetDict['insar_dwzd'] = insar_dwzd_map
    datasetDict['insar_model'] = insar_model_map
    datasetDict['insar_obs'] = insar_zenith
    
    datasetDict['insar_dem'] = dem
    datasetDict['insar_coh'] = coherence_data
    datasetDict['insar_inc'] = inc_sar
    
    datasetDict['insar_stratified_dwzd'] = insar_stratified_map
    datasetDict['insar_turbulent_dwzd'] = insar_turbulent_map
    datasetDict['insar_trend_dwzd'] = insar_trend_map

    attr0['UNIT'] = 'm'
    attr0['FILE_TYPE'] = 'cmve'
    OUT = pair_dir + '/' + pair + '_cmve.h5'
    ut0.write_h5(datasetDict, OUT, metadata=attr0, ref_file=None, compression=None)
    
    
    coherence_data0 = coherence_data.flatten()
    latitude_sar0 = latitude_sar.flatten()
    longitude_sar0 = longitude_sar0.flatten()
    insar_turbulent_map0 = insar_turbulent_map.flatten()
    
    
    idx1 = np.where(coherence_data0 > inps.coh_threshold)
    idx1 = np.array(idx1)
    idx1 = idx1.reshape(-1,1)
    list_length = len(idx1)
    print('')
    print('Coherence threshold : ' + str(inps.coh_threshold))
    print('Total pixel number: ' + str(len(idx1)))
    print('Used GPS station number: ' + inps.nearestNumb)
    
    split_numb = 1000
    list_para = split_list(list_length, processors = split_numb)
    
    nearest_numb = inps.nearestNumb
    data_parallel = []
    
    # remove repeat stations
    lat1_u,idx1_u = np.unique(lat1,return_index=True) 
    lat2_u,idx2_u = np.unique(lat2,return_index=True)
    lon1_u = lon1[idx1_u]
    lon2_u = lon2[idx2_u]
    wzd_turb1_u = wzd_turb1[idx1_u]
    wzd_turb2_u = wzd_turb2[idx2_u]
    
    station1_u = station1[idx1_u]
    station2_u = station2[idx2_u]
    hei1_u = hei1_cor[idx1_u]
    hei2_u = hei2_cor[idx2_u]
    
    ######################### cross-validation ##############################
    stationC, latC, lonC, heiC, wzdC1, wzdC2, fgC1, fgC2 = get_common_station(station1_u, lat1_u, lon1_u, hei1_u, wzd_turb1_u, station2_u, lat2_u, lon2_u, hei2_u, wzd_turb2_u)
    
    row_sar, col_sar = get_sarCoord(latC,lonC,latitude_sar,longitude_sar)
    
    lat_sar = latitude_sar[row_sar,col_sar]
    lon_sar = longitude_sar[row_sar,col_sar]
    
    insar_obs = insar_zenith[row_sar,col_sar]
    insar_coh = coherence_data[row_sar,col_sar]
    insar_hei = dem[row_sar,col_sar]
    
    fg_cross0 = sorted(random.sample(list(np.arange(len(row_sar))),  inps.cross_validation_numb)) 
    
    
    fg_cross1 = fgC1[fg_cross0]
    fg_cross2 = fgC2[fg_cross0]
    
    row_sar = np.array(row_sar)
    row_sar = row_sar.flatten()
    
    col_sar = np.array(col_sar)
    col_sar = col_sar.flatten()
    row_sar_cross = row_sar[fg_cross0]
    col_sar_cross = col_sar[fg_cross0]
    
    fg_cmve1 = []
    fg_cmve2 = []
    
    for i in range(len(lat1_u)):
        if i not in fg_cross1:
            fg_cmve1.append(i)
    
    for i in range(len(lat2_u)):
        if i not in fg_cross2:
            fg_cmve2.append(i)       
    
    fg_cmve1 = np.array(fg_cmve1)
    fg_cmve2 = np.array(fg_cmve2)
    
    lat1_cross = lat1_u[fg_cross1]
    lon1_cross = lon1_u[fg_cross1]
    hei1_cross = hei1_u[fg_cross1]
    station1_cross = station1_u[fg_cross1]
    wzd_turb1_cross = wzd_turb1_u[fg_cross1]
    
    lat1_cmve = lat1_u[fg_cmve1]
    lon1_cmve = lon1_u[fg_cmve1]
    hei1_cmve = hei1_u[fg_cmve1]
    station1_cmve = station1_u[fg_cmve1]
    wzd_turb1_cmve = wzd_turb1_u[fg_cmve1]
    
    lat2_cross = lat2_u[fg_cross2]
    lon2_cross = lon2_u[fg_cross2]
    hei2_cross = hei2_u[fg_cross2]
    station2_cross = station2_u[fg_cross2]
    wzd_turb2_cross = wzd_turb2_u[fg_cross2]
    
    lat2_cmve = lat2_u[fg_cmve2]
    lon2_cmve = lon2_u[fg_cmve2]
    hei2_cmve = hei2_u[fg_cmve2]
    station2_cmve = station2_u[fg_cmve2]
    wzd_turb2_cmve = wzd_turb2_u[fg_cmve2]
    
    
    wzd_turb1_cross_kriging = []
    wzd_turb2_cross_kriging = []
    
    for i in range(len(lat1_cross)):
        lat0 = lat1_cross[i]
        lon0 = lon1_cross[i]
        
        wx1 = kriging_interp(lat1_cmve, lon1_cmve, wzd_turb1_cmve, lat0, lon0, variogram_para1, vari_func, 30)
        wx2 = kriging_interp(lat2_cmve, lon2_cmve, wzd_turb2_cmve, lat0, lon0, variogram_para2, vari_func, 30)
        
        wzd_turb1_cross_kriging.append(wx1)
        wzd_turb2_cross_kriging.append(wx2)
    
    wzd_turb1_cross_kriging = np.array(wzd_turb1_cross_kriging)
    wzd_turb1_cross = np.array(wzd_turb1_cross)
    wzd_turb1_cross_kriging = wzd_turb1_cross_kriging.flatten()
    wzd_turb1_cross = wzd_turb1_cross.flatten()
    
    wzd_turb2_cross_kriging = np.array(wzd_turb2_cross_kriging)
    wzd_turb2_cross = np.array(wzd_turb2_cross)
    wzd_turb2_cross_kriging = wzd_turb2_cross_kriging.flatten()
    wzd_turb2_cross = wzd_turb2_cross.flatten()
    
    print(wzd_turb1_cross, wzd_turb1_cross_kriging)
    print(wzd_turb2_cross, wzd_turb2_cross_kriging)
    ###########################################################################
    
    AA = kriging_matrix(lat1_u,lon1_u,wzd_turb1_u, variogram_para1,lat2_u,lon2_u,wzd_turb2_u, variogram_para2,vari_func)
    #print(AA)
    
    for i in range(len(list_para)):
        lat_list0 = latitude_sar0[idx1[list_para[i]]]
        lon_list0 = longitude_sar0[idx1[list_para[i]]]
        data_list0 = insar_turbulent_map0[idx1[list_para[i]]]
        data0 = (AA, wzd_turb1_cmve, lat1_cmve, lon1_cmve, variogram_para1, wzd_turb2_cmve, lat2_cmve, lon2_cmve, variogram_para2, vari_func, lat_list0, lon_list0, data_list0, nearest_numb)
        data_parallel.append(data0)
        
    #sar1, sar2 = cmve_list(wzd_turb1_u, lat1_u, lon1_u, variogram_para1, wzd_turb2_u, lat2_u, lon2_u, variogram_para2, vari_func, lat_list0, lon_list0, data_list0, nearest = '15')
    
    #print(sar1,sar2)
    
    future = ut0.parallel_process(data_parallel, cmve_para, n_jobs=inps.parallelNumb, use_kwargs=False)
   
    row, col = latitude_sar.shape
    zz_sar1 = np.zeros((row,col))
    zz_sar2 = np.zeros((row,col))
    
    zz_sar1 = zz_sar1.flatten()
    zz_sar2 = zz_sar2.flatten()
    
    for i in range(split_numb):
        id0 = idx1[list_para[i]]
        gg = future[i]
        xx1 = np.array(gg[0])
        xx2 = np.array(gg[1])
        xx1 = xx1.reshape(-1,1)
        xx2 = xx2.reshape(-1,1)
        zz_sar1[id0] = xx1
        zz_sar2[id0] = xx2
    
    zz_sar1 = zz_sar1.reshape(row,col)
    zz_sar2 = zz_sar2.reshape(row,col)
    
    datasetDict = dict()  
    datasetDict['turbulent_wzd1'] = zz_sar1
    datasetDict['turbulent_wzd2'] = zz_sar2
    
    attr0['UNIT'] = 'm'
    attr0['FILE_TYPE'] = 'cmve'
    OUT = pair_dir + '/SAR_wzd.h5'
    ut0.write_h5(datasetDict, OUT, metadata=attr0, ref_file=None, compression=None)
    ##print(kk10)
    
    
    ff0 = '/Users/caoy0a/Documents/SCRATCH/LosAngelesT71F479S1D_New/gigpy/cmve/20171205_20171217/SAR_wzd.h5'
    zz_sar1 = read_hdf5(ff0,datasetName = 'turbulent_wzd1', box=None)[0]
    zz_sar2 = read_hdf5(ff0,datasetName = 'turbulent_wzd2', box=None)[0]
    
    wzd_turb1_cross_cmve = zz_sar1[row_sar_cross,col_sar_cross]
    wzd_turb2_cross_cmve = zz_sar2[row_sar_cross,col_sar_cross]
    
    txt_gps_insar_cross = cmve_dir + '/' + pair + '/'+ pair + '_cmve_cross_validation.txt'
    if os.path.isfile(txt_gps_insar_cross):
        os.remove(txt_gps_insar_cross)
    
    dryOrb_sar0 = dryOrb_sar - dryOrb_sar.mean()
    with open(txt_gps_insar_cross, 'w') as f:
        STR0 = '  Station   latitude   longitude  gps1  kriging1   cmve1   gps2  kriging2  cmve2 \n'
        STR00 = '  Station   latitude   longitude  gps1  kriging1   cmve1   gps2  kriging2  cmve2'
        print(STR00)
        f.write(STR0)
        for i in range(len(lat1_cross)):
            STR0 = '  ' + station1_cross[i].decode("utf-8") + '   ' + str(round(lat1_cross[i]*100000)/100000) + '   ' + str(round(lon1_cross[i]*100000)/100000) + '   ' + str(round(wzd_turb1_cross[i]*100000)/100) + '    ' + str(round(wzd_turb1_cross_kriging[i]*100000)/100) + '   ' + str(round(wzd_turb1_cross_cmve[i]*100000)/100) + '   '+ str(round(wzd_turb2_cross[i]*100000)/100)  + '   ' + str(round(wzd_turb2_cross_kriging[i]*100000)/100) + '   ' + str(round(wzd_turb2_cross_cmve[i]*100000)/100) + '\n'
            
            STR00 = '  ' + station1_cross[i].decode("utf-8") + '   ' + str(round(lat1_cross[i]*100000)/100000) + '   ' + str(round(lon1_cross[i]*100000)/100000) + '   ' + str(round(wzd_turb1_cross[i]*100000)/100) + '    ' + str(round(wzd_turb1_cross_kriging[i]*100000)/100) + '   ' + str(round(wzd_turb1_cross_cmve[i]*100000)/100) + '   '+ str(round(wzd_turb2_cross[i]*100000)/100)  + '   ' + str(round(wzd_turb2_cross_kriging[i]*100000)/100) + '   ' + str(round(wzd_turb2_cross_cmve[i]*100000)/100)
            print(STR00)
            
            f.write(STR0)
    
    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])
