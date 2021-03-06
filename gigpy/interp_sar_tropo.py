#! /usr/bin/env python
#################################################################
###  This program is part of GigPy  v1.0                      ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################

import sys
import os
import re
import subprocess
import argparse
import numpy as np
import h5py
from gigpy import elevation_models

from pykrige import OrdinaryKriging

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from gigpy import _utils as ut
###############################################################

model_dict = {'linear': elevation_models.linear_elevation_model,
                      'onn': elevation_models.onn_elevation_model,
                      'onn_linear': elevation_models.onn_linear_elevation_model,
                      'exp': elevation_models.exp_elevation_model,
                      'exp_linear': elevation_models.exp_linear_elevation_model}



def function_trend(lat,lon,para):
    # mod = a*x + b*y + c*x*y
    lat = lat/180*np.pi
    lon = lon/180*np.pi  
    lon = lon*np.cos(lat) # to get isometrics coordinates
    
    a0,b0,c0,d0 = para
    data_trend = a0 + b0*lat + c0*lon +d0*lat*lon
    
    return data_trend


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

def get_dataNames(FILE):
    with h5py.File(FILE, 'r') as f:
        dataNames = []
        for k0 in f.keys():
            dataNames.append(k0)
    return dataNames

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

def adjust_aps_lat_lon_unavco(gps_aps_h5,epoch = 0):
    
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

def adjust_aps_lat_lon_unr(gps_aps_h5,epoch = 0):
    
    FILE = gps_aps_h5
    
    gps_hei = read_hdf5(FILE,datasetName='gps_height')[0]
    gps_lat = read_hdf5(FILE,datasetName='gps_lat')[0]
    gps_lon = read_hdf5(FILE,datasetName='gps_lon')[0]
    gps_nm = read_hdf5(FILE,datasetName='gps_name')[0]
    gps_nm = list(gps_nm)

    date = read_hdf5(FILE,datasetName='date')[0]
    station = read_hdf5(FILE,datasetName='station')[0]
    tzd = read_hdf5(FILE,datasetName='tzd_turb')[0]
    
    station= list(station[epoch])
    tzd= list(tzd[epoch])
    
    k0 =9999
    for i in range(len(station)):
        if station[i].decode("utf-8")=='0.0':
            if i < k0:
                k0 =i
    station = station[0:k0]   
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
    
    return tzd_turb, lat, lon

def get_bounding_box(meta):
    """Get lat/lon range (roughly), in the same order of data file
    lat0/lon0 - starting latitude/longitude (first row/column)
    lat1/lon1 - ending latitude/longitude (last row/column)
    """
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    if 'Y_FIRST' in meta.keys():
        # geo coordinates
        lat0 = float(meta['Y_FIRST'])
        lon0 = float(meta['X_FIRST'])
        lat_step = float(meta['Y_STEP'])
        lon_step = float(meta['X_STEP'])
        lat1 = lat0 + lat_step * (length - 1)
        lon1 = lon0 + lon_step * (width - 1)
    else:
        # radar coordinates
        lats = [float(meta['LAT_REF{}'.format(i)]) for i in [1,2,3,4]]
        lons = [float(meta['LON_REF{}'.format(i)]) for i in [1,2,3,4]]
        lat0 = np.mean(lats[0:2])
        lat1 = np.mean(lats[2:4])
        lon0 = np.mean(lons[0:3:2])
        lon1 = np.mean(lons[1:4:2])
    return lat0, lat1, lon0, lon1


def get_lat_lon(meta):
    """Get 2D array of lat and lon from metadata"""
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    lat0, lat1, lon0, lon1 = get_bounding_box(meta)
    lat, lon = np.mgrid[lat0:lat1:length*1j, lon0:lon1:width*1j]
    return lat, lon


def correct_timeseries(timeseries_file, trop_file, out_file):
    print('\n------------------------------------------------------------------------------')
    print('correcting delay for input time-series by calling diff.py')
    cmd = 'diff.py {} {} -o {} --force'.format(timeseries_file,
                                               trop_file,
                                               out_file)
    print(cmd)
    status = subprocess.Popen(cmd, shell=True).wait()
    if status is not 0:
        raise Exception(('Error while correcting timeseries file '
                         'using diff.py with tropospheric delay file.'))
    return out_file


def split_lat_lon_kriging(nn, processors = 4):

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

def OK_function(data0):
    OK,lat0,lon0,np = data0
    z0,s0 = OK.execute('points', lon0, lat0, n_closest_points= np, backend='loop')
    return z0,s0

def dist_weight_interp(data0):
    lat0,lon0,z0,lat1,lon1 = data0
    
    lat0 = np.asarray(lat0)
    lon0 = np.asarray(lon0)
    z0 = np.asarray(z0)
    
    if len(z0)==1:
        z0 = z0[0]
    nn = len(lat1)
    data_interp = np.zeros((nn,))
    weight_all = []
    for i in range(nn):
        dist0 = latlon2dis(lat0,lon0,lat1[i],lon1[i])
        weight0 = (1/dist0)**2
        if len(weight0) ==1:
            weight0 = weight0[0]
        weight = weight0/sum(weight0[:])
        data_interp[i] = sum(z0*weight)
        weight_all.append(weight)
    return data_interp,weight_all

def latlon2dis(lat1,lon1,lat2,lon2,R=6371):
    
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

    return dist
    
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Interpolate high-resolution tropospheric product map.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('date', help='SAR acquisition date.')
    parser.add_argument('gps_file',help='input file name (e.g., gps_aps_variogram.h5).')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('--type', dest='type', choices = {'tzd','wzd'}, default = 'tzd',help = 'type of the high-resolution tropospheric map.[default: tzd]')
    parser.add_argument('--method', dest='method', choices = {'kriging','weight_distance'},default = 'kriging',help = 'method used to interp the high-resolution map. [default: kriging]')
    parser.add_argument('-o','--out', dest='out_file', metavar='FILE',help='name of the prefix of the output file')
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, help='Enable parallel processing and Specify the number of processors.')
    parser.add_argument('--kriging-points-numb', dest='kriging_points_numb', type=int, default=15, help='Number of the closest points used for Kriging interpolation. [default: 15]')
       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   Generate high-resolution GPS-based tropospheric maps (delays & water vapor) for InSAR Geodesy & meteorology.
'''

EXAMPLE = """Example:
  
  interp_sar_tropo.py 20190101 gps_file geometry.h5 --type tzd -o 20190101_gps_aps.h5
  interp_sar_tropo.py 20190101 gps_file geometry.h5 --type wzd --parallel 8
  interp_sar_tropo.py 20190101 gps_file geometry.h5 --method kriging 
  interp_sar_tropo.py 20190101 gps_file geometry.h5 --method kriging --kriging-points-numb 15
  interp_sar_tropo.py 20190101 gps_file geometry.h5 --method weight_distance
  interp_sar_tropo.py 20190101 gps_file geometry.h5 --type tzd 
  interp_sar_tropo.py 20190101 gps_file geometry.h5 --type wzd --parallel 4
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse() 
    gps_file = inps.gps_file
    geom_file = inps.geo_file
    date = inps.date
    
    datasetNames = ut.get_dataNames(gps_file)
    
    meta0 = read_attr(gps_file)
    meta = read_attr(geom_file)
    WIDTH = int(meta['WIDTH'])
    LENGTH = int(meta['LENGTH'])
    date_list = read_hdf5(gps_file,datasetName='date')[0]
    date_list = date_list.astype('U13')
    date_list = list(date_list)

    
    root_path = os.getcwd()
    gig_dir = root_path + '/gigpy'
    gig_atm_dir = gig_dir  + '/atm'
    gig_atm_raw_dir = gig_dir  + '/atm/raw'
    gig_atm_sar_raw_dir = gig_dir  + '/atm/sar_raw'
    
    gig_atm_sar_tzd_dir = gig_dir  + '/atm/sar_tzd'
    gig_atm_sar_wzd_dir = gig_dir  + '/atm/sar_wzd'
    
    if not os.path.isdir(gig_atm_sar_tzd_dir):
        os.mkdir(gig_atm_sar_tzd_dir)
    
    if not os.path.isdir(gig_atm_sar_wzd_dir):
        os.mkdir(gig_atm_sar_wzd_dir)
    
    if inps.out_file: OUT = inps.out_file
    else: OUT = date + '_' + inps.type + '.h5'
    
    if inps.type =='tzd': OUT = gig_atm_sar_tzd_dir + '/' + OUT
    elif inps.type =='wzd': OUT = gig_atm_sar_wzd_dir + '/' + OUT

    dem = read_hdf5(geom_file,datasetName='height')[0]
    #inc = read_hdf5(geom_file,datasetName='incidenceAngle')[0]
    
    dataNames = get_dataNames(geom_file)
    if 'latitude' in dataNames:
        grid_lat = read_hdf5(geom_file,datasetName='latitude')[0]
        grid_lon = read_hdf5(geom_file,datasetName='longitude')[0]
    else:
        grid_lat, grid_lon = get_lat_lon(meta)
    
    lats = grid_lat.flatten()
    lons = grid_lon.flatten()
    
    
    where_are_NaNs = np.isnan(lats)
    lats[where_are_NaNs] = 0

    where_are_NaNs = np.isnan(lons)
    lons[where_are_NaNs] = 0
    #print(grid_lat.shape)
    # TWD, ZWD, TURB_Kriging, Turb_ITD, Trend, HgtCor, 
    k0 = date_list.index(date)
    
    if 'wzd_turb' in datasetNames:
        tzd_turb, wzd_turb, lat, lon = adjust_aps_lat_lon_unavco(gps_file,epoch = k0)
    else:
        tzd_turb, lat, lon = adjust_aps_lat_lon_unr(gps_file,epoch = k0)

    #print(inps.type)
    if inps.type == 'tzd': 
        S0 = 'tzd'
        turb =tzd_turb
    elif inps.type == 'wzd':
        S0 = 'wzd'
        turb = wzd_turb
    
    lat0, lon0, turb0 = remove_numb(lat,lon,turb,int(meta0['remove_numb']))
    
    mean_lons = np.mean(lons)
    lons0 = round((np.mean(lon0) - mean_lons)/360)*360 + lons # adjust to the same trend system
    #lons0 = lons
    lats0 = lats
    #print(lats)
    #print(lons)
    if np.mean(lon0) > 180:
        lon0 = lon0 - 360
    #print(lat0)
    #print(lon0)
    variogram_para = read_hdf5(gps_file,datasetName = S0 + '_variogram_parameter')[0]
    trend_para = read_hdf5(gps_file,datasetName = S0 + '_trend_parameter')[0]
    elevation_para = read_hdf5(gps_file,datasetName = S0 + '_elevation_parameter')[0]
    
    variogram_para0 = variogram_para[k0,:]
    trend_para0 = trend_para[k0,:]
    elevation_para0 = elevation_para[k0,:]
    
    Trend0 = function_trend(lats0,lons0,trend_para0)
    Trend0 = Trend0.reshape(LENGTH,WIDTH)
    
    hei = dem.flatten()
    elevation_function = model_dict[meta0['elevation_model']]
    Dem_cor0 = elevation_function(elevation_para0,hei)
    Dem_cor0 = Dem_cor0.reshape(LENGTH,WIDTH)
    
    OK = OrdinaryKriging(lon0, lat0, turb0, variogram_model=meta0['variogram_model'], verbose=False,enable_plotting=False)
    para = variogram_para0[0:3]
    #print(para)
    para[1] = para[1]/6371/np.pi*180
    #print(para)
    OK.variogram_model_parameters = para

    Numb = inps.parallelNumb
    zz = np.zeros((len(lats),))
    zz_sigma = np.zeros((len(lats),))
    #print(len(lats))
    n_jobs = inps.parallelNumb
    
    # split dataset into multiple subdata
    split_numb = 1000
    idx_list = split_lat_lon_kriging(len(lats),processors = split_numb)
    #print(idx_list[split_numb-1])
    
    print('------------------------------------------------------------------------------')
    if inps.type =='tzd':
        print('Start to interpolate the high-resolution tropospheric delay for SAR acquisition: ' + date)
        print('Total number of the available GPS stations: %s' % str(len(lat)))
    elif inps.type =='wzd':
        print('Start to interpolate the high-resolution atmospheric water vapor for SAR acquisition: ' + date)
        print('Total number of the available GPS stations: %s' % str(len(lat)))
        
    if inps.method =='kriging':
        np0 = inps.kriging_points_numb
        data = []
        for i in range(split_numb):
            data0 = (OK,lats[idx_list[i]],lons[idx_list[i]],np0)
            data.append(data0)
        future = ut.parallel_process(data, OK_function, n_jobs=Numb, use_kwargs=False)
        
    elif inps.method =='weight_distance':
        data = []
        for i in range(split_numb):
            data0 = (lat0,lon0,turb0,lats[idx_list[i]],lons[idx_list[i]])
            data.append(data0)
        future = ut.parallel_process(data, dist_weight_interp, n_jobs=Numb, use_kwargs=False)
    
    for i in range(split_numb):
        id0 = idx_list[i]
        gg = future[i]
        zz[id0] = gg[0]
        
    turb = zz.reshape(LENGTH,WIDTH)
    datasetDict =dict()
    
    aps = Dem_cor0 + Trend0 + turb
    datasetDict['hgt_sar'] = Dem_cor0
    datasetDict['trend_sar'] = Trend0
    datasetDict['turb_sar'] = turb
    datasetDict['aps_sar'] = aps
    
    if inps.method == 'kriging':
        for i in range(split_numb):
            id0 = idx_list[i]
            gg = future[i]
            zz_sigma[id0] = gg[1]
        sigma = zz_sigma.reshape(LENGTH,WIDTH)
        datasetDict['turb_sar_sigma'] = sigma
    
    meta['DATA_TYPE'] = inps.type
    meta['interp_method'] = inps.method
    
    for key, value in meta0.items():
        meta[key] = str(value)
    
    ut.write_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)
    
    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])