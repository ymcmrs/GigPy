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
import glob


###############################################################

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


def write_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):
    
    if os.path.isfile(out_file):
        print('delete exsited file: {}'.format(out_file))
        os.remove(out_file)

    print('create HDF5 file: {} with w mode'.format(out_file))
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

def generate_datelist_txt(date_list,txt_name):
    
    if os.path.isfile(txt_name): os.remove(txt_name)
    for list0 in date_list:
        call_str = 'echo ' + list0 + ' >>' + txt_name 
        os.system(call_str)
        
    return

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('ts_file',help='input InSAR time-series file name (e.g., timeseries.h5).')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('--elevation-model', dest='elevation_model', 
                        choices = {'linear','onn','onn_linear','exp','exp_linear'},default = 'onn_linear',
                       help = 'model used to estimate the elevation-correlated components. [default: onn_linear]')
    parser.add_argument('--variogram-model', dest='variogram_model', 
                        choices = {'spherical','exponential','gaussian','linear'},default = 'spherical',
                        help = 'method used to model the variogram. [default: spherical]')
    parser.add_argument('--interp-method', dest='interp_method', choices = {'kriging','weight_distance'},
                        default = 'kriging',
                        help = 'method used to interp the high-resolution turbulence map. [default: kriging]')
    parser.add_argument('--kriging-points-numb', dest='kriging_points_numb', type=int, default=20, 
                        help='Number of the closest points used for Kriging interpolation. [default: 20]')
    parser.add_argument('--type', dest='type', choices = {'tzd','wzd'},default = 'tzd',
                       help = 'data type of the output, aps: atmospheric delay; pwv: atmosperic water vapor.')
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, 
                        help='Enable parallel processing and Specify the number of processors.[default: 1]')
    parser.add_argument('--variogram-remove-numb', dest='removeNumb', type=int, default=5, 
                        help='Number of the removed outliers for calculating the turbulent variogram.[default: 5]')
    parser.add_argument('--variogram-bin-numb', dest='binNumb', type=int, default=50, 
                        help='Number of the bins or variance samples of the turbulent variogram.[default: 50]')
    parser.add_argument('--max-length', dest='maxLength', type=float, default=150.0, 
                        help='max lenghth used for modeling the turbulent variogram.[default: 150 km]')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',help='name of output file')

    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   Generate high-resolution map of GPS tropospheric measurements for synchronous SAR acquisitions.
'''

EXAMPLE = """example:
  
  tropo_gigpy.py timeseries.h5 geometryRadar.h5
  tropo_gigpy.py timeseries.h5 geometryRadar.h5 --type pwv 
  tropo_gigpy.py timeseries.h5 geometryRadar.h5 --type aps --parallel 4
  tropo_gigpy.py timeseries.h5 geometryRadar.h5 --elevation-model linear --type pwv --parallel 8
  tropo_gigpy.py timeseries.h5 geometryRadar.h5 --elevation-model linear --variogram-model spherical
  tropo_gigpy.py timeseries.h5 geometryRadar.h5 --elevation-model linear --interp-method kriging --type pwv
  tropo_gigpy.py timeseries.h5 geometryRadar.h5 --elevation-model onn_linear --interp-method kriging 
  
###################################################################################
"""


def main(argv):
    
    inps = cmdLineParse() 
    ts_file = inps.ts_file
    geo_file = inps.geo_file
    root_dir = os.getcwd()
    gps_dir = root_dir + '/GPS'
    atm_dir = gps_dir + '/atm'
    
    meta = read_attr(ts_file)
    
    if not os.path.isfile('search_gps.txt'):
        print('----------------------------------------------------------------')
        print('Start to search available GPS tations over the research area...')
        call_str = 'search_gps.py -f ' + ts_file
        os.system(call_str)
        print('')
    else:
        print('')
        print('----------------------------------------------------------------')
        print('search_gps.txt exist, skip search gps stations.')
        print('')
    
    if not os.path.isdir(gps_dir): 
        os.mkdir(gps_dir)
        print('Generate GPS directory: %s' % gps_dir)
    if not os.path.isdir(atm_dir): 
        os.mkdir(atm_dir)
        print('Generate GPS-atm directory: %s' % atm_dir)
    
    date_list_exist = [os.path.basename(x).split('_')[3] for x in glob.glob(atm_dir + '/Global_GPS_Trop*')]
   
    date_list = read_hdf5(ts_file,datasetName='date')[0]
    date_list = date_list.astype('U13')
    date_list = list(date_list)
    
    date_list_download = []
    for i in range(len(date_list)):
        if not date_list[i] in date_list_exist:
            date_list_download.append(date_list[i])
            
    print('---------------------------------------')
    print('Total number of gps data: ' + str(len(date_list)))
    print('Exist number of gps data: ' + str(len(date_list)-len(date_list_download)))
    print('Number of date to download: ' + str(len(date_list_download)))
    
    if len(date_list_download) > 0:
        print('Start to download gps data...')
        txt_download = 'datelist_download.txt'
        generate_datelist_txt(date_list_download,txt_download)
        call_str = 'download_sar_atm.py ' + txt_download
        os.system(call_str)
    
    extract_list_exist = [os.path.basename(x).split('_')[3] for x in glob.glob(atm_dir + '/SAR_GPS_Trop_*')]
    date_list_extract = []
    for i in range(len(date_list)):
        if not date_list[i] in extract_list_exist:
            date_list_extract.append(date_list[i])
    
    print('---------------------------------------')
    print('Exist number of extracted gps data: ' + str(len(date_list)-len(date_list_extract)))
    print('Number of gps data to extract: ' + str(len(date_list_extract)))
    
    
    if len(date_list_extract) > 0:
        print('Start to extract gps data...')
        txt_extract = 'datelist_extract.txt'
        generate_datelist_txt(date_list_extract,txt_extract)
        call_str = 'extract_sar_atm.py search_gps.txt ' + meta['CENTER_LINE_UTC'] + ' --date_txt ' + txt_extract
        os.system(call_str)
        
    print('')
    print('---------------------------------------')
    print('Start to analyze the elevation-dependent components of the tropospheric products...')
    print('Used elevation model: %s' % inps.elevation_model)
    print('')
    call_str = 'elevation_correlation.py gps_aps.h5 -m ' + inps.elevation_model  
    os.system(call_str)
    
    print('')
    print('---------------------------------------')
    print('Start to calculate the variogram of the turbulent tropospheric products...')
    #print('Used variogram model: %s' % inps.variogram_model)
    call_str = 'variogram_gps.py gps_aps_HgtCor.h5 --remove_numb ' + str(inps.removeNumb) + ' --bin_numb ' + str(inps.binNumb)
    os.system(call_str)
    
    print('')
    print('---------------------------------------')
    print('Start to estimate the variogram model of the turbulent tropospheric products...')
    print('Used variogram model: %s' % inps.variogram_model)
    print('Max length used for model estimation: %s km' % inps.maxLength)
    print('')
    call_str = 'gps_variogram_modeling.py gps_aps_variogram.h5  --max-length ' + str(inps.maxLength) + ' --model ' + inps.variogram_model
    os.system(call_str)
    
    if inps.type =='tzd': 
        Stype = 'Tropospheric delays'
    else:  
        Stype = 'Atmospheric water vapor'
    print('')
    print('---------------------------------------')
    print('Start to generate high-resolution tropospheric maps ...' )
    print('Type of the tropospheric products: %s' % Stype)
    print('Method used for interpolation: %s' % inps.interp_method)
    print('Number of processors used for interpolation: %s' % str(inps.parallelNumb))
    print('')
    
    date_generate = []
    for i in range(len(date_list)):
        out0 = date_list[i] + '_' + inps.type + '.h5'
        if not os.path.isfile(out0):
            date_generate.append(date_list[i])
    print('Total number of data set: %s' % str(len(date_list)))          
    print('Exsit number of data set: %s' % str(len(date_list)-len(date_generate))) 
    print('Number of high-resolution maps need to be interpolated: %s' % str(len(date_generate)))  
    
    if len(date_generate) > 0 :
        txt_generate = 'datelist_generate.txt'
        generate_datelist_txt(date_generate,txt_generate)
        call_str = 'interp_sar_tropo_list.py ' + txt_generate + ' gps_aps_variogramModel.h5 ' + inps.geo_file + ' --type ' + inps.type + ' --method ' + inps.interp_method + '  --kriging-points-numb ' + str(inps.kriging_points_numb) + ' --parallel ' + str(inps.parallelNumb)
        os.system(call_str)
        
        
    print('')
    print('---------------------------------------')
    print('Start to generate time-series of tropospheric data ...' )
    txt_list = 'date_list.txt'
    generate_datelist_txt(date_list,txt_list)
    call_str = 'generate_timeseries_tropo.py ' + txt_list +' ' + inps.ts_file + ' --type ' + inps.type
    os.system(call_str)
    
    print('Done.')
    
    if inps.type =='aps':
        print('')
        print('---------------------------------------')
        print('Transfer the atmospheric delays from zenith direction to Los direction ...')
        call_str = 'zennith2los.py timeseries_gps_aps.h5 ' + inps.geo_file
        os.system(call_str)
        print('Done.')
        print('')
        print('Start to correct InSAR time-series tropospheric delays ...' )
        call_str = 'diff_gigpy.py ' + inps.ts_file + ' ' + ' timeseries_gps_aps_los.h5 --add  -o timeseries_gpsCor.h5'
        os.system(call_str)
        print('Done.')

    sys.exit(1)


###############################################################

if __name__ == '__main__':
    main(sys.argv[:])

