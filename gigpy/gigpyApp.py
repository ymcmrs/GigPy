#!/usr/bin/env python
############################################################
# Project:  GigPy                                          #
# Purpose: GPS-based Imaging Geodesy software in Python    #
# Author:  Yunmeng Cao                                     #
# Created: July 2019                                       #
# Copyright (c) 2019, Yunmeng Cao                          #
############################################################

import sys
import os
import re
import subprocess
import argparse
import numpy as np
import h5py
import glob


###############################################################
def check_variable_name(path):
    s=path.split("/")[0]
    if len(s)>0 and s[0]=="$":
        p0=os.getenv(s[1:])
        path=path.replace(path.split("/")[0],p0)
    return path


def read_cfg(File, delimiter='='):
    '''Reads the gigpy-configure file into a python dictionary structure.
    Input : string, full path to the template file
    Output: dictionary, gigpy configure content
    Example:
        tmpl = read_cfg(LosAngelse.cfg)
        tmpl = read_cfg(R1_54014_ST5_L0_F898.000.pi, ':')
    '''
    cfg_dict = {}
    for line in open(File):
        line = line.strip()
        c = [i.strip() for i in line.split(delimiter, 1)]  #split on the 1st occurrence of delimiter
        if len(c) < 2 or line.startswith('%') or line.startswith('#'):
            next #ignore commented lines or those without variables
        else:
            atrName  = c[0]
            atrValue = str.replace(c[1],'\n','').split("#")[0].strip()
            atrValue = check_variable_name(atrValue)
            cfg_dict[atrName] = atrValue
    return cfg_dict


def read_attr(fname):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        atr = dict(f.attrs)
        
    return atr

def get_dataNames(FILE):
    with h5py.File(FILE, 'r') as f:
        dataNames = []
        for k0 in f.keys():
            dataNames.append(k0)
    return dataNames

def generate_datelist_txt(date_list,txt_name):
    
    if os.path.isfile(txt_name): os.remove(txt_name)
    for list0 in date_list:
        call_str = 'echo ' + list0 + ' >>' + txt_name 
        os.system(call_str)
        
    return

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generate high-resolution maps of GPS tropospheric measurements.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('cfg_file',help='name of the input configure file')
    parser.add_argument('-g',  dest='generate_cfg' ,help='generate an example of configure file.')

    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   GigPy: GPS-based Imaging Geodesy software in Python.
   
   Start from download raw data to generate high-resolution maps of tropospheric measurements.
'''

EXAMPLE = """example:

                 gigpyApp.py -h
                 gigpyApp.py -g
                 gigpyApp.py LosAngels.cfg
  
###################################################################################
"""


def main(argv):
    
    inps = cmdLineParse() 
    templateContents = read_cfg(inps.cfg_file)
    
    if 'process_dir' in templateContents: root_dir = templateContents['process_dir']
    else: root_dir = os.getcwd()
    
    data_source = 'UNAVCO'
    print('')
    print('---------------- Basic Parameters ---------------------')
    print('Working directory: %s' % root_dir)
    print('Data source : %s' % data_source)
    
        
    gig_dir = root_dir + '/gigpy'
    atm_dir = root_dir + '/gigpy/atm'
    atm_raw_dir = root_dir + '/gigpy/atm/raw'
    atm_sar_raw_dir = root_dir + '/gigpy/atm/sar_raw'
    atm_sar_tzd_dir = root_dir + '/gigpy/atm/sar_tzd'
    atm_sar_wzd_dir = root_dir + '/gigpy/atm/sar_wzd'
    
    # Get the interested data type
    if 'interested_type' in templateContents: interested_type = templateContents['interested_type']
    else: interested_type = 'delay'
    print('Interested data type: %s' % interested_type)
    
    
    if interested_type == 'delay': 
        inps_type = 'tzd'
        atm_sar_dir = atm_sar_tzd_dir
    elif interested_type == 'pwv': 
        inps_type = 'wzd'
        atm_sar_dir = atm_sar_wzd_dir
    
    if not os.path.isdir(gig_dir): os.mkdir(gig_dir)
    if not os.path.isdir(atm_dir): os.mkdir(atm_dir)
    if not os.path.isdir(atm_raw_dir): os.mkdir(atm_raw_dir)
    if not os.path.isdir(atm_sar_raw_dir): os.mkdir(atm_sar_raw_dir)
    if not os.path.isdir(atm_sar_tzd_dir): os.mkdir(atm_sar_tzd_dir)
    if not os.path.isdir(atm_sar_wzd_dir): os.mkdir(atm_sar_wzd_dir)
    
    # ------------ Get the parallel processing parameters
    if 'download_parallel' in templateContents: download_parallel = templateContents['download_parallel']
    else: download_parallel = 1
    
    if 'extract_parallel' in templateContents: extract_parallel = templateContents['extract_parallel']
    else: extract_parallel = 1
        
    if 'interp_parallel' in templateContents: interp_parallel = templateContents['interp_parallel']
    else: interp_parallel = 1    

    # ------------ Get research time in seconds
    if 'research_time' in templateContents: research_time = templateContents['research_time']
    elif 'research_time_file' in templateContents: 
        research_time_file = templateContents['research_time_file']
        meta = read_attr(research_time_file)
        research_time = meta['CENTER_LINE_UTC']
    print('Research UTC-time (s): %s' % str(research_time))
    print('')
    
    # Get date_list
    date_list = []
    if 'date_list' in templateContents: 
        date_list0 = templateContents['date_list']
        date_list1 = date_list0.split(',')[:]
        for d0 in date_list1: 
            date_list.append(d0)
    
    if 'date_list_txt' in templateContents: 
        date_list_txt = templateContents['date_list_txt']
        date_list2 = np.loadtxt(date_list_txt,dtype=np.str)
        date_list2 = date_list2.tolist()
        for d0 in date_list2: 
            date_list.append(d0)
    
    print('Interested date list:')
    print('')
    for k0 in date_list:
        print(k0)
    
    # ------------ Get model parameters
    if 'elevation_model' in templateContents: elevation_model = templateContents['elevation_model']
    else: elevation_model = 'onn_linear'
    
    if 'remove_numb' in templateContents: remove_numb = templateContents['remove_numb']
    else: remove_numb = 5
        
    if 'bin_numb' in templateContents: bin_numb = templateContents['bin_numb']
    else: bin_numb = 50    
        
    if 'variogram_model' in templateContents: variogram_model = templateContents['variogram_model']
    else: variogram_model = 'spherical'    
        
    if 'max_length' in templateContents: max_length = templateContents['max_length']
    else: max_length  = 150 
          
    # ------------ Get interpolate parameters    
    if 'interp_method'  in templateContents: interp_method = templateContents['interp_method']
    else: interp_method  = 'kriging'  
        
    if 'kriging_points_numb'  in templateContents: kriging_points_numb = templateContents['kriging_points_numb']
    else: kriging_points_numb  = 20
        
    if 'interp_parallel' in templateContents: interp_parallel = templateContents['interp_parallel']     
    else: interp_parallel = 1
    
    print('')
    print('*************   Step 1: search data   *************')
    print('')
    print('Start to search available GPS stations  ...')
    # Check research area and research area file
    if not os.path.isfile('gps_station_info.txt'):
        #print('Start to search available GPS tations over the research area...')
        if 'research_area_file' in templateContents: 
            research_area_file = templateContents['research_area_file'] 
            print('Determining the research area based on the provided file %s' % research_area_file)
            call_str = 'search_gps.py -f ' + research_area_file
            os.system(call_str)
        elif 'research_area' in templateContents: 
            research_area = templateContents['research_area'] 
            print('Determining the research area based on the provided corners %s' % research_area)
            call_str = 'search_gps.py -b ' + research_area
            os.system(call_str)      
    else:
        #print('')
        print('gps_station_info.txt exist, skip search gps stations.')
        print('')
     
    
    # ------------ Check the available geometry file [height, latitude, longitude]
    if 'resolution' in templateContents: resolution = templateContents['resolution']     
    else: resolution = 60
    
    if 'research_area_file' in templateContents: 
        geometry0_file = templateContents['research_area_file']
        dataNames = get_dataNames(geometry0_file)
        if ('height' in dataNames) and ('latitude' in dataNames): 
            geometry_file = geometry0_file
        else:
            print('Geometry file is not found.')
            print('Using generate_geometry.py to generate the geometry file...')
            call_str = 'generate_geometry.py --ref ' + geometry0_file + ' --resolution ' + str(resolution) 
            os.system(call_str)
            geometry_file = 'geometry.h5'
    else:
        print('Geometry file is not found.')
        print('Using generate_geometry.py to generate the geometry file...')
        call_str = 'generate_geometry.py --region ' + research_area + ' --resolution ' + str(resolution) 
        os.system(call_str)
        geometry_file = 'geometry.h5'
    
    print('')
    
    date_list_exist = [os.path.basename(x).split('_')[3] for x in glob.glob(atm_raw_dir + '/Global_GPS_Trop*')]
     
    #date_list = read_hdf5(ts_file,datasetName='date')[0]
    #date_list = date_list.astype('U13')
    #date_list = list(date_list)
    
    date_list_download = []
    for i in range(len(date_list)):
        if not date_list[i] in date_list_exist:
            date_list_download.append(date_list[i])
            
    print('')
    print('*************  Step 2: download data     *************')
    print('')
    print('Total number of gps data: ' + str(len(date_list)))
    print('Exist number of downloaded gps data: ' + str(len(date_list)-len(date_list_download)))
    print('Number of data to be downloaded: ' + str(len(date_list_download)))
    
    if len(date_list_download) > 0:
        #print('Start to download gps data...')
        txt_download = 'datelist_download.txt'
        generate_datelist_txt(date_list_download,txt_download)
        call_str = 'download_gps_atm.py ' + txt_download + ' --parallel ' + str(download_parallel)
        os.system(call_str)
    
    extract_list_exist = [os.path.basename(x).split('_')[3] for x in glob.glob(atm_sar_raw_dir + '/SAR_GPS_Trop_*')]
    date_list_extract = []
    for i in range(len(date_list)):
        if not date_list[i] in extract_list_exist:
            date_list_extract.append(date_list[i])
    
    print('')
    print('*************   Step 3: extract data      *************')
    print('')
    print('Exist number of extracted gps data: ' + str(len(date_list)-len(date_list_extract)))
    print('Number of gps data to extract: ' + str(len(date_list_extract)))
    
    
    if len(date_list_extract) > 0:
        print('Start to extract gps data...')
        txt_extract = 'datelist_extract.txt'
        generate_datelist_txt(date_list_extract,txt_extract)
        call_str = 'extract_sar_atm.py gps_station_info.txt ' + str(research_time) + ' --date-txt ' + txt_extract + ' --parallel ' + str(extract_parallel)
        os.system(call_str)
        
    print('')    
    print('********  Step 4: analyze different components   ********')
    print('')
    #print('Start to analyze the elevation-dependent components of the tropospheric products...')
    print('Seperate elevation-correlated components, atmosphere ramp, and tropospheric turbulence ..')
    print('Used elevation model: %s' % elevation_model)
    print('Used ramp model: linear') # the only option
    print('')
    call_str = 'elevation_correlation.py gps_aps.h5 -m ' + elevation_model  
    os.system(call_str)
    
    print('')
    print('******* Step 5: calculate variogram samples *******')
    print('')
    print('Number of the removed outliers: %s' % str(remove_numb))
    print('Number of the used bins: %s' % str(bin_numb))
    #print('Start to calculate the variogram of the turbulent tropospheric products...')
    #print('Used variogram model: %s' % inps.variogram_model)
    call_str = 'variogram_gps.py gps_aps_HgtCor.h5 --remove_numb ' + str(remove_numb) + ' --bin_numb ' + str(bin_numb)
    os.system(call_str)
    
    print('')
    print('******* Step 6: estimate variogram models ********')
    print('')
    #print('Start to estimate the variogram model of the turbulent tropospheric products...')
    print('Used variogram model: %s' % variogram_model)
    print('Max length used for model estimation: %s km' % max_length)
    print('')
    call_str = 'gps_variogram_modeling.py gps_aps_variogram.h5  --max-length ' + str(max_length) + ' --model ' + variogram_model
    os.system(call_str)
    
    if inps_type =='tzd': 
        Stype = 'Tropospheric delays'
    else:  
        Stype = 'Atmospheric water vapor'
    print('')
    print('******* Step 7: generate high-resolution maps ******')
    print('')
    #print('Start to generate high-resolution tropospheric maps ...' )
    print('Type of the tropospheric products: %s' % Stype)
    print('Method used for interpolation: %s' % interp_method)
    print('Number of processors used for interpolation: %s' % str(interp_parallel))
    print('')
    
    date_generate = []
    for i in range(len(date_list)):
        out0 = atm_sar_dir + '/' + date_list[i] + '_' + inps_type + '.h5'
        if not os.path.isfile(out0):
            date_generate.append(date_list[i])
    print('Total number of data set: %s' % str(len(date_list)))          
    print('Exsit number of data set: %s' % str(len(date_list)-len(date_generate))) 
    print('Number of high-resolution maps need to be interpolated: %s' % str(len(date_generate)))  
    
    if len(date_generate) > 0 :
        txt_generate = 'datelist_generate.txt'
        generate_datelist_txt(date_generate,txt_generate)
        call_str = 'interp_sar_tropo_list.py ' + txt_generate + ' gps_aps_variogramModel.h5 ' + geometry_file + ' --type ' + inps_type + ' --method ' + interp_method + '  --kriging-points-numb ' + str(kriging_points_numb) + ' --parallel ' + str(interp_parallel)
        os.system(call_str)
        
        
    print('')
    print('**** Step 8: generate time-series of tropospheric maps ****')
    #print('')
    #print('Start to generate time-series of tropospheric data ...' )
    txt_list = 'date_list.txt'
    generate_datelist_txt(date_list,txt_list)
    call_str = 'generate_timeseries_tropo.py --date-txt ' + txt_list + ' --type ' + inps_type
    os.system(call_str)
    
    #print('Done.')
    
    #if inps.type =='tzd':
    #    print('')
    #    print('---------------------------------------')
    #    print('Transfer the atmospheric delays from zenith direction to Los direction ...')
    #    call_str = 'zenith2los.py timeseries_gps_tzd.h5 ' + inps.geo_file
    #    os.system(call_str)
    #    print('Done.')
    #    print('')
    #    print('Start to correct InSAR time-series tropospheric delays ...' )
    #    call_str = 'diff_gigpy.py ' + inps.ts_file + ' ' + ' timeseries_gps_aps_los.h5 --add  -o timeseries_gpsCor.h5'
    #    os.system(call_str)
    #    print('Done.')

    sys.exit(1)


###############################################################

if __name__ == '__main__':
    main(sys.argv[:])

