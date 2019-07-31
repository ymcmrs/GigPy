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
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generate high-resolution tropospheric product map for a list of SAR acquisitions',formatter_class=argparse.RawTextHelpFormatter,epilog=INTRODUCTION+'\n'+EXAMPLE)
    parser.add_argument('date_list', help='SAR acquisition date.')
    parser.add_argument('ts_file',help='input InSAR time-series file name (e.g., timeseries.h5).')
    parser.add_argument('--data', dest='data', choices = {'turb','aps','hgt','trend','sigma'}, default = 'aps',help = 'type of the high-resolution tropospheric map.[default: aps]')
    parser.add_argument('--type', dest='type', choices = {'tzd','wzd'}, default = 'tzd',help = 'type of the high-resolution tropospheric map.[default: tzd]')
    parser.add_argument('--absolute', dest='absolute', action='store_true',help='generate absolute value based tropospheric map. ')
    

       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v2.0
   
   Generate time-series of high-resolution GPS-based tropospheric maps (include maps of turb, atmospheric delay, trend, elevation-correlated components).
'''

EXAMPLE = """Example:
  
  generate_timeseries_tropo.py date_list timeseries.h5 --data aps --absolute 
  generate_timeseries_tropo.py date_list timeseries.h5 --data turb --type wzd
  generate_timeseries_tropo.py date_list timeseries.h5 --data hgt --absolute
  generate_timeseries_tropo.py date_list timeseries.h5 --data trend --type tzd
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse()
    date_list_txt = inps.date_list      
    date_list = np.loadtxt(date_list_txt,dtype=np.str)
    date_list = date_list.tolist()
    N=len(date_list)
    
    ts_file = inps.ts_file
    meta = read_attr(ts_file)
    WIDTH = int(meta['WIDTH'])
    LENGTH = int(meta['LENGTH'])
    REF_X = int(meta['REF_X'])
    REF_Y = int(meta['REF_Y'])
    
    if inps.data =='aps': S0 = 'aps_sar'       # total tropospheric map (tropospheric delay & atmospheric water vapor)
    elif inps.data =='turb': S0 ='turb_sar'    # turbulent tropospheric map
    elif inps.data=='hgt': S0 ='hgt_sar'      # elevation-correlated tropospheric map
    elif inps.data =='trend': S0 ='trend_sar'  # spatial trend/ramp of the tropospheric map
    elif inps.data =='sigma': S0 ='sigma_sar'  # uncertainty map of the turbulent tropospheric map
    
    ts_gps = np.zeros((len(date_list),LENGTH,WIDTH),dtype = np.float32)
    for i in range(len(date_list)):
        file0 = date_list[i] + '_' + inps.type + '.h5'
        data0 = read_hdf5(file0,datasetName = S0)[0]
        if not inps.absolute:
            ts_gps[i,:,:] = data0 - data0[REF_Y,REF_X]
        else:
            ts_gps[i,:,:] = data0
        
    #ts0 = ts_gps[2,:,:]
    # relative to the first date
    #if not inps.absolute:
     #   for i in range(len(date_list)):
     #       ts_gps[i,:,:] = ts_gps[i,:,:] - ts0
    
    if not inps.absolute:
        ts_gps -=ts_gps[0,:,:]
    else:
        del meta['REF_X']
        del meta['REF_Y']
        
    out_ts_gps = 'timeseries_gps_' + inps.data + '.h5'
    datasetDict = dict()
    datasetDict['timeseries'] = ts_gps
    date_list = read_hdf5(ts_file,datasetName='date')[0]
    bperp = read_hdf5(ts_file,datasetName='bperp')[0]
    datasetDict['date'] = date_list
    datasetDict['bperp'] = bperp

    write_h5(datasetDict, out_ts_gps, metadata=meta, ref_file=None, compression=None)
    print('Done.')   

    sys.exit(1)
    
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])