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
import getopt
import array
import argparse
from skimage import io

def get_sufix(STR):
    n = len(STR.split('.'))
    SUFIX = STR.split('.')[n-1]
   
    return SUFIX 

def read_region(STR):
    WEST = STR.split('/')[0]
    EAST = STR.split('/')[1].split('/')[0]
    
    SOUTH = STR.split(EAST+'/')[1].split('/')[0]
    NORTH = STR.split(EAST+'/')[1].split('/')[1]
    
    WEST =float(WEST)
    SOUTH=float(SOUTH)
    EAST=float(EAST)
    NORTH=float(NORTH)
    return WEST,SOUTH,EAST,NORTH

def get_meta_corner(meta):
    if 'Y_FIRST' in meta.keys():
        lat0 = float(meta['Y_FIRST'])
        lon0 = float(meta['X_FIRST'])
        lat_step = float(meta['Y_STEP'])
        lon_step = float(meta['X_STEP'])
        lat1 = lat0 + lat_step * (length - 1)
        lon1 = lon0 + lon_step * (width - 1)
        
        NORTH = lat0
        SOUTH = lat1  
        WEST = lon0
        EAST = lon1
        
    else:
        
        lats = [float(meta['LAT_REF{}'.format(i)]) for i in [1,2,3,4]]
        lons = [float(meta['LON_REF{}'.format(i)]) for i in [1,2,3,4]]
        
        lat0 = np.max(lats[:])
        lat1 = np.min(lats[:])
        lon0 = np.min(lons[:])
        lon1 = np.max(lons[:])
        
        NORTH = lat0 + 0.1
        SOUTH = lat1 + 0.1
        WEST = lon0 + 0.1
        EAST = lon1 + 0.1
    
    return WEST,SOUTH,EAST,NORTH
        
def write_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):

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
   Copy Right(c) : @GigPy 2019, Yunmeng Cao 
   
   Generating geometry file includes dem/latitude/longitude and write into hdf5 file.  

'''

EXAMPLE = '''
    Usage:
            generate_geometry.py --region west/east/south/north --resolution 80
            generate_geometry.py --ref hdf5file 
            generate_geometry.py --ref hdf5file -o geometry.h5
            
    Examples:
            generate_geometry.py --region " -118/-116/33/34 " --resolution 60 -o geometry_test.h5
            generate_geometry.py --ref timeseries.h5 --resolution 80 
            
##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generating geometry file based on the corners info.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)


    parser.add_argument('--region',dest = 'region',help='Research region, west/east/south/north.')
    parser.add_argument('--ref', dest='ref_file', help='Reference hdf5 file which includes the corners info.')
    parser.add_argument('--resolution', dest='resolution', help='Spatial resolution of the output geometry file.')
    parser.add_argument('-o', dest='out', help='Output name of the geometry file.')

    inps = parser.parse_args()
    
    if (not inps.region) and (not inps.ref_file):
        parser.print_usage()
        sys.exit(os.path.basename(sys.argv[0])+': error: at least one of the two should be provided: research region,ref-h5.')

    return inps

################################################################################

def main(argv):
    
    inps = cmdLineParse()          
    if inps.region: 
        region = inps.region
        west,south,east,north = read_region(region)
    elif inps.ref_file:
        meta = read_attr(ref_file)
        west,south,east,north = get_meta_corner(meta)
        
   
    if inps.out: OUT = inps.out
    else: OUT = 'geometry.h5'
        
    print('Research region: %s(west)  %s(south)  %s(east)  %s(north)' % (west,south,east,north))
    print('>>> Ready to download SRTM1 dem over research region.')
    call_str='wget -q -O dem.tif "http://ot-data1.sdsc.edu:9090/otr/getdem?north=%f&south=%f&east=%f&west=%f&demtype=SRTMGL1"' % (north,south,east,west)
    os.system(call_str)
    print('>>> DEM download finished.')
        

    DEM = 'dem.tif'     
    SUFIX = get_sufix(DEM)
    SS ='.' + SUFIX
    DTIF = DEM.replace(SS,'.tif')

    if not SUFIX == 'tif':
        call_str = 'gdal_translate ' + DEM + ' -of GTiff ' + DTIF
        os.system(call_str)

    DEM = DTIF
    call_str = 'gdalinfo ' + DEM + ' >ttt'     
    os.system(call_str)
    
    f = open('ttt')    
    for line in f:
        if 'Origin =' in line:
            STR1 = line
            AA = STR1.split('Origin =')[1]
            Corner_LON = AA.split('(')[1].split(',')[0]
            Corner_LAT = AA.split('(')[1].split(',')[1].split(')')[0]
        elif 'Pixel Size ' in line:
            STR2 = line
            AA = STR2.split('Pixel Size =')[1]
            post_Lon = AA.split('(')[1].split(',')[0]
            post_Lat = AA.split('(')[1].split(',')[1].split(')')[0]
        
        elif 'Size is' in line:
            STR3 = line
            AA =STR3.split('Size is')[1]
            WIDTH = AA.split(',')[0]
            FILE_LENGTH = AA.split(',')[1]
    f.close()
    
    dem_data = io.imread(DEM)
    dem_data = np.asarray(dem_data, dtype = np.float32)
    row, col = dem_data.shape
    
    x = np.arange(0,col)
    y = np.arange(0,row)
    xv, yv = np.meshgrid(x, y)

    LAT = float(NORTH) + yv*float(post_Lat)
    LON = float(WEST) + xv*float(post_Lon)
    
    datasetDict = dict()
    datasetDict['height'] = dem_data
    datasetDict['latitude'] = np.asarray(LAT,dtype = np.float32)
    datasetDict['longitude'] = np.asarray(LON,dtype = np.float32)
    
    meta = {}
    meta['WIDTH'] = col
    meta['LENGTH'] = row
    meta['FILE_TYPE'] = 'geometry'
    meta['Y_FIRST'] = str(NORTH)
    meta['X_FIRST'] = str(WEST)
    meta['Y_STEP'] =str(post_Lat)
    meta['X_STEP'] = str(post_Lon)
    
    
    write_h5(datasetDict, OUT , metadata=meta, ref_file=None, compression=None)

    sys.exit(1)
       

if __name__ == '__main__':
    main(sys.argv[1:])
