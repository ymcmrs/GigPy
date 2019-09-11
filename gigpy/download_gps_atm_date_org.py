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
import math
import astropy.time
import dateutil.parser


def float_yyyymmdd(DATESTR):
    year = float(DATESTR.split('-')[0])
    month = float(DATESTR.split('-')[1])
    day = float(DATESTR.split('-')[2])
    
    date = year + month/12 + day/365 
    return date

def rm(FILE):
    call_str = 'rm ' + FILE
    os.system(call_str)

def add_zero(s):
    if len(s)==1:
        s="00"+s
    elif len(s)==2:
        s="0"+s
    return s   
    
def yyyymmdd2yyyydd(DATE):
    LE = len(str(int(DATE)))
    DATE = str(DATE)
    
    if LE == 6:
        YY = int(DATE[0:2])
        if YY > 80:
            DATE = '19' + DATE
        else:
            DATE = '20' + DATE
            
    year = int(str(DATE[0:4]))
    month = int(str(DATE[4:6]))
    day = int(str(DATE[6:8]))
    
    if year%4==0:
        x = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        x = [31,28,31,30,31,30,31,31,30,31,30,31]

    kk = np.ones([12,1])
    kk[month-1:]= 0
    
    x = np.asarray(x)
    kk = np.asarray(kk)
    
    day1 = np.dot(x,kk)
    DD = day + day1 
    DD= str(int(DD[0]))
    
    DD = add_zero(str(DD))
   
    ST = str(year) + str(DD)
    
    return ST,DATE

def yyyy2yyyymmddhhmmss(t0):
    hh = int(t0*24)
    mm = int((t0*24 - int(t0*24))*60)
    ss = (t0*24*60 - int(t0*24*60))*60
    ST = str(hh)+':'+str(mm)+':'+str(ss)
    return ST  

def unitdate(DATE):
    LE = len(str(int(DATE)))
    DATE = str(int(DATE))
    if LE==5:
        DATE = '200' + DATE  
    
    if LE == 6:
        YY = int(DATE[0:2])
        if YY > 80:
            DATE = '19' + DATE
        else:
            DATE = '20' + DATE
    return DATE

def readdate(DATESTR):
    s1 = DATESTR
    DD=[]

    if ',' not in s1:
        DD.append(s1)
    else:
        k = len(s1.split(','))
        for i in range(k):
            DD.append(s1.split(',')[i])
            
    return DD
        
        
###################################################################################################

INTRODUCTION = '''GPS:
    GPS stations are searched from Nevada Geodetic Laboratory by using search_gps.py 
    website:  http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html
   
    GPS atmosphere data is download from UNAVOC, please check: download_gps_atm.py
    website:  ftp://data-out.unavco.org/pub/products/troposphere
    
    GPS deformation data is download from Nevada Geodetic Laboratory                              
    website:  http://geodesy.unr.edu/gps_timeseries/tenv3/IGS08
    
'''

EXAMPLE = '''EXAMPLES:
    download_gps_atm_date.py 20150101 
    download_gps_atm_date.py 20150101 --station BGIS
    download_gps_atm_date.py 20150101 --station_txt search_gps_inside.txt
'''    
    


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Download GPS data over SAR coverage.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('date',help='GPS station name.')
    #parser.add_argument('--station', dest='station_name', help='GPS station name.')
    #parser.add_argument('--station_txt', dest='station_txt', help='GPS station txet file.')
    
    inps = parser.parse_args()

    
    return inps

    
####################################################################################################
def main(argv):
    
    inps = cmdLineParse()
    DATE = inps.date
    DATE = str(int(DATE))
    ST,DATE = yyyymmdd2yyyydd(DATE)
    YEAR = ST[0:4]
    DAY = ST[4:]
    
    root_path = os.getcwd()
    gig_dir = root_path + '/gigpy'
    gig_atm_dir = gig_dir  + '/atm'
    gig_atm_raw_dir = gig_dir  + '/atm/raw'
    #gps_def_dir = PATH + '/GPS/def'
    
    if not os.path.isdir(gig_dir):
        os.mkdir(gig_dir)
    
    if not os.path.isdir(gig_atm_dir):
        os.mkdir(gig_atm_dir)
        
    if not os.path.isdir(gig_atm_raw_dir):
        os.mkdir(gig_atm_raw_dir)
        
        
    ttt = gig_atm_dir + '/ttt_' + DATE
    ttt_aps = gig_atm_dir + '/ttt_aps_' + DATE
    ttt0 = gig_atm_dir + '/ttt0_' + DATE
    ttt_size = gig_atm_dir + '/ttt_size_' + DATE
    
    ttt_pwv = gig_atm_dir + '/ttt_pwv_' + DATE
    
    call_str = 'curl ftp://data-out.unavco.org/pub/products/troposphere/' + YEAR + '/' + DAY + '/' + ' > ' + ttt 
    os.system(call_str)    
    
    # get aps file for downloading 
    call_str ="grep 'cwu' " + ttt +' > ' + ttt_aps
    os.system(call_str)
    call_str ="grep '.gz' " + ttt_aps + ' > ' + ttt0
    os.system(call_str)
    BB = np.loadtxt(ttt0,dtype=np.str)
    
    call_str = "awk '{print $5}' " + ttt0 + ' > ' + ttt_size
    os.system(call_str)
    
    AA = np.loadtxt(ttt_size)
    kk = AA.size
    
    if kk>1:
        AA = list(map(int,AA))
        IDX = AA.index(max(AA))
        FILE = BB[int(IDX),8]
    else:
        AA = int(AA)
        FILE = BB[8]
    
    
    # get pwv file for downloading 
    call_str ="grep 'nmt' " + ttt + ' > ' + ttt_pwv
    os.system(call_str)
    
    fl =1
    if os.path.getsize(ttt_pwv) > 0:
        call_str ="grep '.gz' " + ttt_pwv + ' > ' + ttt0
        os.system(call_str)
        BB = np.loadtxt(ttt0 , dtype=np.str)
    
        call_str = "awk '{print $5}' " +  ttt0 + ' > ' + ttt_size
        os.system(call_str)
    
        AA = np.loadtxt(ttt_size)
        kk = AA.size
        
        if kk>1:
            AA = list(map(int,AA))
            IDX = AA.index(max(AA))
            FILE_PWV = BB[int(IDX),8]
        elif kk==1:
            AA = int(AA)
            FILE_PWV = BB[8]
    else:
        FILE_PWV = ''
        fl=0
        print('No PWV-file is found.')
    
    Trop_GPS = gig_atm_raw_dir + '/' + 'Global_GPS_Trop_' + DATE 
    
    if not os.path.isfile(FILE):    
        call_str = 'wget -q ftp://data-out.unavco.org/pub/products/troposphere/' + YEAR + '/' + DAY + '/' + FILE
        print('Downloading GPS tropospheric delay data: %s' % DATE)
        os.system(call_str)
    print('Download finish.')
    print('')
        
    FILE0 = FILE.replace('.gz','')
    if os.path.isfile(FILE0):
        os.remove(FILE0)

    call_str = 'gzip -d ' + FILE
    os.system(call_str)
    
    call_str ='cp ' + FILE0 + ' ' + Trop_GPS
    os.system(call_str)
    os.remove(FILE0)    
    

    if (not os.path.isfile(FILE_PWV)) and (fl==1):
        call_str = 'wget -q ftp://data-out.unavco.org/pub/products/troposphere/' + YEAR + '/' + DAY + '/' + FILE_PWV
        print('Downloading GPS PWV data: %s' % DATE)
        os.system(call_str)
        print('Download finish.')
    
    if fl ==1:
        Trop_PWV_GPS = gig_atm_raw_dir + '/' + 'Global_GPS_PWV_' + DATE
        
        FILE0 = FILE_PWV.replace('.gz','')
        if os.path.isfile(FILE0):
            os.remove(FILE0)
        call_str = 'gzip -d ' + FILE_PWV
        os.system(call_str)
        
        call_str ='cp ' + FILE0 + ' ' + Trop_PWV_GPS
        os.system(call_str)
        
        os.remove(FILE0)

    os.remove(ttt)
    os.remove(ttt_aps) 
    os.remove(ttt0)
    os.remove(ttt_size)
    
    os.remove(ttt_pwv)  
    #if inps.station_name:
    #    DD=readdate(inps.station_name)
    #    k = len(DD)
    #elif inps.station_txt:
    #    GPS= np.loadtxt(inps.station_txt, dtype = np.str)
    #    DD =GPS[:,0]
    #    k=len(DD)
    #    DD = DD.tolist()
    #if k>0:
    #    print('Extracting tropospheric delays for ' + str(int(k)) + ' GPS stations:')
    #    for i in range(k):
    #        Nm=DD[i]
    #        print(Nm)
    #        OUT = gps_atm_dir + '/' + Nm + '_Trop_' + DATE
    #        call_str = "grep " + Nm + ' ' + Trop_GPS + '> ' + OUT
    #        os.system(call_str)
    #        
    #        OUT = gps_atm_dir + '/' + Nm + '_Trop_PWV_' + DATE
    #        call_str = "grep " + Nm + ' ' + Trop_PWV_GPS + '> ' + OUT
    #        os.system(call_str)
            
            
if __name__ == '__main__':
    main(sys.argv[1:])

