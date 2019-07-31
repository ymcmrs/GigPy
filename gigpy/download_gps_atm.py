#! /usr/bin/env python
############################################################
# Program is part of PyGPS v1.0                            #
# Copyright(c) 2017, Yunmeng Cao                           #
# Author:  Yunmeng Cao                                     #
############################################################


import numpy as np
import getopt
import sys
import os
import h5py
import argparse
import math
import astropy.time
import dateutil.parser

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def print_progress(iteration, total, prefix='calculating:', suffix='complete', decimals=1, barLength=50, elapsed_time=None):
    """Print iterations progress - Greenstick from Stack Overflow
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int) 
        barLength   - Optional  : character length of bar (Int) 
        elapsed_time- Optional  : elapsed time in seconds (Int/Float)
    
    Reference: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    if elapsed_time:
        sys.stdout.write('%s [%s] %s%s    %s    %s secs\r' % (prefix, bar, percents, '%', suffix, int(elapsed_time)))
    else:
        sys.stdout.write('%s [%s] %s%s    %s\r' % (prefix, bar, percents, '%', suffix))
    sys.stdout.flush()
    if iteration == total:
        print("\n")

    '''
    Sample Useage:
    for i in range(len(dateList)):
        print_progress(i+1,len(dateList))
    '''
    return

def download_atm_date(date0):

    DATE = str(int(date0))
    ST = yyyymmdd2yyyydd(DATE)
    YEAR = ST[0:4]
    DAY = ST[4:]
    
    PATH = os.getcwd()
    gps_dir = PATH + '/GPS'
    gps_atm_dir = PATH + '/GPS/atm'
    gps_def_dir = PATH + '/GPS/def'
    
    if not os.path.isdir(gps_dir):
        call_str = 'mkdir ' + gps_dir
        os.system(call_str)
    
    if not os.path.isdir(gps_atm_dir):
        call_str = 'mkdir ' + gps_atm_dir
        os.system(call_str)

    ttt = 'ttt_' + DATE
    ttt0 = 'ttt0_' + DATE
    ttt_aps = 'ttt_aps_' + DATE
    ttt_size = 'ttt_size_' + DATE 
    
    call_str = 'curl -s ftp://data-out.unavco.org/pub/products/troposphere/' + YEAR + '/' + DAY + '/' + ' > ' + ttt 
    os.system(call_str)    
    
    # get aps file for downloading 
    call_str ="grep 'cwu' " + ttt + " > " + ttt_aps
    os.system(call_str)
    call_str ="grep '.gz' " + ttt_aps + " > " + ttt0
    os.system(call_str)
    BB = np.loadtxt(ttt0,dtype=np.str)
    
    call_str = "awk '{print $5}' " + ttt0 + " > " + ttt_size
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

    ttt0 = 'ttt0_' + DATE
    ttt_pwv = 'ttt_pwv_' + DATE
    ttt_size = 'ttt_size_' + DATE
    
    # get pwv file for downloading 
    call_str ="grep 'nmt' " + ttt + " " + " > " + ttt_pwv
    os.system(call_str)
    
    fl =1
    if os.path.getsize(ttt_pwv) > 0:
        call_str ="grep '.gz' " + ttt_pwv + " " +" > " + ttt0
        os.system(call_str)
        BB = np.loadtxt(ttt0,dtype=np.str)
    
        call_str = "awk '{print $5}' " + ttt0 + " > " + ttt_size
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
        #print('No PWV-file is found.')
    
    Trop_GPS = gps_atm_dir + '/' + 'Global_GPS_Trop_' + DATE 
    #print(FILE)
    if not os.path.isfile(FILE):    
        call_str = 'wget -q ftp://data-out.unavco.org/pub/products/troposphere/' + YEAR + '/' + DAY + '/' + FILE
        os.system(call_str)
    
    FILE0 = FILE.replace('.gz','')
    if os.path.isfile(FILE0):
        os.remove(FILE0)

    call_str = 'gzip -d ' + FILE
    os.system(call_str)
    
    call_str ='cp ' + FILE0 + ' ' + Trop_GPS
    os.system(call_str)
    if os.path.isfile(FILE0):
        os.remove(FILE0)    
    

    if (not os.path.isfile(FILE_PWV)) and (fl==1):
        call_str = 'wget -q ftp://data-out.unavco.org/pub/products/troposphere/' + YEAR + '/' + DAY + '/' + FILE_PWV
        #print('Downloading GPS PWV data: %s' % DATE)
        os.system(call_str)
        #print('Download finish.')
    
    if fl ==1:
        Trop_PWV_GPS = gps_atm_dir + '/' + 'Global_GPS_PWV_' + DATE
        
        FILE0 = FILE_PWV.replace('.gz','')
        if os.path.isfile(FILE0):
            os.remove(FILE0)
        call_str = 'gzip -d ' + FILE_PWV
        os.system(call_str)
    
        call_str ='cp ' + FILE0 + ' ' + Trop_PWV_GPS
        os.system(call_str)
        
        return

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=4):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

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
    
    return ST

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
        DD.append(str(int(s1)))
    else:
        k = len(s1.split(','))
        for i in range(k):
            DD.append(str(int(s1.split(',')[i])))
            
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

    download_gps_atm.py date_list

'''    
    


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Download GPS data over SAR coverage.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('date_list',help='Date list for downloading trop list.')

    
    inps = parser.parse_args()

    return inps

    
####################################################################################################
def main(argv):
    
    inps = cmdLineParse()
    LIST = inps.date_list
    DATE = np.loadtxt(LIST,dtype=np.str)
    DATE = DATE.tolist()
    N=len(DATE)

    #for i in range(N):
    #    DATE0 = DATE[i]  
    #    call_str = 'download_gps_atm_date.py ' + DATE0
    #    os.system(call_str)
    
    parallel_process(DATE, download_atm_date, n_jobs=8, use_kwargs=False, front_num=1)
    
    

if __name__ == '__main__':
    main(sys.argv[1:])

