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
import glob

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False    
    
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
    
    t0 = float(t0)/3600/24
    hh = int(t0*24)
    mm = int((t0*24 - int(t0*24))*60)
    ss = (t0*24*60 - int(t0*24*60))*60
    ST = str(hh)+':'+str(mm)+':'+str(ss)
    h0 = str(hh)
    return ST,h0

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

def extract_sar_aps(data0):
    
    DATE0,JDSEC_SAR,GPS_NM = data0
    DD = GPS_NM
    
    root_path = os.getcwd()
    gig_dir = root_path + '/gigpy'
    gig_atm_dir = gig_dir  + '/atm'
    gig_atm_raw_dir = gig_dir  + '/atm/raw'
    gig_atm_sar_raw_dir = gig_dir  + '/atm/sar_raw'
    
    Trop_SAR = gig_atm_sar_raw_dir + '/SAR_GPS_Trop_' + str(DATE0)
    Trop_GPS = gig_atm_raw_dir + '/Global_GPS_Trop_' + str(DATE0)
    # remove the first four lines
    count = len(open(Trop_GPS,'r').readlines())
    
    tt_all = 'tt_all_' + str(DATE0)
    tt_name = 'tt_name_' + str(DATE0)
    
    call_str = 'sed -n 5,' + str(count) + 'p ' +  Trop_GPS + ' >' + tt_all
    os.system(call_str)
    # extract all of the available station names
    call_str = "awk '{print $10}' " + tt_all + ' >' + tt_name
    os.system(call_str)
    GPS_all = np.loadtxt(tt_name, dtype = np.str)
    DD_all = GPS_all
    k_all=len(DD_all)
    #print(k_all)
    DD_all.tolist()
    RR = np.zeros((k_all,),dtype = bool)
    for i in range(k_all):
        k0 = DD_all[i]
        if k0 in DD:
            RR[i] = 1
    data = np.loadtxt(tt_all, dtype = np.str)
    data_use = data[RR]
    #print(data_use.shape)
    JDSEC_all = data_use[:,0]
    #JDSEC_all = np.asarray(JDSEC_all,dtype = int)
    nn = len(JDSEC_all)
    RR2 = np.zeros((nn,),dtype = bool)
    for i in range(nn):
        #print(int(float(JDSEC_all[i])))
        if int(float(JDSEC_all[i])) == JDSEC_SAR:
            RR2[i] =1
    data_use_final = data_use[RR2]
    #print(data_use_final.shape)     
    np.savetxt(Trop_SAR,data_use_final,fmt='%s', delimiter=',')    
    os.remove(tt_all)
    os.remove(tt_name)
    
    return
        
def extract_sar_pwv(data0):
        
    DATE0,HH0,GPS_NM = data0
    DD = GPS_NM
    
    root_path = os.getcwd()
    gig_dir = root_path + '/gigpy'
    gig_atm_dir = gig_dir  + '/atm'
    gig_atm_raw_dir = gig_dir  + '/atm/raw'
    gig_atm_sar_raw_dir = gig_dir  + '/atm/sar_raw'
    
    DATE0 = unitdate(DATE0)
    
    PWV_GPS = gig_atm_raw_dir + '/Global_GPS_PWV_' + str(DATE0)
    PWV_SAR = gig_atm_sar_raw_dir + '/SAR_GPS_PWV_' + str(DATE0)
    
    tt_all = 'tt_all_' + str(DATE0)
    tt_name = 'tt_name_' + str(DATE0)
        
    count = len(open(PWV_GPS,'r').readlines())
    call_str = 'sed -n 2,' + str(count) + 'p ' +  PWV_GPS + ' >' + tt_all
    os.system(call_str)
    # extract all of the available station names
    call_str = "awk '{print $18}' " + tt_all + ' >' + tt_name
    os.system(call_str)
        
    GPS_all = np.loadtxt(tt_name, dtype = np.str)
    DD_all = GPS_all
    k_all=len(DD_all)
    #print(k_all)
    DD_all.tolist()
        
    data = np.loadtxt(tt_all, dtype = np.str)
    row,col = data.shape
    RR = np.zeros((row,),dtype = bool)
    for i in range(row):
        k0 = DD_all[i]
        if k0 in DD:
            RR[i] = 1
    data_use = data[RR]
    #print(data_use.shape)
    HH_all = data_use[:,2]
    nn = len(HH_all)
    RR2 = np.zeros((nn,),dtype = bool)
    for i in range(nn):
        if int(float(HH_all[i])) == HH0:
            RR2[i] =1
    data_use_final = data_use[RR2]
    #print(data_use_final.shape)     
    np.savetxt(PWV_SAR,data_use_final,fmt='%s', delimiter=',')  
    os.remove(tt_all)
    os.remove(tt_name)
    
    return
    

def write_gps_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):

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

def get_lack_datelist(date_list, date_list_exist):
    date_list0 = []
    for k0 in date_list:
        if k0 not in date_list_exist:
            date_list0.append(k0)
    return date_list0

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
###################################################################################################

INTRODUCTION = '''
    Extract the SAR synchronous GPS tropospheric products.
'''

EXAMPLE = '''EXAMPLES:

    extract_sar_atm.py search_gps.txt imaging_time --source UNAVCO --parallel 4
    extract_sar_atm.py search_gps.txt imaging_time --source UNR --date 20180101 
    extract_sar_atm.py search_gps.txt imaging_time --source UNR --date-txt date_list.txt

'''    
##################################################################################################   


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Extract the SAR synchronous GPS tropospheric products.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('gps_txt',help='GPS station text file.')
    parser.add_argument('imaging_time',help='Center line UTC sec.')
    parser.add_argument('-d', '--date', dest='date_list', nargs='*',help='date list to extract.')
    parser.add_argument('--date-txt', dest='date_txt',help='date list text to extract.')
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, 
                        help='Enable parallel processing and Specify the number of processors.[default: 1]')
    
    inps = parser.parse_args()

    return inps

    
####################################################################################################
def main(argv):
    
    inps = cmdLineParse()
    date_list = []
    if inps.date_list:
        date_list = inps.date_list
    if inps.date_txt:
        date_list2 = np.loadtxt(inps.date_txt,dtype=np.str)
        date_list2 = date_list2.tolist()
        for list0 in date_list2:
            if (list0 not in date_list) and is_number(list0):
                date_list.append(list0)
    
    root_path = os.getcwd()
    gig_dir = root_path + '/gigpy'
    gig_atm_dir = gig_dir  + '/atm'
    gig_atm_raw_dir = gig_dir  + '/atm/raw'
    gig_atm_sar_raw_dir = gig_dir  + '/atm/sar_raw'
    
    if not os.path.isdir(gig_atm_sar_raw_dir):
        os.mkdir(gig_atm_sar_raw_dir)
    
    print('')
    print('--------------------------------------------')
    if (not inps.date_txt) and (not inps.date_list):
        print('Obtain the GPS data-date automatically from %s' % gig_atm_raw_dir)
        date_list = [os.path.basename(x).split('_')[3] for x in glob.glob(gig_atm_raw_dir + '/Global_GPS_Trop*')]
        
    date_list_exist = [os.path.basename(x).split('_')[3] for x in glob.glob(gig_atm_sar_raw_dir + '/SAR_GPS_Trop*')]
    date_list_extract = get_lack_datelist(date_list, date_list_exist)
    
    print('Existed SAR synchoronous dataset: %s' % str(len(date_list_exist)))
    print('Number of dataset need to be extracted: %s' % str(len(date_list_extract)))
    print('')
    
    aps_list_no = []
    pwv_list_no = []
    
    aps_list_yes = []
    pwv_list_yes = []
    for k0 in date_list_extract:
        s0_aps = gig_atm_raw_dir + '/Global_GPS_Trop_' + k0
        s0_pwv = gig_atm_raw_dir + '/Global_GPS_PWV_' + k0
        
        if not os.path.isfile(s0_aps): aps_list_no.append(k0)
        else: aps_list_yes.append(k0)
            
        if not os.path.isfile(s0_pwv): pwv_list_no.append(k0)
        else: pwv_list_yes.append(k0)
    
    if len(aps_list_no) > 0:
        print('The following raw-delay data are not downloaded/available: ')
        for k0 in aps_list_no:
            print(k0)
        print('Number of the delay-data to be extracted: %s' % str(len(aps_list_yes)))
    else:
        print('All of the to be extracted delay-data are available.')
        print('Number of the delay-data to be extracted: %s' % str(len(aps_list_yes)))
        
    print('')
    print('Start to extract the SAR synchoronous tropospheric dataset ...')
    print('Number of the parallel processors used: %s' % str(inps.parallelNumb))
    
    # adjust the SAR acquisition time
    t0 = inps.imaging_time
    SST,HH = yyyy2yyyymmddhhmmss(float(t0))
    t0 =float(t0)/3600/24
    t0 = float(t0)*24*12
    t0 = round(t0)
    t0 = t0 * 300
    
    hh0 = float(t0)/3600
    HH0 = int(round(float(hh0)/2)*2) # for extracting the atmospheric water vapor data
    Tm =str(int(t0))
    
    # adjust the gps_txt file
    GPS = np.loadtxt(inps.gps_txt, dtype = np.str)
    DD = GPS[:,0]
    k=len(DD)
    DD.tolist()
    
    
    data_parallel = []
    for k0 in aps_list_yes:
        DATE0 = unitdate(k0)
        
        dt = dateutil.parser.parse(DATE0)
        time = astropy.time.Time(dt)
        JD = time.jd - 2451545.0
        JDSEC = JD*24*3600        
        JDSEC_SAR = int(JDSEC + t0)
        
        data0 = (k0,JDSEC_SAR,DD)
        data_parallel.append(data0)
    
    # Start to extract data using parallel process
    parallel_process(data_parallel, extract_sar_aps, n_jobs=inps.parallelNumb, use_kwargs=False, front_num=1)
    
    print('')
    print('--------------------------------------------')
    if len(pwv_list_no) > 0:
        print('The following raw-pwv data are not downloaded/available: ')
        for k0 in pwv_list_no:
            print(k0)
        print('Number of the pwv-data to be extracted: %s' % str(len(pwv_list_yes)))
    else:
        print('All of the to be extracted pwv-data are available.')
        print('Number of the pwv-data to be extracted: %s' % str(len(pwv_list_yes)))
    
    
    print('')
    print('Start to extract the SAR synchronous atmospheric water vapor dataset ...')
    print('Number of the parallel processors used: %s' % str(inps.parallelNumb))
    
    data_parallel = []
    for k0 in aps_list_yes:
        
        data0 = (k0,HH0,DD)
        data_parallel.append(data0)
    
    # Start to extract data using parallel process
    parallel_process(data_parallel, extract_sar_pwv, n_jobs=inps.parallelNumb, use_kwargs=False, front_num=1)
    
     
################### generate gps_aps.h5 & gps_pwv.h5 ################################
    print('')
    print('')
    print('--------------------------------------------')
    print('Start to convert the availabe SAR synchronous GPS data into hdf5 file ...')
    print('')
    PATH = gig_atm_sar_raw_dir
    #data_list = glob.glob(PATH + '/SAR_GPS_Trop*')
    date_list = [os.path.basename(x).split('_')[3] for x in glob.glob(PATH + '/SAR_GPS_Trop*')]
    
    date_list = sorted(date_list)
    data_list = [(PATH + '/SAR_GPS_Trop_' + x) for x in date_list]
    N = len(date_list)
    NN = 0
    
    N1 = np.zeros((N,))
    for i in range(N):
        #print(data_list[i])
        GPS0 = np.loadtxt(data_list[i],delimiter=',', dtype = np.str)
        y0= GPS0[:,0]
        N0 = len(y0)
        N1[i] = N0
        if NN < N0:
            NN = N0
    
    GPS_TD = np.zeros((N,NN),dtype=np.float32)
    GPS_HD = np.zeros((N,NN),dtype=np.float32)
    GPS_WD = np.zeros((N,NN),dtype=np.float32)
    
    date00 = np.zeros((N,NN)) #
    GPS_NM = date00.astype(np.string_)
    GPS_NM = np.asarray(GPS_NM, dtype='<S8')
    
    
    for i in range(N):
        GPS0 = np.loadtxt(data_list[i],delimiter=',', dtype = np.str)
        n0 = len(GPS0[:,8])    
        GPS_TD[i,0:n0] = np.asarray(GPS0[:,1],dtype = np.float32)
        GPS_WD[i,0:n0] = np.asarray(GPS0[:,3],dtype = np.float32)
        GPS_HD[i,0:n0] = np.asarray(GPS0[:,2],dtype = np.float32)
        GPS_NM[i,0:n0] = GPS0[:,9]
        
    datasetDict =dict()
    
    datasetDict['gps_name'] = np.asarray(GPS[:,0],dtype = np.string_)
    datasetDict['gps_height'] = np.asarray(GPS[:,3],dtype = np.float32)
    datasetDict['gps_lat'] = np.asarray(GPS[:,1],dtype = np.float32)
    datasetDict['gps_lon'] = np.asarray(GPS[:,2],dtype = np.float32)
    
    datasetDict['wzd'] = GPS_WD
    datasetDict['hzd'] = GPS_HD
    datasetDict['tzd'] = GPS_TD
    datasetDict['date'] = np.asarray(date_list,dtype = np.string_)
    datasetDict['station'] = GPS_NM

    meta = {}
    meta['UNIT'] = 'm'
    meta['DATA_TYPE'] = 'aps'
    
    write_gps_h5(datasetDict, 'gps_aps.h5', metadata = meta, ref_file = None, compression = None)
    
#####################################################################
    #data_list = glob.glob(PATH + '/SAR_GPS_PWV*')
    date_list = [os.path.basename(x).split('_')[3] for x in glob.glob(PATH + '/SAR_GPS_PWV*')]
    
    date_list = sorted(date_list)
    data_list = [(PATH + '/SAR_GPS_PWV_' + x) for x in date_list]
    
    N = len(date_list)
    NN = 0
    
    N1 = np.zeros((N,))
    for i in range(N):
        #print(data_list[i])
        GPS0 = np.loadtxt(data_list[i],delimiter=',', dtype = np.str)
        y0= GPS0[:,8]
        N0 = len(y0)
        N1[i] = N0
        if NN < N0:
            NN = N0
    
    GPS_PW = np.zeros((N,NN),dtype=np.float32)
    GPS_TD = np.zeros((N,NN),dtype=np.float32)
    GPS_HD = np.zeros((N,NN),dtype=np.float32)
    GPS_WD = np.zeros((N,NN),dtype=np.float32)
    
    date00 = np.zeros((N,NN)) #
    GPS_NM = date00.astype(np.string_)
    GPS_NM = np.asarray(GPS_NM, dtype='<S8')
    GPS_TE = np.zeros((N,NN),dtype=np.float32)
    
    for i in range(N):
    
        GPS0 = np.loadtxt(data_list[i],delimiter=',', dtype = np.str)
        n0 = len(GPS0[:,8])
        
        GPS_PW[i,0:n0] = np.asarray(GPS0[:,8],dtype = np.float32)
        GPS_TD[i,0:n0] = np.asarray(GPS0[:,5],dtype = np.float32)
        GPS_WD[i,0:n0] = np.asarray(GPS0[:,6],dtype = np.float32)
        GPS_HD[i,0:n0] = np.asarray(GPS0[:,12],dtype = np.float32)
        GPS_NM[i,0:n0] = GPS0[:,17]
        GPS_TE[i,0:n0] = np.asarray(GPS0[:,11],dtype = np.float32)
        
    datasetDict =dict()
    
    datasetDict['gps_name'] = np.asarray(GPS[:,0],dtype = np.string_)
    datasetDict['gps_height'] = np.asarray(GPS[:,3],dtype = np.float32)
    datasetDict['gps_lat'] = np.asarray(GPS[:,1],dtype = np.float32)
    datasetDict['gps_lon'] = np.asarray(GPS[:,2],dtype = np.float32)
    
    datasetDict['pwv'] = GPS_PW
    datasetDict['tem'] = GPS_TE
    datasetDict['wzd'] = GPS_WD
    datasetDict['hzd'] = GPS_HD
    datasetDict['tzd'] = GPS_TD
    datasetDict['date'] = np.asarray(date_list,dtype = np.string_)
    datasetDict['station'] = GPS_NM
    
    meta = {}
    meta['UNIT'] = 'mm'
    meta['DATA_TYPE'] = 'pwv'
    
    write_gps_h5(datasetDict, 'gps_pwv.h5', metadata=meta, ref_file=None, compression=None)
       
    #########
    print('Done.')     
    sys.exit(1)        

if __name__ == '__main__':
    main(sys.argv[1:])

