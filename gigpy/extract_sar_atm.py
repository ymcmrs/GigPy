#! /usr/bin/env python
#################################################################
###  This program is part of GigPy  v1.0                      ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################
import argparse
import sys
import os
import glob
import astropy.time
import dateutil.parser
import numpy as np

from gigpy import _utils as ut

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

def yyyy2yyyymmddhhmmss(t0):
    
    t0 = float(t0)/3600/24
    hh = int(t0*24)
    mm = int((t0*24 - int(t0*24))*60)
    ss = (t0*24*60 - int(t0*24*60))*60
    ST = str(hh)+':'+str(mm)+':'+str(ss)
    h0 = str(hh)
    return ST,h0

def get_lack_datelist(date_list, date_list_exist):
    date_list0 = []
    for k0 in date_list:
        if k0 not in date_list_exist:
            date_list0.append(k0)
    return date_list0

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
            
###################################################################################################

INTRODUCTION = '''
    Extract the SAR synchronous GPS tropospheric products.
'''

EXAMPLE = '''EXAMPLES:

    extract_sar_atm.py gps_station_info.txt imaging_time --date-txt date_list.txt --source UNAVCO 
    extract_sar_atm.py gps_station_info.txt imaging_time --source UNR --date 20180101 --parallel 4
    extract_sar_atm.py gps_station_info.txt imaging_time --source UNR --date-txt date_list.txt

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
    parser.add_argument('-s','--source', dest='source', choices = {'unavco','unr'}, default = 'unr',help = 'source of the GPS data.[default: unavco]')
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
    
    #aps_list_no = []
    #pwv_list_no = []
    
    #aps_list_yes = []
    #pwv_list_yes = []
    #for k0 in date_list_extract:
    #    s0_aps = gig_atm_raw_dir + '/Global_GPS_Trop_' + k0
    #    #s0_pwv = gig_atm_raw_dir + '/Global_GPS_PWV_' + k0
        
    #    if not os.path.isfile(s0_aps): aps_list_no.append(k0)
    #    else: aps_list_yes.append(k0)
            
    #    #if not os.path.isfile(s0_pwv): pwv_list_no.append(k0)
    #    #else: pwv_list_yes.append(k0)
    
    #if len(aps_list_no) > 0:
    #    print('The following raw-delay data are not downloaded/available: ')
    #    for k0 in aps_list_no:
    #        print(k0)
    #    print('Number of the delay-data to be extracted: %s' % str(len(aps_list_yes)))
    #else:
    #    print('All of the to be extracted delay-data are available.')
    #    print('Number of the delay-data to be extracted: %s' % str(len(aps_list_yes)))
        
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
    #GPS = np.loadtxt(inps.gps_txt, dtype = np.str)
    GPS = ut.read_txt2array(inps.gps_txt)
    DD = GPS[:,0]
    k=len(DD)
    N_total = len(DD)
    DD.tolist()
    
    
    data_parallel = []
    
    if inps.source =='unavco':
        for k0 in  date_list_extract:
            DATE0 = unitdate(k0)
        
            dt = dateutil.parser.parse(DATE0)
            time = astropy.time.Time(dt)
            JD = time.jd - 2451545.0
            JDSEC = JD*24*3600        
            JDSEC_SAR = int(JDSEC + t0)
        
            data0 = (k0,JDSEC_SAR,DD)
            data_parallel.append(data0)
    
        # Start to extract data using parallel process
        ut.parallel_process(data_parallel, ut.extract_sar_delay_unavco, n_jobs=inps.parallelNumb, use_kwargs=False)
    
    elif inps.source =='unr':
        #print(date_list_extract)
        if len(date_list_extract)>0:
            for k0 in  date_list_extract:
            
                DATE0 = unitdate(k0)
                data_path = gig_atm_raw_dir + '/' + DATE0
                data0 = (DATE0,DD,str(int(float(inps.imaging_time))),data_path,gig_atm_sar_raw_dir)
                #print(data0)
                data_parallel.append(data0)

            # Start to extract data using parallel process
            ut.parallel_process(data_parallel, ut.extract_sar_delay_unr, n_jobs=inps.parallelNumb, use_kwargs=False)
        else:
            print('All of the data have been extracted, skip this step.')
    #print('')
    #print('--------------------------------------------')
    #if len(pwv_list_no) > 0:
    #    print('The following raw-pwv data are not downloaded/available: ')
    #    for k0 in pwv_list_no:
    #        print(k0)
    #    print('Number of the pwv-data to be extracted: %s' % str(len(pwv_list_yes)))
    #else:
    #    print('All of the to be extracted pwv-data are available.')
    #    print('Number of the pwv-data to be extracted: %s' % str(len(pwv_list_yes)))
    
    
    #print('')
    #print('Start to extract the SAR synchronous atmospheric water vapor dataset ...')
    #print('Number of the parallel processors used: %s' % str(inps.parallelNumb))
    
    #data_parallel = []
    #for k0 in aps_list_yes:
        
    #    data0 = (k0,HH0,DD)
    #    data_parallel.append(data0)
    
    # Start to extract data using parallel process
    #parallel_process(data_parallel, extract_sar_pwv, n_jobs=inps.parallelNumb, use_kwargs=False, front_num=1)
    
     
################### generate gps_aps.h5 & gps_pwv.h5 ################################
    print('')
    print('')
    #print('--------------------------------------------')
    print('Start to convert the availabe SAR synchronous GPS data into hdf5 file ...')
    print('')
    PATH = gig_atm_sar_raw_dir
    #data_list = glob.glob(PATH + '/SAR_GPS_Trop*')
    #date_list = [os.path.basename(x).split('_')[3] for x in glob.glob(PATH + '/SAR_GPS_Trop*')]
    
    #print(date_list)
    date_list = sorted(date_list)
    data_list = [(PATH + '/SAR_GPS_Trop_' + x) for x in date_list]
    N = len(date_list)
    
    #NN = 0
    #N1 = np.zeros((N,))
    #for i in range(N):
    #    #print(data_list[i])
    #    GPS0 = np.loadtxt(data_list[i],delimiter=',', dtype = np.str)
    #    y0= GPS0[:,0]
    #    N0 = len(y0)
    #    N1[i] = N0
    #    if NN < N0:
    #        NN = N0
    
    GPS_TD = np.zeros((N,N_total),dtype=np.float32)
    GPS_HD = np.zeros((N,N_total),dtype=np.float32)
    GPS_WD = np.zeros((N,N_total),dtype=np.float32)
    
    date00 = np.zeros((N,N_total)) #
    GPS_NM = date00.astype(np.string_)
    GPS_NM = np.asarray(GPS_NM, dtype='<S8')
    
    for i in range(N):
        
        if inps.source =='unavco':
            GPS0 = np.loadtxt(data_list[i],delimiter=',', dtype = np.str)
        elif inps.source =='unr':
            GPS0 = ut.read_txt2array(data_list[i])
        n0 = len(GPS0[:,0])
        #print(n0)
        
        if inps.source =='unavco':
            GPS_WD[i,0:n0] = np.asarray(GPS0[:,3],dtype = np.float32)
            GPS_HD[i,0:n0] = np.asarray(GPS0[:,2],dtype = np.float32)
            
            GPS_TD[i,0:n0] = np.asarray(GPS0[:,1],dtype = np.float32)
            GPS_NM[i,0:n0] = GPS0[:,9]
            
        elif inps.source =='unr':
            
            GPS_TD[i,0:n0] = np.asarray(GPS0[:,2],dtype = np.float32)
            GPS_TD[i,0:n0] =GPS_TD[i,0:n0]/1000
            GPS_NM[i,0:n0] = GPS0[:,0]
        
    datasetDict =dict()
    
    datasetDict['gps_name'] = np.asarray(GPS[:,0],dtype = np.string_)
    datasetDict['gps_height'] = np.asarray(GPS[:,3],dtype = np.float32)
    datasetDict['gps_lat'] = np.asarray(GPS[:,1],dtype = np.float32)
    datasetDict['gps_lon'] = np.asarray(GPS[:,2],dtype = np.float32)
    
    if inps.source =='unavco': 
        datasetDict['wzd'] = GPS_WD
        datasetDict['hzd'] = GPS_HD
        datasetDict['tzd'] = GPS_TD
    datasetDict['tzd'] = GPS_TD
    datasetDict['date'] = np.asarray(date_list,dtype = np.string_)
    datasetDict['station'] = GPS_NM

    meta = {}
    meta['UNIT'] = 'm'
    meta['DATA_TYPE'] = 'aps'
    
    ut.write_h5(datasetDict, 'gps_delay.h5', metadata = meta, ref_file = None, compression = None)
    
#####################################################################
    ##data_list = glob.glob(PATH + '/SAR_GPS_PWV*')
    #date_list = [os.path.basename(x).split('_')[3] for x in glob.glob(PATH + '/SAR_GPS_PWV*')]
    
    #date_list = sorted(date_list)
    #data_list = [(PATH + '/SAR_GPS_PWV_' + x) for x in date_list]
    
    #N = len(date_list)
    #NN = 0
    
    #N1 = np.zeros((N,))
    #for i in range(N):
    #    #print(data_list[i])
    #    GPS0 = np.loadtxt(data_list[i],delimiter=',', dtype = np.str)
    #    y0= GPS0[:,8]
    #    N0 = len(y0)
    #    N1[i] = N0
    #    if NN < N0:
    #        NN = N0
    
    #GPS_PW = np.zeros((N,NN),dtype=np.float32)
    #GPS_TD = np.zeros((N,NN),dtype=np.float32)
    #GPS_HD = np.zeros((N,NN),dtype=np.float32)
    #GPS_WD = np.zeros((N,NN),dtype=np.float32)
    
    #date00 = np.zeros((N,NN)) #
    #GPS_NM = date00.astype(np.string_)
    #GPS_NM = np.asarray(GPS_NM, dtype='<S8')
    #GPS_TE = np.zeros((N,NN),dtype=np.float32)
    
    #for i in range(N):
    
    #   GPS0 = np.loadtxt(data_list[i],delimiter=',', dtype = np.str)
    #    n0 = len(GPS0[:,8])
        
    #    GPS_PW[i,0:n0] = np.asarray(GPS0[:,8],dtype = np.float32)
    #    GPS_TD[i,0:n0] = np.asarray(GPS0[:,5],dtype = np.float32)
    #    GPS_WD[i,0:n0] = np.asarray(GPS0[:,6],dtype = np.float32)
    #    GPS_HD[i,0:n0] = np.asarray(GPS0[:,12],dtype = np.float32)
    #    GPS_NM[i,0:n0] = GPS0[:,17]
    #    GPS_TE[i,0:n0] = np.asarray(GPS0[:,11],dtype = np.float32)
        
    #datasetDict =dict()
    
    #datasetDict['gps_name'] = np.asarray(GPS[:,0],dtype = np.string_)
    #datasetDict['gps_height'] = np.asarray(GPS[:,3],dtype = np.float32)
    #datasetDict['gps_lat'] = np.asarray(GPS[:,1],dtype = np.float32)
    #datasetDict['gps_lon'] = np.asarray(GPS[:,2],dtype = np.float32)
    
    #datasetDict['pwv'] = GPS_PW
    #datasetDict['tem'] = GPS_TE
    #datasetDict['wzd'] = GPS_WD
    #datasetDict['hzd'] = GPS_HD
    #datasetDict['tzd'] = GPS_TD
    #datasetDict['date'] = np.asarray(date_list,dtype = np.string_)
    #datasetDict['station'] = GPS_NM
    
    #meta = {}
    #meta['UNIT'] = 'mm'
    #meta['DATA_TYPE'] = 'pwv'
    
    #write_gps_h5(datasetDict, 'gps_pwv.h5', metadata=meta, ref_file=None, compression=None)
       
    #########
    print('Done.')     
    sys.exit(1)        

if __name__ == '__main__':
    main(sys.argv[1:])

