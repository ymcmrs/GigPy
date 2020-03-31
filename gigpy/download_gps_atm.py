#! /usr/bin/env python
#################################################################
###  This program is part of GigPy  v1.0                      ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################


import numpy as np
import sys
import os
import argparse

from gigpy import _utils as ut

def download_unr_para(data0):
    date,stationName,path = data0
    ut.download_atm_unr_station(date,stationName,path)
    return

def download_unavco_para(data0):
    date,path = data0
    ut.download_atm_unavco(date,path)
    return


def check_existed_file_unr(date,stationList,path):
    year,doy = ut.yyyymmdd2yyyyddd(date)
    stationDownload = []
    for i in range(len(stationList)):
        k0 = path + '/' + stationList[i].upper()+doy + '0.' + year[2:4] + 'zpd.gz' 
        if not os.path.isfile(k0):
            stationDownload.append(stationList[i])
    return stationDownload

def check_existed_file_unavco(date_list,path):
    return


    
###################################################################################################

INTRODUCTION = '''

Download GPS troposphere data based on date from UNAVCO or Nevada Geodetic Lab.
    
'''

EXAMPLE = '''EXAMPLES:

    download_gps_atm.py --date-list-txt date_list_txt --source UNAVCO --parallel 4
    download_gps_atm.py --date-list 20180101 20181002 20190304 --source UNR --station-txt gps_stations_info.txt 
    download_gps_atm.py --date-list-txt date_list_txt --source UNR --station-txt gps_stations_info.txt --parallel 8

'''    
    


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Download GPS data over SAR coverage.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('--date-list',dest='date_list',nargs='*',help='Date list for downloading trop list.')
    parser.add_argument('--date-list-txt',dest='date_list_txt',help='Text file of date list for downloading trop list.')
    parser.add_argument('-s','--source', dest='source', choices = {'unavco','unr'}, default = 'unr',help = 'source of the GPS data.[default: unavco]')
    parser.add_argument('--station', dest='station',nargs='*',help = 'names of the GPS stations to be downloaded.')
    parser.add_argument('--station-txt', dest='station_txt',help = 'text file of the GPS stations.')
    parser.add_argument('--process-dir', dest='process_dir',help = 'processing directory. [default: the current directory]')
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, 
                        help='Enable parallel processing and Specify the number of processors.[default: 1]')

    inps = parser.parse_args()
    
    if not inps.date_list and not inps.date_list_txt:
        parser.print_usage()
        sys.exit(os.path.basename(sys.argv[0])+': error: date_list or date_list_txt should provide at least one.')

    return inps

    
####################################################################################################
def main(argv):
    
    inps = cmdLineParse()
    
    # Get date list
    date_list = []
    if inps.date_list:
        date_list = inps.date_list
    if inps.date_list_txt:
        date_list0 = ut.read_txt2list(inps.date_list_txt)      
        for i in range(len(date_list0)):
            k0 = date_list0[i]
            if k0 not in date_list:
                date_list.append(k0)
    
    date_list = ut.sort_unique_list(date_list)
    nn = len(date_list)
    
    # Get stations
    stationList = []
    if inps.station:
        stationList = inps.station
    if inps.station_txt:
        A0 = ut.read_txt2array(inps.station_txt)
        stationList0 = A0[:,0]
        for i in range(len(stationList0)):
            k0 = stationList0[i]
            if k0 not in stationList:
                stationList.append(k0)
    stationList = ut.sort_unique_list(stationList)
    
    # Get stations
    if inps.process_dir: root_path = inps.process_dir
    else: root_path = os.getcwd()
    
    print('')
    print('Process directory: %s' % root_path)
        
    # Check folder
    gig_dir = root_path + '/gigpy'
    gig_atm_dir = gig_dir  + '/atm'
    gig_atm_raw_dir = gig_dir  + '/atm/raw'
    print('Downloaded raw GPS data will be saved under: %s' % gig_atm_raw_dir)
    
    if not os.path.isdir(gig_dir):
        os.mkdir(gig_dir)
    
    if not os.path.isdir(gig_atm_dir):
        os.mkdir(gig_atm_dir)
        
    if not os.path.isdir(gig_atm_raw_dir):
        os.mkdir(gig_atm_raw_dir)
    
    
    data_para = []
    if inps.source =='unavco':
        print('')
        print('Download gps tropospheric products from UNAVCO.')
        print('Number of date to be downloaded: %s' % str(nn))
        print('Used processor number: %s' % str(inps.parallelNumb))
        print('')
        for i in range(nn):
            data0 = [date_list[i],gig_atm_raw_dir]
            data_para.append(data0)
        ut.parallel_process(data_para, download_unavco_para, n_jobs=inps.parallelNumb, use_kwargs=False)
        
    elif inps.source =='unr':
        print('')
        print('Download gps tropospheric products from Nevada Geodetic Lab.')
        print('Number of date to be downloaded: %s' % str(nn))
        print('Number of GPS stations to be download: %s' % str(len(stationList)))
        print('Used processor number: %s' % str(inps.parallelNumb))
        print('')
        for i in range(nn):
            SS0 = str(date_list[i]) + ' (' + str(int(i+1))+'/'+str(nn)+')'
            print('Starting to download GPS data for date: %s' % SS0)
            root_path0 = gig_atm_raw_dir + '/' + date_list[i]
            if not os.path.isdir(root_path0):
                os.mkdir(root_path0) 
            stationDownloadList0 = ut.get_download_unr(date_list[i],stationList)
            stationDownloadList = check_existed_file_unr(date_list[i],stationDownloadList0,root_path0)
            data_para = []
            for j in range(len(stationDownloadList)):
                data0 = [date_list[i],stationDownloadList[j],root_path0]
                data_para.append(data0)
            if len(stationDownloadList) > 0:  
                ut.parallel_process(data_para, download_unr_para, n_jobs=inps.parallelNumb, use_kwargs=False)
                
            #    FILE0 = gig_atm_raw_dir + '/' + 'Global_GPS_Trop_' + date_list[i]
            #    open(FILE0,'a').close()
            #else:
            #    FILE0 = gig_atm_raw_dir + '/' + 'Global_GPS_Trop_' + date_list[i]
            #    open(FILE0,'a').close()
            #    print('Done.')
                
    
if __name__ == '__main__':
    main(sys.argv[1:])

