############################################################
# Program is part of GigPy                                 #
# Copyright 2019 Yunmeng Cao                               #
# Contact: ymcmrs@gmail.com                                #
############################################################

import os
import numpy as np
from gigpy import _utils as ut


def extract_sar_delay_unavco(data0):
    
    DATE0,JDSEC_SAR,GPS_NM,root_path = data0
    DD = GPS_NM
    
    #root_path = os.getcwd()
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
        
def extract_sar_pwv_unavco(data0):
        
    DATE0,HH0,GPS_NM,root_path = data0
    DD = GPS_NM
    
    #root_path = os.getcwd()
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

def extract_sar_delay_unr(data0):
    date, stations, imaging_time, data_path, out_path = data0
    unrSec = str(round(int(imaging_time)/300)*300)
    
    if len(unrSec) ==1: unrSec = '0000' + unrSec
    elif len(unrSec) ==3: unrSec = '00' + unrSec
    elif len(unrSec) ==4: unrSec = '0' + unrSec
        
    year, doy = ut.yyyymmdd2yyyyddd(date)
    extract_time = year[2:4] + ':' + doy + ':' + unrSec
    
    out_file = out_path + '/SAR_GPS_Trop_' + date
    if os.path.isfile(out_file):
        os.remove(out_file)
    nn = len(stations)
    for i in range(nn):
        file0 = data_path + '/' + stations[i].upper() + doy + '0.' + year[2:4] + 'zpd.gz'
        #STR0 = stations[i].upper() + ' ' + extract_time
        STR0 = '"' + stations[i].upper() + ' ' + extract_time + '"'
        call_str = 'zgrep -i ' + STR0 + ' ' + file0 + ' >>' + out_file
        os.system(call_str)
    
    return
    
    

