############################################################
# Program is part of GigPy                                 #
# Copyright 2019 Yunmeng Cao                               #
# Contact: ymcmrs@gmail.com                                #
############################################################
from datetime import datetime
import urllib.request
import os
import numpy as np
import random
import h5py
from pathlib import Path

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

unr_trop_url = 'ftp://gneiss.nbmg.unr.edu/trop/'
unavco_trop_url = 'ftp://data-out.unavco.org/pub/products/troposphere/'

def generate_random_name(sufix):
    
    nowTime = datetime.now().strftime("%Y%m%d%H%M%S") #generate the present time
    randomNum = random.randint(0,100)
    randomNum1 = random.randint(0,100)
    Nm = nowTime + str(randomNum)+ str(randomNum1)  + sufix
    
    return Nm

def get_unr_atm_name(year,doy,stationName):
    NM = stationName.upper()
    NM_unr = NM  + str(doy) + '0.' + year[2:4] + 'zpd.gz'  
    return NM_unr

def get_unr_atm_url(year,doy,stationName):
    nm_unr = get_unr_atm_name(year,doy,stationName)
    url0 = unr_trop_url + year + '/' + doy + '/' + nm_unr
    return nm_unr, url0

def yyyymmdd2yyyyddd(date):
    dt = datetime.strptime(date, "%Y%m%d") # get datetime object
    day_of_year = (dt - datetime(dt.year, 1, 1))  # Jan the 1st is day 1
    doy = day_of_year.days + 1
    year = dt.year
    
    doy = str(doy)
    if len(doy) ==1:
        doy = '00' + doy
    elif len(doy) ==2:
        doy ='0'+ doy
    year = str(year)
    
    return year, doy

def read_txt2list(txt):
    A = np.loadtxt(txt,dtype=np.str)
    if isinstance(A[0],bytes):
        A = A.astype(str)
    A = list(A)    
    return A

def read_txt2array(txt):
    A = np.loadtxt(txt,dtype=np.str)
    if isinstance(A[0],bytes):
        A = A.astype(str)
    #A = list(A)    
    return A

def get_filelist_filesize(url0):
    ttt = generate_random_name('.txt')
    call_str = 'curl -s ' + url0 + ' > ' + ttt 
    os.system(call_str)
    A = read_txt2array(ttt)
    
    A_size = A[:,4]
    A_size = A_size.astype(int)
    A_size = A_size/1024/1024  #bytes to Mb
    A_name = A[:,8]
    if os.path.isfile(ttt):
        os.remove(ttt)
        
    return A_name, A_size
    
def get_stationlist_atm_unr(date):
    ttt = generate_random_name('.txt')
    year, doy = yyyymmdd2yyyyddd(date)
    url0 = unr_trop_url + year + '/' + doy + '/'
    call_str = 'curl -l -s ' + url0 + ' > ' + ttt 
    os.system(call_str)
    A = read_txt2list(ttt)
    if os.path.isfile(ttt):
        os.remove(ttt)
    
    unr_station_list = []
    for i in range(len(A)):
        a0 = A[i]
        k0 = a0[0:4]
        unr_station_list.append(k0)
        
    return unr_station_list
    
def get_downloadFile_unavo(url0):
    A_name, A_size = get_filelist_filesize(url0)
    nn = len(A_name)
    Apwv = ''
    Atzd = ''
    b0_pwv = 0
    b0_tzd = 0
    
    for i in range(nn):
        k0 = A_name[i]
        if 'nmt' in k0:
            b_pwv = A_size[i]
            if b_pwv > b0_pwv:
                b0_pwv = b_pwv
                Apwv = k0
        if 'cwu' in k0:
            b_tzd = A_size[i]
            if b_tzd > b0_tzd:
                b0_tzd = b_tzd
                Atzd = k0
    return Atzd, Apwv

def download_atm_unavco(date,path):
    year, doy = yyyymmdd2yyyyddd(date)
    ttt = 'ttt_' + date
    url0 = unavco_trop_url + str(year) + '/' + str(doy) + '/'
    Atzd, Apwv = get_downloadFile_unavo(url0)
    Atzd_download = unavco_trop_url + str(year) + '/' + str(doy) + '/' + Atzd
    Apwv_download = unavco_trop_url + str(year) + '/' + str(doy) + '/' + Apwv
    
    out_tzd = path + '/' + Atzd
    out_pwv = path + '/' + Apwv
    
    if len(Atzd) > 0 :
        # Download tzd file
        urllib.request.urlretrieve(Atzd_download,out_tzd)
        Trop_GPS = path + '/' + 'Global_GPS_Trop_' + date 
        
        FILE0 = out_tzd.replace('.gz','')
        call_str = 'gzip -d ' + out_tzd
        os.system(call_str)
        
        call_str ='cp ' + FILE0 + ' ' + Trop_GPS
        os.system(call_str)
        
        if os.path.isfile(FILE0):
            os.remove(FILE0)
        
    if len(Apwv) > 0:
        #Download pwv file
        urllib.request.urlretrieve(Apwv_download,out_pwv)   
        Trop_PWV_GPS = path + '/' + 'Global_GPS_PWV_' + date
         
        FILE0 = out_pwv.replace('.gz','')
        call_str = 'gzip -d ' + out_pwv
        os.system(call_str)
        
        call_str ='cp ' + FILE0 + ' ' + Trop_PWV_GPS
        os.system(call_str)
        
        if os.path.isfile(FILE0):
            os.remove(FILE0)

    return
    
def download_atm_unr_station(date,stationName,path):
    year, doy = yyyymmdd2yyyyddd(date)
    name0 = get_unr_atm_name(year,doy,stationName)
    url0 = unr_trop_url + year + '/' + doy + '/' + name0
    #print(url0)
    out0 = path + '/' + name0
    urllib.request.urlretrieve(url0,out0)       
    
    return

def download_atm_unr(date,stationList,path):
    unr_stations = get_stationlist_atm_unr(date)
    nn = len(stationList)
    station_download = []
    for i in range(nn):
        k0 = stationList[i]
        if k0 in unr_stations:
            station_download.append(k0)
    nn_download = len(station_download)
    for i in range(nn_download):
        download_atm_unr_station(date,station_download[i],path)
    
    Na=len(path.split('/'))
    SS = '/' + path.split('/')[Na-1]
    path0 = path.split(SS)[0]

    FILE0 = path0 + '/' + 'Global_GPS_Trop_' + date
    print(FILE0)
    open(FILE0,'a').close()
    #f = open(FILE0,'w')
    #f.close()
    #call_str = 'touch ' + FILE0
    #os.system(call_str)
    return

def get_download_unr(date,stationList):
    unr_stations = get_stationlist_atm_unr(date)
    nn = len(stationList)
    station_download = []
    for i in range(nn):
        k0 = stationList[i]
        if k0 in unr_stations:
            station_download.append(k0)
    return station_download
    
def parallel_process(array, function, n_jobs=16, use_kwargs=False):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return [function(**a) if use_kwargs else function(a) for a in tqdm(array[:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[:]]
        else:
            futures = [pool.submit(function, a) for a in array[:]]
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
    return out    

def sort_unique_list(numb_list):
    list_out = sorted(set(numb_list))
    return list_out 

############################ extract data #####################
def extract_sar_delay_unavco(data0):
    
    DATE0,JDSEC_SAR,GPS_NM  = data0
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
    
    #call_str = 'sed -n 5,' + str(count) + 'p ' +  Trop_GPS + ' >' + tt_all
    call_str = "sed -e '/#/d' " + Trop_GPS + " >" + tt_all
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

def extract_sar_delay_unr(data0):
    date, stations, imaging_time, data_path, out_path = data0
    unrSec = str(round(int(imaging_time)/300)*300)
    
    if len(unrSec) ==1: unrSec = '0000' + unrSec
    elif len(unrSec) ==3: unrSec = '00' + unrSec
    elif len(unrSec) ==4: unrSec = '0' + unrSec
        
    year, doy = yyyymmdd2yyyyddd(date)
    extract_time = year[2:4] + ':' + doy + ':' + unrSec
    
    out_file = out_path + '/SAR_GPS_Trop_' + date
    if os.path.isfile(out_file):
        os.remove(out_file)
    nn = len(stations)
    for i in range(nn):
        file0 = data_path + '/' + stations[i].upper() + doy + '0.' + year[2:4] + 'zpd.gz'
        #STR0 = stations[i].upper() + ' ' + extract_time
        STR0 = '"' + stations[i].upper() + ' ' + extract_time + '"'
        if os.path.isfile(file0):
            call_str = 'zgrep -i ' + STR0 + ' ' + file0 + ' >>' + out_file
            os.system(call_str)
    
    return

############################# write & read #####################################
def read_attr(fname):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        atr = dict(f.attrs)
        
    return atr

def read_hdf5(fname, datasetName=None, box=None):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        data = f[datasetName][:]
        atr = dict(f.attrs)
        
    return data, atr

def get_dataNames(FILE):
    with h5py.File(FILE, 'r') as f:
        dataNames = []
        for k0 in f.keys():
            dataNames.append(k0)
    return dataNames

def read_unr_sar_file(unr_sar_file):
    
    GPS0 = read_txt2array(unr_sar_file)
    
    GPS_NM = GPS0[:,0]
    GPS_TD = GPS_TD[:,2]/1000 # convert to meters
    
    return GPS_NM, GPS_TD

def read_unavco_sar_file(unavco_sar_file):
    
    GPS0 = read_txt2array(unavco_sar_file)
 
    GPS_TD = np.asarray(GPS0[:,1],dtype = np.float32)
    GPS_WD = np.asarray(GPS0[:,3],dtype = np.float32)
    GPS_HD = np.asarray(GPS0[:,2],dtype = np.float32)
    GPS_NM = GPS0[:,9]
    
    return GPS_NM, GPS_TD, GPS_WD, GPS_HD

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

#######################################################################

class progressBar:
    """Creates a text-based progress bar. Call the object with 
    the simple print command to see the progress bar, which looks 
    something like this:
    [=======> 22%       ]
    You may specify the progress bar's min and max values on init.

    note:
        modified from mintPy (https://github.com/insarlab/MintPy/wiki)
        Code originally from http://code.activestate.com/recipes/168639/

    example:
        from mintpy.utils import ptime
        date12_list = ptime.list_ifgram2date12(ifgram_list)
        prog_bar = ptime.progressBar(maxValue=1000, prefix='calculating:')
        for i in range(1000):
            prog_bar.update(i+1, suffix=date)
            prog_bar.update(i+1, suffix=date12_list[i])
        prog_bar.close()
    """

    def __init__(self, maxValue=100, prefix='', minValue=0, totalWidth=70, print_msg=True):
        self.prog_bar = "[]"  # This holds the progress bar string
        self.min = minValue
        self.max = maxValue
        self.span = maxValue - minValue
        self.suffix = ''
        self.prefix = prefix

        self.print_msg = print_msg
        ## calculate total width based on console width
        #rows, columns = os.popen('stty size', 'r').read().split()
        #self.width = round(int(columns) * 0.7 / 10) * 10
        self.width = totalWidth
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.amount = 0  # When amount == max, we are 100% done
        self.update_amount(0)  # Build progress bar string

    def update_amount(self, newAmount=0, suffix=''):
        """ Update the progress bar with the new amount (with min and max
        values set at initialization; if it is over or under, it takes the
        min or max value as a default. """
        if newAmount < self.min:
            newAmount = self.min
        if newAmount > self.max:
            newAmount = self.max
        self.amount = newAmount

        # Figure out the new percent done, round to an integer
        diffFromMin = np.float(self.amount - self.min)
        percentDone = (diffFromMin / np.float(self.span)) * 100.0
        percentDone = np.int(np.round(percentDone))

        # Figure out how many hash bars the percentage should be
        allFull = self.width - 2 - 18
        numHashes = (percentDone / 100.0) * allFull
        numHashes = np.int(np.round(numHashes))

        # Build a progress bar with an arrow of equal signs; special cases for
        # empty and full
        if numHashes == 0:
            self.prog_bar = '%s[>%s]' % (self.prefix, ' '*(allFull-1))
        elif numHashes == allFull:
            self.prog_bar = '%s[%s]' % (self.prefix, '='*allFull)
            if suffix:
                self.prog_bar += ' %s' % (suffix)
        else:
            self.prog_bar = '[%s>%s]' % ('='*(numHashes-1), ' '*(allFull-numHashes))
            # figure out where to put the percentage, roughly centered
            percentPlace = int(len(self.prog_bar)/2 - len(str(percentDone)))
            percentString = ' ' + str(percentDone) + '% '
            # slice the percentage into the bar
            self.prog_bar = ''.join([self.prog_bar[0:percentPlace],
                                     percentString,
                                     self.prog_bar[percentPlace+len(percentString):]])
            # prefix and suffix
            self.prog_bar = self.prefix + self.prog_bar
            if suffix:
                self.prog_bar += ' %s' % (suffix)
            # time info - elapsed time and estimated remaining time
            if percentDone > 0:
                elapsed_time = time.time() - self.start_time
                self.prog_bar += '%5ds / %5ds' % (int(elapsed_time),
                                                  int(elapsed_time * (100./percentDone-1)))

    def update(self, value, every=1, suffix=''):
        """ Updates the amount, and writes to stdout. Prints a
         carriage return first, so it will overwrite the current
          line in stdout."""
        if value % every == 0 or value >= self.max:
            self.update_amount(newAmount=value, suffix=suffix)
            if self.print_msg:
                sys.stdout.write('\r' + self.prog_bar)
                sys.stdout.flush()

    def close(self):
        """Prints a blank space at the end to ensure proper printing
        of future statements."""
        if self.print_msg:
            print(' ')