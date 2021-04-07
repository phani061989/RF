# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:18:01 2020

@author: User
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from UtilitiesLib import filetodictionary

def get_data_from_list(folder,plot=True):
    """
    This functions opens the files in the specified folder, load them if they
    are valid calibration files and extracts the calibration results.
    
    pars:
        - folder: the folder that contains the calibration files
        - plot (def True): plots the results at the end with matplot
        
    returns:
        - frequencies, LSB(f),Carrier(f),RSB(f)
    
    """
    
    f_list = glob.glob(folder+'/*')
    x = []
    l = []
    c = []
    r = []
    
    skipped=0
    
    for i,dict_file in enumerate(f_list):
        try:
            tmp = filetodictionary(dict_file)
            x.append(tmp[tmp['Sideband']])
            res= tmp['Calibration results']
            l.append(res[0])
            c.append(res[1])
            r.append(res[2])
        except:
            skipped+=1
            
    if skipped != 0:
        print('WARNING: skipped {} files\n'.format(skipped))
        
    x = np.array(x)
    l = np.array(l)
    c = np.array(c)
    r = np.array(r)
    
    if plot:
        plt.figure(0)
        plt.plot(x,l,'-b' )
        plt.ylabel('PSD (dBm)')
        plt.xlabel('Frequency (GHz)')
        plt.title('LSB')
        
        plt.figure(1)
        plt.plot(x,c,'-b' )
        plt.ylabel('PSD (dBm)')
        plt.xlabel('Frequency (GHz)')
        plt.title('Carrier')
        
        plt.figure(2)
        plt.plot(x,r,'-b' )
        plt.ylabel('PSD (dBm)')
        plt.xlabel('Frequency (GHz)')
        plt.title('RSB')
    
    return x,l,c,r

def check_calibrations_in_folder(cal,folder,amp_to_check,plot=True):
    """
    This functions opens the files in the specified folder, load them if they
    are valid calibration files and measure again the sidebands and carrier
    for the given amp_to_check AWG amplitude.
    
    pars:
        - cal: an initialized IQCal object
        - folder: the folder that contains the calibration files
        - amp_to_check: the AWG channel amplitude to check
        - plot (def True): plots the results at the end with matplot
        
    returns:
        - frequencies, LSB(f),Carrier(f),RSB(f)
    
    """
    import tqdm
    f_list = glob.glob(folder+'/*')
    
    
    
    x2 = []
    l2 = []
    c2 = []
    r2 = []
    
    cal.AWG_calibration_amplitude(amp_to_check)
    
    skipped=0 #it contains the number of files that can't be opened
    for file in tqdm.tqdm( f_list):
        try:
            tmp = filetodictionary(file)
            cal.load_calibration(tmp)
            cal.set_LO()
            cal.apply_correction()
            
            
            x2.append(tmp[tmp['Sideband']])
            res = cal.measure_SB(0,0)
            l2.append(res[0])
            c2.append(res[1])
            r2.append(res[2])
        except:
            skipped+=1
    
    if skipped>0:
        print('WARNING: skipped {} files\n'.format(skipped))
    
    x2=np.array(x2)
    l2=np.array(l2)
    c2=np.array(c2)
    r2=np.array(r2)
    
    #close connections
    cal.close_SA_connection()    
    cal._sgLO.close_driver()
    
    if plot:
        plt.figure(0)
        plt.plot(x2,l2,'-r' )
        plt.ylabel('PSD (dBm)')
        plt.xlabel('Frequency (GHz)')
        plt.title('LSB')
        
        plt.figure(1)
        plt.plot(x2,c2,'-r' )
        plt.ylabel('PSD (dBm)')
        plt.xlabel('Frequency (GHz)')
        plt.title('Carrier')
        
        plt.figure(2)
        plt.plot(x2,c2,'-r' )
        plt.ylabel('PSD (dBm)')
        plt.xlabel('Frequency (GHz)')
        plt.title('RSB')
    
    return x2,l2,c2,r2
    
    