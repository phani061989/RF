# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:20:00 2019

@author: User
"""

import numpy as np

def connect_HDAWG(device_id):
    """This function tries to connect to a zi HDAWG device and returns a tuple
    with the device communication object and the AWG object"""
    
    from zhinst.utils import create_api_session
    
    (zi, device_id, _) = create_api_session(device_id, 6, required_devtype='HDAWG',required_err_msg='')

    # Create an instance of the AWG Module
    awg = zi.awgModule()
    awg.set('awgModule/device', device_id)

    return zi,awg

def apply_correction(dev,device_id,cal_dict,amplitude):
    """ This function applies the calibration to the ziAWG (initialized and passed to the function).
    
    pars:
        - dev: initialized and connected zhinst object
        - cal_dict: calibration dictionary v2.x.x
        - amplitude: it is possible to change the amplitude of the channels, if it is None (def),
        the amplitude used for calibration will be used.
    """
    
            
    chI = cal_dict['AWG chI']-1 #the numeration goes from 1 to 8 in the device, but from 0 to 7 in the programming...
    
    #apply offsets
    dev.setDouble('/{}/sigouts/{}/offset'.format(device_id,chI), cal_dict['Offset chI'])
    dev.setDouble('/{}/sigouts/{}/offset'.format(device_id,chI+1), cal_dict['Offset chQ']);
    
    #freq of sin1 and sin2
    dev.setDouble('/{}/oscs/{}/freq'.format(device_id,int(chI/2)), np.round(cal_dict['AWG frequency']*1e6,9)) #sin1 and sin2 freq
    
    
    
    
    
    #applying amplitude
    mod = dev.getInt('/{}/awgs/{}/outputs/0/modulation/mode'.format(device_id,int(chI/2)))
        
    
    if mod == 0: #modulation is off, the device should be in oscillator mode
        #enable sin1 and sin2
        dev.setInt('/{}/sines/{}/enables/0'.format(device_id,chI), 1)
        dev.setInt('/{}/sines/{}/enables/1'.format(device_id,chI+1), 1)    
    
        if cal_dict['Amplitude corrected channel'] == 'chI':
            dev.setDouble('/{}/sines/{}/amplitudes/0'.format(device_id,chI), amplitude*cal_dict['Amplitude ratio'])
            dev.setDouble('/{}/sines/{}/amplitudes/1'.format(device_id,chI+1), amplitude)
        elif cal_dict['Amplitude corrected channel'] == 'chQ':
            dev.setDouble('/{}/sines/{}/amplitudes/0'.format(device_id,chI+1), amplitude*cal_dict['Amplitude ratio'])
            dev.setDouble('/{}/sines/{}/amplitudes/1'.format(device_id,chI), amplitude)
        else:
            print('Amplitude corrected channel and used channels don\'t match\n')
            raise ValueError
    else: #mod is on, using AWG amplitude
        if cal_dict['Amplitude corrected channel'] == 'chI':
            dev.setDouble('/{}/sines/{}/amplitudes/0'.format(device_id,chI), 0)
            dev.setDouble('/{}/sines/{}/amplitudes/1'.format(device_id,chI+1), 0)
            dev.setDouble('/{}/awgs/{}/outputs/0/amplitude'.format(device_id,int(chI/2)), amplitude*cal_dict['Amplitude ratio']);
            dev.setDouble('/{}/awgs/{}/outputs/1/amplitude'.format(device_id,int(chI/2)), amplitude)
        elif cal_dict['Amplitude corrected channel'] == 'chQ':
            dev.setDouble('/{}/sines/{}/amplitudes/0'.format(device_id,chI), 0)
            dev.setDouble('/{}/sines/{}/amplitudes/1'.format(device_id,chI+1), 0)
            dev.setDouble('/{}/awgs/0/outputs/{}/amplitude'.format(device_id,chI+1), amplitude*cal_dict['Amplitude ratio']);
            dev.setDouble('/{}/awgs/0/outputs/{}/amplitude'.format(device_id,chI), amplitude)
        else:
            print('Amplitude corrected channel and used channels don\'t match\n')
            raise ValueError
    
    #sin1 to ch0 and sin2 to ch2
    #dev.setInt('/{}/sines/{}/enables/0'.format(device_id,chI), 1)
    #dev.setInt('/{}/sines/{}/enables/1'.format(device_id,chI+1), 1)
    
    #phase1 to 0, phase 2 to 90 or 270 + phase_corr
    if cal_dict['Sideband'] == 'LSB':
        angle = 270
    else:
        angle = 90
        
    dev.setDouble('/{}/sines/{}/phaseshift'.format(device_id,chI), 0)
    dev.setDouble('/{}/sines/{}/phaseshift'.format(device_id,chI+1), np.round(angle+cal_dict['Phase correction chQ'],9))

    
    #output on
    dev.setInt('/{}/sigouts/{}/on'.format(device_id,chI), 1)
    dev.setInt('/{}/sigouts/{}/on'.format(device_id,chI+1), 1)
