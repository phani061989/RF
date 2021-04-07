# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 10:33:30 2016

@author: oscar

v3.0.0:
I am using 2 new HVIs, one should be general for all the usual experiments, from Rabi to TEcho. The other is for using a "saturation pulse", so two generators instead of AWG + 1 signal generator.

I created new functions that creates the pulses, for example the readout_pulse sets the signal generator and the digitizer to perform a readout; the pi-pulse will add a pi_pulse to the sequence, ecc.

In the final section there are the functions used to sweep parameters, so peak scan will sweep the readout frequency, the two-tone will sweep the frequency of the excitation signal generator, ecc.

Finally I tried to make everything compatible with the new instruments.

v2.0.3:
- added a default to pulse_length in the initialization of a signal generator


v2.0.2:
added functions for some flux-vs-freq scan
v2.0.1

Fixed the T2 dephasing (osc_per)

v2.0.0

After some tests and after making more experience, I changed the HVIs in order to use always the modulation and reduce the memory usage of the board.
Check functions comments for details

"""

import keysightSD1 as sigd
from UtilitiesLib import *
from SIGGEN import Siggen
from UtilitiesLib import progressive_plot_2d
from SensorReader import SensorReader
from PULSE import Pulse
import numpy as np


sigd_dig_SF = 0.1 #Gsamples/s
sigd_awg_SF = 1. #Gsamples/s
sigd_dig_channel_used = 4
sigd_dig_ref_channel_used = 5

sigd_dig_amp_used = 3

print('SIGDTT v3.1.0')

def initialize_sdr(gain):
    from SDR14 import SDR14
    
    sd=SDR14()

    #External Chassis Clock
    sd.clock_ref()

    #AWG
    #setting AWG

    #SW trigger
    #sd.awg_trigger(3,'INT')
    

    #Normal trigger mode
    #sd.awg_trigin(3,0)
    
    
    #sd.awg_enablesegments(3,1)
    
    sd.GainOffset(1,gain)
    
    #sd.awg_continuous(3,0)
    #sd.awg_autorearm(3,1)
    
    
    
    return sd

def initialize_sg(IP):
    '''
    
    
    This function set a signal generatore in pulse mode (def pulse 2 us) and return the initialized variable to control the instrument.
    
    - IP: string that contains the IP address
    
    NOTE: power is set to -30 dBm
    '''
    
    sg = Siggen(IP)
    
    sg.frequency_reference('EXT')
    sg.ALC(0)
    
        
    sg.power(-30)
    
    sg.output(1) # I put the output on here because the generator in pulse mode doesn't send pulses until triggered
    
    return sg
    

def initialize_LO(IP,power=None,channel=1):
    '''
        
    This function set a generator to be used as a local oscillator. The 
    function returns a variable that can be used to control the instrument.
    
    - IP: string that contains the IP address
    - power: dBm (def None, it will not be set)
    - channel: used for multi channel signal generator (def 1)
    '''
    sg = Siggen(IP)
    
    try:
        sg.ALC("on")
        sg.pulse_triggered(0)
    except:
        pass
        
    if power is not None:
        sg.power(power,channel,False)
    
    sg.frequency_reference('EXT')
    
    sg.output(1,channel)
    
    return sg





##Next is a set of functions used to acquire data, 2 for debugging purposes, one to be used in the real measurement

def readout_debug_SDR14(sdr,dio,hvi,ave,rep_period=1000,points=None,acq_delay=None,board_delay=800,SF=0.8):
    '''This function will play the sequences and read the data.
    The hvi and sequences must be previously loaded.
    
    if points and acq_delay are None, the defaults value from the readout pulse will be taken.
    
    This debug function will return all the acquired waves'''
    
    #bugfix
    ave = np.int(ave+1)
    
    def points_check(points):
        points2 = np.ceil(points/32)*32
        discard = points2-points
        
        return np.int(points2),np.int(discard)
    
    dio.writeRegisterByNumber(0,ave)
    dio.writeRegisterByNumber(3,np.int(rep_period*100)) #Rep time
    
    #Setting the acquisition delay, usually it should be equal to the readout pulse delay
    
    
    #get delay and acquisition length in samples on SDR14
    
    #the acquired data must be a multiple of 128 bytes
    points,discard = points_check(points)
    
    delay_points = np.int((acq_delay+board_delay)*SF)
    
    
    
    
    sdr.trigger_mode(5)
    sdr.PXItrigger()
    
    """
    if sdr.ADQAPI.ADQ_SetTransferTimeout(sdr._sdr,1,np.int(2000*ave) ) == 0:
        print('Error in setting the timings:')
        raise Exception('TIMERR')
    """
    if sdr.multirecord(ave,points,delay_points )==0: #bug on acquisition of the first wave
        print('Error in setting the multirecord')
        raise Exception('ACQSETUPERR')
    
    #setting the trigger for the DIO
    #sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,1,9)
    #sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,2,9)

    #sdr.awg_writesegment(np.ones(800),1,1)
    
    
    
    #sdr14 starts first
    sdr.trigarm()
    #sdr.awg_arm(3,1)
    
    #time.sleep(1)
    
    ##after starting the HVI, the DIO will trigger the SDR14 via PXIeTrigger number 0
    hvi.start()
    
    
    #waiting for data acquisition
    
    while(not sdr.acquisition_ended()):
            time.sleep(1)
        
    data = sdr.acquire_data(ave,points,3)
    
    
    
    sdr.trigarm('OFF')
    #sdr.awg_arm(3,0)
    
    return data[0][1:],data[1][1:]
    
    


    
def readout_SDR14(sdr,dio,hvi,ave,acq_delay,points,rep_period=1000,board_delay=800,SF=0.8,return_all=False):
    '''This function will play the sequences and read the data.
    The hvi and sequences must be previously loaded.
    
    if points and acq_delay are None, the defaults value from the readout pulse will be taken.
    
    This debug function will return all the acquired waves'''
    
    #bugfix
    ave = np.int(ave+1)
    
    def points_check(points):
        points2 = np.ceil(points/32)*32
        discard = points2-points
        
        return np.int(points2),np.int(discard)
    
    dio.writeRegisterByNumber(0,ave)
    dio.writeRegisterByNumber(3,np.int(rep_period*100)) #Rep time
    

        
    #the acquired data must be a multiple of 128 bytes
    points,discard = points_check(points)
    
    delay_points = np.int((acq_delay+board_delay)*SF)
    
    
    
    
    sdr.trigger_mode(5)
    sdr.PXItrigger()
    
    """
    if sdr.ADQAPI.ADQ_SetTransferTimeout(sdr._sdr,1,np.int(2000*ave) ) == 0:
        print('Error in setting the timings:')
        raise Exception('TIMERR')
    """
    if sdr.multirecord(ave,points,delay_points )==0: #bug on acquisition of the first wave
        print('Error in setting the multirecord')
        raise Exception('ACQSETUPERR')
    
    #setting the trigger for the DIO
    #sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,1,9)
    #sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,2,9)

    #sdr.awg_writesegment(np.ones(800),1,1)
    
    
    
    #sdr14 starts first
    sdr.trigarm()
    #sdr.awg_arm(3,1)
    
    #time.sleep(1)
    
    ##after starting the HVI, the DIO will trigger the SDR14 via PXIeTrigger number 0
    hvi.start()
    
    
    #waiting for data acquisition
    
    while(not sdr.acquisition_ended()):
            time.sleep(0.01)
        
    data = sdr.acquire_data(ave,points,3)
    
    
    
    sdr.trigarm('OFF')
    #sdr.awg_arm(3,0)
    
    sig_tot,ref_tot = data[0][1:],data[1][1:]
    sig_tot = np.sum(sig_tot,0)/len(sig_tot)
    ref_tot = np.sum(ref_tot,0)/len(ref_tot)
    del data
    
    
    p_length = points/0.8
    index = np.int(p_length*0.01)
    fsig = np.fft.fft(sig_tot)[index]/len(sig_tot)
    fref = np.fft.fft(ref_tot)[index]/len(ref_tot)
    
    
    if return_all is True:
        return np.abs(fsig)*2, np.angle(fsig)-np.angle(fref),sig_tot,ref_tot
    else:
        return np.abs(fsig)*2, np.angle(fsig)-np.angle(fref)
 

def dig_readout_debug(inst,hvi,ave,channel=0,ro_pulse_length=1000):
    '''readout_debug(inst,hvi,ave,channel=0,ro_pulse_length=1000):
    
    This function return the set of waves acquired by the digitizer (time domain)
    
    Args:
    - inst: instruments used, it is a tuple/list inst=(digitizer,sg_readout,sg_downmixer_LO,awg,sg_upmixer_LO) or (dig,sg_readout,sg_down_mixerLO,sg_excitation)
    - hvi: the HVI used for the measurement, both are fine (AWG or signal generator as excitation)
    - ave: averages number 
    - channel: digitizer channel used to acquire the signal (def 0)
    - ro_pulse_length: pulse length of the readout generator (def 1000 ns)
    
    NOTE: to convert digit in volt, multiply the value with Vpp / (2**14-1), where Vpp is the peak-to-peak voltage set on the digitizer (usually 3V)
    NOTE: the acquisition delay is set in the HVI 
    NOTE: the HVI must be loaded manually before using the function, this happens because this function is called in the measurements routines
    '''
    
    dig = inst[0]
    sgro = inst[1]    
    
    sgro.pulse_triggered(1,ro_pulse_length)
    sgro.output(1)
    points = np.int(np.round(ro_pulse_length*sigd_dig_SF,10))
    
    #hvi.load()
    dig.DAQconfig(channel,points,1,0,1) # channel,points,cycles,delay,SW trigger
    
    
    index = np.int(0.01*ro_pulse_length)
    
    if index-np.int(index)!=0:
        print('FF index is not an integer')
        raise Exception('FFindex')
    

    dig.writeRegisterByNumber(0,ave)

    hvi.start()

    a=[]
    for i in range(ave):    
        tmp,timeout = 0,0
        while(tmp==0):
            timeout+=1
            time.sleep(0.001)
            tmp= dig.DAQread(channel,points,10)
            if timeout> 10:
                raise Exception('TIMEOUT')
        
        dig.DAQstart(channel)
        dig.writeRegisterByNumber(2,1)
        a.append(np.array(tmp))
    

        
    return a
    

def dig_readout(inst,hvi,ave,channel=0,ro_pulse_length=1000):
    '''function readout_debug_with_scope(inst,hvi,ave,channel=0,ro_pulse_length=1000)::
    
    This function measures the readout pulse with the digitizer, it will return a tuple of three elements:
    
    - the transmitted amplitude in digits, after averaging.
    - future development
    - future development
    
    
    Args:
    - inst: instruments used, it is a tuple/list inst=(digitizer,sg_readout,sg_downmixer_LO,awg,sg_upmixer_LO) or (dig,sg_readout,sg_down_mixerLO,sg_excitation)
    - hvi: the HVI used for the measurement
    - ave: averages number (per point)
    - channel: digitizer channel used to acquire the signal (def 0)
    - ro_pulse_length: pulse length of the readout generator (def 1000 ns)
    
    
    NOTE: the acquisition delay is set in the HVI 
    NOTE: to convert digit in volt, multiply the value with Vpp / (2**14-1), where Vpp is the peak-to-peak voltage set on the digitizer (def 0.3V)
    NOTE: the HVI must be loaded manually before using the function, this happens because this function is called in the measurements routines
    '''
    
    dig = inst[0]
    sgro = inst[1]    
    
    sgro.pulse_triggered(1,ro_pulse_length,0)
    sgro.output(1)
    
    
    points = np.int(ro_pulse_length*sigd_dig_SF)
    dig.DAQconfig(channel,points,1,0,1) # channel,points,cycles,delay,SW trigger
    
    
    
    index = np.int(np.round(ro_pulse_length*0.01,12))
    
    if index-np.int(index)!=0:
        print('FF index is not an integer')
        raise Exception('FFindex')
    dig.writeRegisterByNumber(0,ave)
    
    hvi.start()
    
    y=np.zeros(points)
    for i in range(ave):    
        tmp,timeout = 0,0
        while(tmp==0):
            timeout+=1
            time.sleep(0.001)
            tmp= dig.DAQread(channel,points,10)
            if timeout> 10:
                raise Exception('TIMEOUT')
        
        dig.DAQstart(channel)
        dig.writeRegisterByNumber(2,1)
        y+= np.array(tmp)
    
    y/=ave
    
    fy = np.fft.fft(y)[index]/len(y)
        
    return np.abs(fy)*2,0,0 # The rest is used for future purposes, like phase measurements

def dig_readout_IQ(inst,hvi,ave,channel1=0,channel2=1,ro_pulse_length=1000):
    '''function readout_debug_with_scope(inst,hvi,ave,channel=0,ro_pulse_length=1000)::
    
    This function measures the readout pulse with the digitizer, it will return a tuple of three elements:
    
    - Amplitude
    - cos(phi)
    - sin(phi)
    
    
    Args:
    - inst: instruments used, it is a tuple/list inst=(digitizer,sg_readout,sg_downmixer_LO,awg,sg_upmixer_LO) or (dig,sg_readout,sg_down_mixerLO,sg_excitation)
    - hvi: the HVI used for the measurement (both are fine)
    - ave: averages number (per point)
    - channel: digitizer channel used to acquire the signal (def 0)
    - ro_pulse_length: pulse length of the readout generator (def 1000 ns)
    
    
    NOTE: the acquisition delay is set in the HVI 
    NOTE: to convert digit in volt, multiply the value with Vpp / (2**14-1), where Vpp is the peak-to-peak voltage set on the digitizer (usually 3V)
    NOTE: the HVI must be loaded manually before using the function, this happens because this function is called in the measurements routines
    '''
    
    dig = inst[0]
    sgro = inst[1]    
    
    sgro.pulse_triggered(1,ro_pulse_length,0)
    sgro.output(1)
    
    
    points = np.int(ro_pulse_length*sigd_dig_SF)
    dig.DAQconfig(channel1,points,1,0,1) # channel,points,cycles,delay,SW trigger
    dig.DAQconfig(channel2,points,1,0,1) # channel,points,cycles,delay,SW trigger
    
    
    
    index = np.int(np.round(ro_pulse_length*0.01,12))
    
    if index-np.int(index)!=0:
        print('FF index is not an integer')
        raise Exception('FFindex')
    dig.writeRegisterByNumber(0,ave)
    
    hvi.start()
    
    y1,y2 = np.zeros(points),np.zeros(points)
    for i in range(ave):    
        tmp1,tmp2,timeout = 0,0,0
        while(tmp1==0 or tmp2 ==0):
            timeout+=1
            time.sleep(0.001)
            tmp1 = dig.DAQread(channel1,points,10)
            if timeout> 10:
                raise Exception('TIMEOUT')
            tmp2 = dig.DAQread(channel2,points,10)
            if timeout> 10:
                raise Exception('TIMEOUT')
        
        dig.DAQstart(channel1)
        dig.DAQstart(channel2)
        dig.writeRegisterByNumber(2,1)
        y1+= np.array(tmp1)
        y2+= np.array(tmp2)
    
    y1/= ave
    y2/= ave
    
    fy1 = np.fft.fft(y1)[index]
    fy2 = np.fft.fft(y2)[index]
    
    
    return np.abs(fy1)*2/points,np.cos(np.angle(fy2)-np.angle(fy1)), np.sin(np.angle(fy2)-np.angle(fy1)) # The rest is used for future purposes, like phase measurements

def dig_readout_IQ_debug(inst,hvi,ave,channel1=0,channel2=1,ro_pulse_length=1000,delay=0):
    '''readout_debug(inst,hvi,ave,channel=0,ro_pulse_length=1000):
    
    This function return the set of waves acquired by the digitizer (time domain)
    
    Args:
    - inst: instruments used, it is a tuple/list inst=(digitizer,sg_readout,sg_downmixer_LO,awg,sg_upmixer_LO) or (dig,sg_readout,sg_down_mixerLO,sg_excitation)
    - hvi: the HVI used for the measurement
    - ave: averages number (per point)
    - channel: digitizer channel used to acquire the signal (def 0)
    - ro_pulse_length: pulse length of the readout generator (def 1000 ns)
    
    NOTE: to convert digit in volt, multiply the value with Vpp / (2**14-1), where Vpp is the peak-to-peak voltage set on the digitizer (3V)
    NOTE: the acquisition delay is set in the HVI 
    NOTE: the HVI must be loaded manually before using the function, this happens because this function is called in the measurements routines
    '''
    
    dig = inst[0]
    sgro = inst[1]    
    
    sgro.pulse_triggered(1,ro_pulse_length)
    sgro.output(1)
    points = np.int(np.round(ro_pulse_length*sigd_dig_SF,10))
    
    #hvi.load()
    dig.DAQconfig(channel1,points,1,delay,1) # channel,points,cycles,delay,SW trigger
    dig.DAQconfig(channel2,points,1,delay,1) # channel,points,cycles,delay,SW trigger
    
    
    index = np.int(0.01*ro_pulse_length)
    
    if index-np.int(index)!=0:
        print('FF index is not an integer')
        raise Exception('FFindex')
    

    dig.writeRegisterByNumber(0,ave)

    hvi.start()

    a=[]
    b=[]
    for i in range(ave):    
        tmp1,tmp2,timeout = 0,0,0
        while(tmp1==0 or tmp2 ==0):
            timeout+=1
            time.sleep(0.001)
            tmp1= dig.DAQread(channel1,points,10)
            if timeout> 10:
                print(i)
                raise Exception('TIMEOUT')
        
            tmp2= dig.DAQread(channel2,points,10)
            if timeout> 10:
                print(i)
                raise Exception('TIMEOUT')
        
        
        dig.DAQconfig(channel1,points,1,delay,1) # channel,points,cycles,delay,SW trigger
        dig.DAQconfig(channel2,points,1,delay,1) # channel,points,cycles,delay,SW trigger        
        dig.DAQstart(channel1)
        dig.DAQstart(channel2)
        
        dig.writeRegisterByNumber(2,1)
        a.append(np.array(tmp1))
        b.append(np.array(tmp2))
    

        
    return a,b

    

##------------------------------------------------------------------------------------------------------------ Utilities funcitons --------------------------------------------------


def load_calibration(awg,calibration,amplitude,awg_freq,ch0=0,ch1=1):
    '''   
    This function load a calibration in the awg and activates the amplitude modulation. it returns the phase_correction value on ch1 (this will be used later for T2 measurements)
    
    args:
    - awg: the card used as awg
    - calibration: this is a list, np.load must be used to load a calibration file, check the proper notebook for more details
    - amplitude: the amplitude you want to set, if the total amplitude is greater than 1.5 Vp you will get a warning    
    
    example: 
    
    cal100 = np.load('calibration-100MHz-6.2GHz.npz')
    load_calibration(awg,cal100,1.)
    
    NOTE: In this version only channels 0->I and 1->Q are used.
    '''    
    
    #Loading parameters
    phase_corr = calibration['phase_coeff']
    off0 = calibration['off0']
    off1 = calibration['off1']
    amp_chan= calibration['amp_chan']
    amp_corr = calibration['amp_coeff']
    amplitude_max = calibration['amplitude']
    #awg_freq = calibration['awg_freq']
    err=0
    
    #the formula is A*cos(w*t) + G *AWG(t)*cos(w*t)
    #so we want to put A = 0 and G = Amplitude
    
    # A=0
    err+= awg.channelAmplitude(ch0,0)
    err+= awg.channelAmplitude(ch1,0)
    
    #Setting channels offset
    err+= awg.channelOffset(ch0,off0)
    err+= awg.channelOffset(ch1,off1)
    
    #Setting cos(w*t)
    for chn in [ch0,ch1]:    
        err+= awg.channelWaveShape(chn,sigd.SD_Waveshapes.AOU_SINUSOIDAL)
        err+= awg.channelFrequency(chn,np.round(awg_freq*1e9,10))
        #err+= awg.modulationAmplitudeConfig(chn,0,0)
    
    #Setting phase difference (90 degrees between the two channels)
    err+= awg.channelPhase(ch1,90+phase_corr)
    err+= awg.channelPhase(ch0,0)
    #err+= awg.channelPhaseReset(ch0)
    #err+= awg.channelPhaseReset(ch1)
    
    #Setting G=amplitude
    if amplitude>amplitude_max:
        print('ERROR: Amplitude too large')
        raise Exception('AMPTOOHIGH')
    
    amp_set(awg,amplitude,amp_corr,amp_chan,ch0,ch1)
    
    if err!=0:
        print('error in calibration procedure')
        raise Exception('CALERR')
        
    return phase_corr,amp_corr,amp_chan



def amp_set(awg,amplitude,amp_corr,amp_chan,ch0=0,ch1=1):
    '''This function sets the amplitude of the first AWG channel, applying the correction to it.
    It is meant to be used with a calibrated IQ-mixer'''
    err=0
    if amp_chan==ch0:
        err+=awg.channelAmplitude(ch0,0)
        err+=awg.channelAmplitude(ch1,0)
        err+=awg.modulationAmplitudeConfig(ch0,sigd.SD_ModulationTypes.AOU_MOD_AM,amplitude*amp_corr)
        err+=awg.modulationAmplitudeConfig(ch1,sigd.SD_ModulationTypes.AOU_MOD_AM,amplitude)
    else:
        err+=awg.channelAmplitude(ch0,0)
        err+=awg.channelAmplitude(ch1,0)
        err+=awg.modulationAmplitudeConfig(ch0,sigd.SD_ModulationTypes.AOU_MOD_AM,amplitude)
        err+=awg.modulationAmplitudeConfig(ch1,sigd.SD_ModulationTypes.AOU_MOD_AM,amplitude*amp_corr)

    if err!=0:
        print('error in the amplitude set')
        raise Exception('CALERR')



def set_phase_in_reg_grad(angle,phase_corr):
    """conversion from grad to register value"""
    tot = (angle+phase_corr) % 360

    return np.int( (2**32-1)/360*tot )

def set_phase_in_reg_rad(angle,phase_corr):
    """conversion from rad to register value"""
    angle = np.rad2deg(angle)
    return set_phase_in_reg_grad(angle,phase_corr)


def create_flux_pulse(length,slope=20):
    """This function creates a flux pulse with a smooth raising time
    NOTE: the slopes are added to the length
    """
    n_points_slope = np.int(slope*sigd_awg_SF)
    n_points_flat = np.int(length*sigd_awg_SF)
    
    t_axis = np.linspace(0,1,n_points_slope)
    signal1 = np.abs(np.sin(t_axis*np.pi/2))
    signal2 = np.ones(n_points_flat)
    signal3 = 1-np.abs(np.sin(t_axis*np.pi/2))

    return np.hstack((signal1,signal2,signal3))

def create_double_flux_pulse(length,slope=20,amp1=1,amp2=None):
    """This function creates a double flux pulse with a smooth raising time,
    
    - amp1 is the amplitude of the pulse (def 1, normalized to +-1)
    - amp2 is the amplitude of the second half of the pulse (def is None that means -amp1)
    
    NOTE: the slopes are added to the length
    """
    n_points_slope = np.int(slope*sigd_awg_SF)
    n_points_flat = np.int(length/2*sigd_awg_SF)
    
    if amp1 <-1 or amp1 >1:
        print('Error: amp1 must be [-1,1]')
        raise Exception('FLAMP')
    
    if amp2 is None:
        amp2=-amp1
    
    if amp2 <-1 or amp2 >1:
        print('Error: amp2 must be [-1,1]')
        raise Exception('FLAMP')
    
    t_axis = np.linspace(0,1,n_points_slope)
    signal1 = np.abs(np.sin(t_axis*np.pi/2))
    signal2 = np.ones(n_points_flat)
    signal3 = 1-2*(np.abs(np.sin(t_axis*np.pi/2)))
    signal4 = amp2*np.ones(n_points_flat)
    signal5 = np.abs(np.sin(t_axis*np.pi/2))-1

    return np.hstack((signal1,signal2,signal3,signal4,signal5))

def awg_reset(awg):
    '''This function empties the AWG memory, sets all used registers to 0 and sets the amplitude of all channels to 0V'''
    awg.waveformFlush()
    for i in range(5):
        awg.writeRegisterByNumber(i,0)
    
    for i in range(4):
        awg.channelAmplitude(i,0)
    
            
##----------------------------------------------------------------------------------------- Pulses functions

def readout_pulse(inst, HVI,frequency, power, averages, pulse_length,delay=0,gap=0):
    """This function performs a readout at the given frequency, it returns the 
    average amplitude of the readout pulse.
    
    The delay is useful only in presence of excitation or flux pulses.
    """
    
    dig = inst[0]
    sgro = inst[1]
    sgLO = inst[2]
    dio = inst[5]

    if pulse_length < 0:
        print('Error: negative or 0 pulse_length')
        raise Exception('PLERR')
    
    if delay < 0:
        print('Error: negative delay is not possible')
        raise Exception('DELERR')
    
    #readout generator setup, it is supposed to be previously initialized
    sgro.pulse_triggered(1,pulse_length,0)
    sgro.frequency(frequency)
    sgro.power(power)
    
    if pulse_length == 0:
        sgro.output(0)
    else:
        sgro.output(1)
    
    #LO generator for down-mixing setup, it is supposed to be previously initialized
    sgLO.frequency(frequency+0.010)
    sgLO.output(1)
    
    #HVI.load()
    
    dio.writeRegisterByNumber(0,np.int(delay/10))
    dig.writeRegisterByNumber(3,np.int((delay+gap)/10))
    
    return dig_readout(inst,HVI,averages,sigd_dig_channel_used,pulse_length)[0]/(2**14-1)*sigd_dig_amp_used


def pi_pulse(inst,  calibration, amplitude, pulse_length, delay=0, frequency = None,shape = 'g',sigma= 0.,ch0=0,ch1=1,wait_precise=0):
    """This function prepare a pi-pulse in the AWG, ch0 and ch1 are used and the calibration file must be opened before 
    (calibration here is NOT the filename).
    
    The function will check if the LO is on, but in principle remember to test the calibration with the usual script before using it.
    
    NOTE:
    delay (def 0 ) is used to delay the pi-pulse, this is usually useful in combination with a flux pulse only
    sigma = 0. (def) will actually set it to pulse_length/6
    """
    
    #dig = inst[0]
    #sgro = inst[1]
    #sgLO = inst[4]
    awg = inst[3]
    sgEXLO = inst[4]
    #dio = inst[5]

    if pulse_length < 0:
        print('Error: negative pulse_length')
        raise Exception('PLERR')
    
    if delay < 0:
        print('Error: negative delay is not possible')
        raise Exception('DELERR')
    
    if sgEXLO.output() == 0:
        print('ERROR: the signal generator for Upmixing LO is off')
        raise Exception('LOOFF')

    phase_corr,amp_corr,amp_chan,awg_freq = load_calibration(awg,calibration,amplitude,ch0,ch1)

    if np.round(sgEXLO.frequency()+awg_freq,12) != frequency:
        print('Warning: pi-pulse frequency mismatch!: '+str(sgEXLO.frequency()+awg_freq/1e3))
    
    #it is supposed that the LO for up-mixing is ready because the calibration should be checked before using the up-mixing
    
    
    #HVI.load()
    #awg.waveformFlush()
    awg.AWGflush(ch0)
    awg.AWGflush(ch1)
    
    if pulse_length == 0.:
        return
        
    p1 = Pulse(Shape=shape,Sigma=sigma,Length=pulse_length,Delay=wait_precise,SF=sigd_awg_SF)
    wave1 = sigd.SD_Wave()
    wave1.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,p1.generate())
    awg.waveformLoad(wave1,1)
    
    awg.AWGqueueWaveform(ch0,1,0,0,1,0) #channel, waveform number, triggerMode, delay, cycles, prescaler
    awg.AWGqueueWaveform(ch1,1,0,0,1,0)
    
    awg.writeRegisterByNumber(2,np.int(pulse_length/10))
    awg.writeRegisterByNumber(0,np.int(delay/10))
    awg.writeRegisterByNumber(1,set_phase_in_reg_grad(90,phase_corr))
    awg.writeRegisterByNumber(3,0)
    awg.writeRegisterByNumber(4,set_phase_in_reg_grad(90,phase_corr))
    
    #return dig_readout(inst,HVI,sigd_dig_channel_used,pulse_length)[0]/(2**14-1)*sigd_dig_amp_used

def pi_pulse_n(inst,  calibration, amplitude, pulse_length, delay=0, frequency = None,shape = 'g',sigma= 0.,ch0=0,ch1=1,wait_precise=0,rep = 1):
    """This function prepare a pi-pulse in the AWG, ch0 and ch1 are used and the calibration file must be opened before 
    (calibration here is NOT the filename).
    
    The function will check if the LO is on, but in principle remember to test the calibration with the usual script before using it.
    
    NOTE:
    delay (def 0 ) is used to delay the pi-pulse, this is usually useful in combination with a flux pulse only
    sigma = 0. (def) will actually set it to pulse_length/6
    """
    
    #dig = inst[0]
    #sgro = inst[1]
    #sgLO = inst[4]
    awg = inst[3]
    sgEXLO = inst[4]
    #dio = inst[5]

    if pulse_length < 0:
        print('Error: negative pulse_length')
        raise Exception('PLERR')
    
    if delay < 0:
        print('Error: negative delay is not possible')
        raise Exception('DELERR')
    
    if sgEXLO.output() == 0:
        print('ERROR: the signal generator for Upmixing LO is off')
        raise Exception('LOOFF')

    phase_corr,amp_corr,amp_chan,awg_freq = load_calibration(awg,calibration,amplitude,ch0,ch1)

    if np.round(sgEXLO.frequency()+awg_freq/1e3,12) != frequency:
        print('Warning: pi-pulse frequency mismatch!: '+str(sgEXLO.frequency()+awg_freq/1e3))
    
    #it is supposed that the LO for up-mixing is ready because the calibration should be checked before using the up-mixing
    
    
    #HVI.load()
    #awg.waveformFlush()
    awg.AWGflush(ch0)
    awg.AWGflush(ch1)
    
    if pulse_length == 0.:
        return
        
    p1 = Pulse(Shape=shape,Sigma=sigma,Length=pulse_length,Delay=wait_precise,SF=sigd_awg_SF)
    wave1 = sigd.SD_Wave()
    wave1.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,p1.generate())
    awg.waveformLoad(wave1,1)
    
    awg.AWGqueueWaveform(ch0,1,1,0,rep,0) #channel, waveform number, triggerMode, delay, cycles, prescaler
    awg.AWGqueueWaveform(ch1,1,1,0,rep,0)
    
    awg.writeRegisterByNumber(2,np.int(pulse_length*rep/10))
    awg.writeRegisterByNumber(0,np.int(delay/10))
    awg.writeRegisterByNumber(1,set_phase_in_reg_grad(90,phase_corr))
    awg.writeRegisterByNumber(3,0)
    awg.writeRegisterByNumber(4,set_phase_in_reg_grad(90,phase_corr))
    
    
def flux_pulse(inst,flux_amplitude=0,flux_function=None,flux_delay=0,channel=3):
    """This function loads flux_function (must be an array of numbers, normalized to [-1,1]) in the AWG ch2
    the pulse can be delayed (def 0), in order to better use the flux_pulse in combination with a pi-pulse and ro-pulse.
    flux_amplitude is the AWG channel amplitude [-1.5,1.5] V
    """
    awg = inst[3]

    if flux_amplitude < -1.5 or flux_amplitude >1.5:
        print('Error: flux amplitude must be [-1.5,1.5] V' )
        raise Exception('FLUXAMPERR')
    
    if flux_delay < 0:
        print('Error: negative delay is not possible')
        raise Exception('DELERR')

    awg.channelAmplitude(channel,flux_amplitude)
    awg.channelOffset(channel,0)
    awg.channelWaveShape(channel,sigd.SD_Waveshapes.AOU_AWG)
    awg.modulationAmplitudeConfig(channel,0,0)
    
    awg.AWGflush(channel)
    

    if flux_function is None:
        return
    else:
        
        
        wave1 = sigd.SD_Wave()
        wave1.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,flux_function)
        
        pause1 = sigd.SD_Wave()
        pause1.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,np.zeros(np.int(20*sigd_awg_SF)))
        
        pause2 = sigd.SD_Wave()
        pause2.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,np.zeros(np.int(30*sigd_awg_SF)))
        
        
        awg.waveformLoad(pause1,2)
        awg.waveformLoad(pause2,3)
        awg.waveformLoad(wave1,4)
        
        if flux_delay == 0:
            awg.AWGqueueWaveform(channel,4,1,0,1,0) #channel, waveform number, triggerMode, delay, cycles, prescaler
        elif flux_delay % 20 == 0:
            awg.AWGqueueWaveform(channel,2,1,0,np.int(flux_delay/20),0)
            awg.AWGqueueWaveform(channel,4,0,0,1,0)
        elif flux_delay % 30 == 0:
            awg.AWGqueueWaveform(channel,3,1,0,np.int(flux_delay/30),0)
            awg.AWGqueueWaveform(channel,4,0,0,1,0)
        else:
            awg.AWGqueueWaveform(channel,3,1,0,1,0)
            awg.AWGqueueWaveform(channel,2,0,0,np.int((flux_delay-30)/20),0)
            awg.AWGqueueWaveform(channel,4,0,0,1,0)
        
        
def T2_pulse(inst,  calibration, amplitude, pulse_length, delay=0, frequency=None,shape = 'g',sigma= 0.,pulses_delay=0,second_angle=0.,ch0=0,ch1=1):
    '''
    
    '''
    #dig = inst[0]
    #sgro = inst[1]
    #sgLO = inst[4]
    awg = inst[3]
    sgEXLO = inst[4]
    #dio = inst[5]

    if pulse_length < 0:
        print('Error: negative pulse_length')
        raise Exception('PLERR')
    
    if delay < 0:
        print('Error: negative delay is not possible')
        raise Exception('DELERR')
    
    if pulses_delay < 0:
        print('Error: negative delay is not possible')
        raise Exception('PDELERR')
    
    if sgEXLO.output() == 0:
        print('ERROR: the signal generator for Upmixing LO is off')
        raise Exception('LOOFF')

    
    
    if pulses_delay < 0:
        print('Error: pulses_delay should be > pulse_length, it is the gap between the halfpi-pulses')
        raise Exception('PDERR')
 
    if pulses_delay == 10 or (pulses_delay % 10 )!= 0:
        print('Error: pulses_delay should be a multiple of 10 and cannot be 10 ns')
        raise Exception('PDERR')
        
    phase_corr,amp_corr,amp_chan,awg_freq = load_calibration(awg,calibration,amplitude,ch0=ch0,ch1=ch1) 

    if np.round(sgEXLO.frequency()+awg_freq,12) != frequency:
        print('Warning: pi-pulse frequency mismatch!: '+str(sgEXLO.frequency()+awg_freq))
    
 
        
    #HVI.load()
    #awg.waveformFlush()
    awg.AWGflush(ch0)
    awg.AWGflush(ch1)
    
    if pulse_length == 0.:
        return
    
    p1 = Pulse(Shape=shape,Sigma=sigma,Length=pulse_length,SF=sigd_awg_SF)
    wave1 = sigd.SD_Wave()
    wave1.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,p1.generate())
    
    pause1 = sigd.SD_Wave()
    pause1.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,np.zeros(np.int(20*sigd_awg_SF)))
    
    pause2 = sigd.SD_Wave()
    pause2.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,np.zeros(np.int(30*sigd_awg_SF)))
    
    awg.waveformLoad(wave1,1)
    awg.waveformLoad(pause1,2)
    awg.waveformLoad(pause2,3)
    
    
    #timings
    err=0
    err+=awg.writeRegisterByNumber(0,np.int((pulse_length+10)/10))
    #err+=awg.writeRegisterByNumber(0,np.int(delay/10))
    if err <0:
        print('ERROR: not possible to write in register: '+str(err))
        raise Exception('REGERR')
    #test
    #awg.writeRegisterByNumber(1,set_phase_in_reg_grad(90,0))
    #awg.writeRegisterByNumber(3,set_phase_in_reg_grad(0,0))
    #awg.writeRegisterByNumber(4,set_phase_in_reg_grad(90,0))
    
    #angles
    err=0
    err+=awg.writeRegisterByNumber(1,set_phase_in_reg_grad(90.,phase_corr))
    err+=awg.writeRegisterByNumber(3,set_phase_in_reg_grad(second_angle,0))
    err+=awg.writeRegisterByNumber(4,set_phase_in_reg_grad(90.+second_angle,phase_corr))
    if err <0:
        print('ERROR: not possible to write in register: '+str(err))
        raise Exception('REGERR')
        
    #delay between the two hpulses
    if pulses_delay == 0:
        for i in [ch0,ch1]:
            awg.AWGqueueWaveform(i,1,0,0,1,0) #channel, waveform number, triggerMode, delay, cycles, prescaler
            awg.AWGqueueWaveform(i,1,0,0,1,0)
    elif pulses_delay % 20 == 0:
        for i in [ch0,ch1]:
            awg.AWGqueueWaveform(i,1,0,0,1,0)
            awg.AWGqueueWaveform(i,2,0,0,np.int(pulses_delay/20),0)
            awg.AWGqueueWaveform(i,1,0,0,1,0)
    elif pulses_delay % 30 == 0:
        for i in [ch0,ch1]:
            awg.AWGqueueWaveform(i,1,0,0,1,0)
            awg.AWGqueueWaveform(i,3,0,0,np.int(pulses_delay/30),0)
            awg.AWGqueueWaveform(i,1,0,0,1,0)
    else:
        for i in [ch0,ch1]:
            awg.AWGqueueWaveform(i,1,0,0,1,0)
            awg.AWGqueueWaveform(i,3,0,0,1,0)
            awg.AWGqueueWaveform(i,2,0,0,np.int((pulses_delay-30)/20),0)
            awg.AWGqueueWaveform(i,1,0,0,1,0)  

            
            
def TEcho_pulse(inst,  calibration, pi_amplitude, hpi_amplitude_ratio, pulse_length, delay=0, frequency=None,shape = 'g',sigma= 0.,pulses_delay=0,second_angle=0.):
    
    #dig = inst[0]
    #sgro = inst[1]
    #sgLO = inst[4]
    awg = inst[3]
    sgEXLO = inst[4]
    #dio = inst[5]

    if pulse_length < 0:
        print('Error: negative pulse_length')
        raise Exception('PLERR')
    
    if delay < 0:
        print('Error: negative delay is not possible')
        raise Exception('DELERR')
    
    if pulses_delay < 0:
        print('Error: negative delay is not possible')
        raise Exception('PDELERR')
    
    if sgEXLO.output() == 0:
        print('ERROR: the signal generator for Upmixing LO is off')
        raise Exception('LOOFF')


    

    #it is supposed that the LO for up-mixing is ready because the calibration should be checked before using the up-mixing
    
    pulses_delay_2 = pulses_delay-pulse_length
    if pulses_delay_2 < 0:
        print('Error: pulses_delay should be > pulse_length, it is the gap between the halfpi-pulses')
        raise Exception('PDERR')
    
    pulses_delay_2/=2
    if pulses_delay_2 == 10 or (pulses_delay_2 % 10 )!= 0:
        print('Error: pulses_delay should be a multiple of 10 and >pulse_length+20 ns')
        raise Exception('PDERR')
    
        
    phase_corr,amp_corr,amp_chan,awg_freq = load_calibration(awg,calibration,pi_amplitude)
    
    if np.round(sgEXLO.frequency()+awg_freq,12) != frequency:
        print('Warning: pi-pulse frequency mismatch!: '+str(sgEXLO.frequency()+awg_freq))

    
    #HVI.load()
    #awg.waveformFlush()
    awg.AWGflush(0)
    awg.AWGflush(1)
    
    if pulse_length == 0.:
        return
        
    p1 = Pulse(Shape=shape,Sigma=sigma,Length=pulse_length,SF=sigd_awg_SF)
    wave1 = sigd.SD_Wave()
    wave1.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,p1.generate())

    wave2 = sigd.SD_Wave()
    wave2.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,p1.generate()*hpi_amplitude_ratio)

    
    pause1 = sigd.SD_Wave()
    pause1.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,np.zeros(np.int(20*sigd_awg_SF)))
    
    pause2 = sigd.SD_Wave()
    pause2.newFromArrayDouble(sigd.SD_WaveformTypes.WAVE_ANALOG,np.zeros(np.int(30*sigd_awg_SF)))
    
    awg.waveformLoad(wave2,1)
    awg.waveformLoad(pause1,2)
    awg.waveformLoad(pause2,3)
    awg.waveformLoad(wave1,5)
        
        #awg.AWGqueueWaveform(0,1,1,0,1,0) #channel, waveform number, triggerMode, delay, cycles, prescaler
        #awg.AWGqueueWaveform(1,1,1,0,1,0)
    
    #timings
    awg.writeRegisterByNumber(2,np.int( (pulse_length*2+pulses_delay/2+10 )/10))
    awg.writeRegisterByNumber(0,np.int(delay/10))
    
    #angles
    awg.writeRegisterByNumber(1,set_phase_in_reg_grad(90.,phase_corr))
    awg.writeRegisterByNumber(3,set_phase_in_reg_grad(second_angle,0))
    awg.writeRegisterByNumber(4,set_phase_in_reg_grad(90.+second_angle,phase_corr))
    
    
    
    
    #delay between the two hpulses
    if pulses_delay_2 == 0:
        for i in range(2):
            awg.AWGqueueWaveform(i,1,0,0,1,0) #channel, waveform number, triggerMode, delay, cycles, prescaler
            awg.AWGqueueWaveform(i,5,0,0,1,0)            
            awg.AWGqueueWaveform(i,1,0,0,1,0)
    elif pulses_delay_2 % 20 == 0:
        for i in range(2):
            awg.AWGqueueWaveform(i,1,0,0,1,0)
            awg.AWGqueueWaveform(i,2,0,0,np.int(pulses_delay_2/20),0)
            awg.AWGqueueWaveform(i,5,0,0,1,0)
            awg.AWGqueueWaveform(i,2,0,0,np.int(pulses_delay_2/20),0)
            awg.AWGqueueWaveform(i,1,0,0,1,0)
    elif pulses_delay_2 % 30 == 0:
        for i in range(2):
            awg.AWGqueueWaveform(i,1,0,0,1,0)
            awg.AWGqueueWaveform(i,3,0,0,np.int(pulses_delay_2/30),0)
            awg.AWGqueueWaveform(i,5,0,0,1,0)
            awg.AWGqueueWaveform(i,3,0,0,np.int(pulses_delay_2/30),0)
            awg.AWGqueueWaveform(i,1,0,0,1,0)
    else:
        for i in range(2):
            awg.AWGqueueWaveform(i,1,0,0,1,0)
            awg.AWGqueueWaveform(i,3,0,0,1,0)
            awg.AWGqueueWaveform(i,2,0,0,np.int((pulses_delay_2-30)/20),0)
            awg.AWGqueueWaveform(i,5,0,0,1,0)  
            awg.AWGqueueWaveform(i,3,0,0,1,0)
            awg.AWGqueueWaveform(i,2,0,0,np.int((pulses_delay_2-30)/20),0)
            awg.AWGqueueWaveform(i,1,0,0,1,0)  


##----------------------------------------------------------------------------------------- Measurements functions ------------------  
def peak_scan(inst,HVI,freq_sweep,ave1,ave2,power,ro_pulse_length,ro_delay=0,read_temperature=True,plot=True,return_all = False):
    '''This function is used to perform a readout mode scan, it means that the frequency of the readout pulse is sweeped.
    
    the ro_delay can be used if a scan in presence of an excitation pulse wants to be done. This also means that the user must perform
    an awg_reset if only the readout pulse must be used.
    '''
    HVI.load()
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    tmp = np.ndarray(len(freq_sweep))
    y= []
    for count in range(ave1):
        
        for i,f in enumerate(freq_sweep):
            tmp[i] = readout_pulse(inst,HVI,f,power, ave2,ro_pulse_length,delay=ro_delay)
            
            if plot is True:
                progressive_plot_2d(freq_sweep[:i],tmp[:i],'-o')
            
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_power = power, ro_pulse_length = ro_pulse_length)
    
    data.x = freq_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data
        
        
def Two_tone_scan(inst,HVI,freq_sweep,ave1,ave2,ro_pulse,ex_pulse,ex_delay=0,ro_delay=None,read_temperature=True,plot=True,return_all = False):
    '''This function performs a two tone scan with a "saturation pulse". This means that two signal generators are used in pulsed mode.
    It is possibile to use the same inst of the other functions IF the sgEXLO is used as the excitation generator.
    
    ro_pulse/ex_pulse must be a tuple/list: (ro_frequency, ro_power, ro_pulse_length)
    
    
    since the ex_pulse frequency will be swept, the ex_frequency parameter is ignored.
    
    If ro_delay is None, it will be automatically set equal to ex_pulse_length so that the two pulses are subsequent.
    If ro_delay is 0 or less than ex_pulse_length, the pulses will be overlapped!
    ex_delay can be used to delay the ex_pulse, usually useful in presence of a flux_pulse.
    '''
    HVI.load()
    
    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    ex_frequency,ex_power,ex_pulse_length = ex_pulse
    
    
    #readout pulse will be prepare during the readout:
    
    #excitation pulse setup:
    sgex = inst[4]
    
    #sgex.frequency(ex_frequency) # the frequency will be sweeped
    sgex.power(ex_power)
    sgex.pulse_triggered(1,ex_pulse_length,0)
    sgex.ALC("OFF")
    sgex.output(1)
    
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    #timing setup
    if ro_delay is None:
        ro_delay = ex_pulse_length
    
    dio = inst[5]
    
    #dio.writeRegisterByNumber(0,np.int(ex_delay/10))
    dio.writeRegisterByNumber(1,np.int(ro_delay/10))


    tmp = np.ndarray(len(freq_sweep))
    y= []
    for count in range(ave1):
        
        for i,f in enumerate(freq_sweep):
            sgex.frequency(f)
            time.sleep(0.01)
            tmp[i] = readout_pulse(inst,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ex_delay,ro_delay)
            
            if plot is True:
                progressive_plot_2d(freq_sweep[:i],tmp[:i])
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, ex_pulse = ex_pulse, ro_delay = ro_delay, ex_delay=ex_delay)
    
    data.x = freq_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data
        
def Rabi(inst,HVI,cal,amp_sweep,ave1,ave2,ro_pulse,ex_pulse,ex_delay=0,ro_delay= 0,read_temperature=True,plot=True,return_all = False,ch0=0,ch1=1):
    '''This function is used to tune up a pi-pulse, so it should be a sequence of pi-pulse -> readout.
    
    the ro_pulse parameter must be: (frequency,power/amplitude,length)
    the ex_pulse parameter must be: (frequency,amplitude,length,[shape],[sigma])
    
    since the excitation frequency should be set through the calibration code, it will be ignored (but checked)
    since the excitation amplitude (awg) is swept, the ex_power will be ignored.
    
    
    ro_delay is used to increase the pause between the pi pulse and the readout pulse, the user is responsible to set ex_delay and ro_delay
    accurately in order to have the right timings (check wich a high sampling scope).
    
    '''
    
    HVI.load()

    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    if len(ex_pulse) >= 3:
        ex_frequency,ex_power,ex_pulse_length = ex_pulse[0],ex_pulse[1],ex_pulse[2]
        shape,sigma = 'g',0.
    else:
        print('ERROR: ex_pulse must have at least 3 values')
        raise Exception('EXPULERR')
        
    if len(ex_pulse) == 4:
        shape = ex_pulse[3]
    
    elif len(ex_pulse) == 5:
        shape = ex_pulse[3]
    elif len(ex_pulse)>5:
        print('ERROR: ex_pulse must have maximum 5 values')
        raise Exception('EXPULERR')
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    #timing setup
    
    

    tmp = np.ndarray(len(amp_sweep))
    y= []
    for count in range(ave1):
        
        for i,ampl in enumerate(amp_sweep):
            pi_pulse(inst,cal,ampl,ex_pulse_length, ex_delay,ex_frequency,shape,sigma,ch0=ch0,ch1=ch1)
            time.sleep(0.01)
            tmp[i] = readout_pulse(inst,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ro_delay+ex_pulse_length)
            
            if plot is True:
                progressive_plot_2d(amp_sweep[:i],tmp[:i],'-o')
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, ex_pulse = ex_pulse, ro_delay = ro_delay, ex_delay=ex_delay)
    
    data.x = amp_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data
        
def Rabi_time(inst,HVI,cal,time_sweep,ave1,ave2,ro_pulse,ex_pulse,ex_delay=0,ro_delay= 0,read_temperature=True,plot=True,return_all = False):
    '''This function is used to tune up a pi-pulse, so it should be a sequence of pi-pulse -> readout.
    
    the ro_pulse parameter must be: (frequency,power/amplitude,length)
    the ex_pulse parameter must be: (frequency,amplitude,length,[shape],[sigma])
    
    since the excitation frequency should be set through the calibration code, it will be ignored (but checked)
    since the excitation amplitude (awg) is swept, the ex_power will be ignored.
    
    
    ro_delay is used to increase the pause between the pi pulse and the readout pulse, the user is responsible to set ex_delay and ro_delay
    accurately in order to have the right timings (check wich a high sampling scope).
    
    '''
    
    HVI.load()
    
    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    if len(ex_pulse) >= 3:
        ex_frequency,ex_power,ex_pulse_length = ex_pulse[0],ex_pulse[1],ex_pulse[2]
        shape,sigma = 'g',0.
    else:
        print('ERROR: ex_pulse must have at least 3 values')
        raise Exception('EXPULERR')
        
    if len(ex_pulse) == 4:
        shape = ex_pulse[3]
    
    elif len(ex_pulse) == 5:
        shape = ex_pulse[3]
    elif len(ex_pulse)>5:
        print('ERROR: ex_pulse must have maximum 5 values')
        raise Exception('EXPULERR')
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    #timing setup
    
    

    tmp = np.ndarray(len(time_sweep))
    y= []
    for count in range(ave1):
        
        for i,plen in enumerate(time_sweep):
            pi_pulse(inst,cal,ex_power,plen, ex_delay,ex_frequency,shape,sigma)
            time.sleep(0.01)
            tmp[i] = readout_pulse(inst,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ro_delay+plen)
            
            if plot is True:
                progressive_plot_2d(time_sweep[:i],tmp[:i],'-o')
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, ex_pulse = ex_pulse, ro_delay = ro_delay, ex_delay=ex_delay)
    
    data.x = time_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data
        
def T1_measurement(inst,HVI,cal,delay_sweep,ave1,ave2,ro_pulse,ex_pulse,ex_delay=0,ro_delay= 0,read_temperature=True,plot=True,return_all = False):
    '''This function is used to tune up a pi-pulse, so it should be a sequence of pi-pulse -> readout.
    
    the ro_pulse parameter must be: (frequency,power/amplitude,length)
    the ex_pulse parameter must be: (frequency,amplitude,length,[shape],[sigma])
    
    since the excitation frequency should be set through the calibration code, it will be ignored (but checked!)
    
    
    ro_delay is used to increase the pause between the pi pulse and the readout pulse, the user is responsible to set ex_delay and ro_delay
    accurately in order to have the right timings (check wich a high sampling scope).
    
    '''
    
    HVI.load()

    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    if len(ex_pulse) >= 3:
        ex_frequency,ex_power,ex_pulse_length = ex_pulse[0],ex_pulse[1],ex_pulse[2]
        shape,sigma = 'g',0.
    else:
        print('ERROR: ex_pulse must have at least 3 values')
        raise Exception('EXPULERR')
        
    if len(ex_pulse) == 4:
        shape = ex_pulse[3]
    
    elif len(ex_pulse) == 5:
        shape = ex_pulse[3]
    elif len(ex_pulse)>5:
        print('ERROR: ex_pulse must have maximum 5 values')
        raise Exception('EXPULERR')
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    #pi-pulse setup
    pi_pulse(inst,cal,ex_power,ex_pulse_length, ex_delay,ex_frequency,shape,sigma)
    

    tmp = np.ndarray(len(delay_sweep))
    y= []
    for count in range(ave1):
        
        for i,de in enumerate(delay_sweep):
            
            time.sleep(0.01)
            tmp[i] = readout_pulse(inst,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ro_delay+ex_pulse_length+de)
            
            if plot is True:
                progressive_plot_2d(delay_sweep[:i],tmp[:i],'-o')
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, ex_pulse = ex_pulse, ro_delay = ro_delay, ex_delay=ex_delay)
    
    data.x = delay_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data
        

def T2_measurement(inst,HVI,cal,delay_sweep,ave1,ave2,ro_pulse,ex_pulse,ex_delay=0,ro_delay= 0,omega=0,read_temperature=True,plot=True,return_all = False):
    '''This function is used to tune up a pi-pulse, so it should be a sequence of pi-pulse -> readout.
    
    the ro_pulse parameter must be: (frequency,power/amplitude,length)
    the ex_pulse parameter must be: (frequency,amplitude,length,[shape],[sigma])
    
    omega is the second half-pi-pulse angle variation, pratically the angle of the second half-pi-pulse is changed of an amount equal to omega * pulses_delay
    
    since the excitation frequency should be set through the calibration code, it will be ignored (but checked!)
    
    
    ro_delay is used to increase the pause between the pi pulse and the readout pulse, the user is responsible to set ex_delay and ro_delay
    accurately in order to have the right timings (check wich a high sampling scope).
    
    '''
    
    HVI.load()

    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    if len(ex_pulse) >= 3:
        ex_frequency,ex_power,ex_pulse_length = ex_pulse[0],ex_pulse[1],ex_pulse[2]
        shape,sigma = 'g',0.
    else:
        print('ERROR: ex_pulse must have at least 3 values')
        raise Exception('EXPULERR')
        
    if len(ex_pulse) == 4:
        shape = ex_pulse[3]
    
    elif len(ex_pulse) == 5:
        shape = ex_pulse[3]
    elif len(ex_pulse)>5:
        print('ERROR: ex_pulse must have maximum 5 values')
        raise Exception('EXPULERR')
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    

    tmp = np.ndarray(len(delay_sweep))
    y= []
    for count in range(ave1):
        
        for i,de in enumerate(delay_sweep):
            #2*hpi-pulse setup
            T2_pulse(inst,cal,ex_power,ex_pulse_length, ex_delay,ex_frequency,shape,sigma,de,omega*de)
    
            
            time.sleep(0.01)
            tmp[i] = readout_pulse(inst,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ro_delay+2*ex_pulse_length+de)
            
            if plot is True:
                progressive_plot_2d(delay_sweep[:i],tmp[:i],'-o')
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, ex_pulse = ex_pulse, ro_delay = ro_delay, ex_delay=ex_delay,omega=omega)
    
    data.x = delay_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data

def T2_measurement_temporary(inst1,inst2,HVI,cal1,cal2,delay_sweep,ave1,ave2,ro_pulse,pi_pulse01,pi_pulse12,ex_delay=0,ro_delay= 0,read_temperature=True,plot=True,return_all = False):
    '''This function is used to tune up a pi12-pulse, so it should be a sequence of pi01-hpi12-delay-hpi12-pi01 -> readout.
    
    the ro_pulse parameter must be: (frequency,power/amplitude,length)
    the ex_pulse parameter must be: (frequency,amplitude,length,[shape],[sigma])
    
    omega is the second half-pi-pulse angle variation, pratically the angle of the second half-pi-pulse is changed of an amount equal to omega * pulses_delay
    
    since the excitation frequency should be set through the calibration code, it will be ignored (but checked!)
    
    
    ro_delay is used to increase the pause between the pi pulse and the readout pulse, the user is responsible to set ex_delay and ro_delay
    accurately in order to have the right timings (check wich a high sampling scope).
    
    '''
    
    HVI.load()

    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    if len(pi_pulse12) >= 3:
        pi12_frequency,pi12_power,pi12_pulse_length = pi_pulse12[0],pi_pulse12[1],pi_pulse12[2]
        shape12,sigma12 = 'g',0.
    else:
        print('ERROR: pi_pulse12 must have at least 3 values')
        raise Exception('EXPULERR')
    
    if len(pi_pulse01) >= 3:
        pi01_frequency,pi01_power,pi01_pulse_length = pi_pulse01[0],pi_pulse01[1],pi_pulse01[2]
        shape01,sigma01 = 'g',0.
    else:
        print('ERROR: pi_pulse01 must have at least 3 values')
        raise Exception('EXPULERR')
    
    
    if len(pi_pulse12) == 4:
        shape12 = pi_pulse12[3]
    
    elif len(pi_pulse12) == 5:
        shape12 = pi_pulse12[3]
    elif len(pi_pulse12)>5:
        print('ERROR: pi_pulse12 must have maximum 5 values')
        raise Exception('EXPULERR')
    
    if len(pi_pulse01) == 4:
        shape01 = pi_pulse01[3]
    
    elif len(pi_pulse01) == 5:
        shape01 = pi_pulse01[3]
    elif len(pi_pulse01)>5:
        print('ERROR: pi_pulse01 must have maximum 5 values')
        raise Exception('EXPULERR')
    
        
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    

    tmp = np.ndarray(len(delay_sweep))
    y= []
    for count in range(ave1):
        
        for i,de in enumerate(delay_sweep):
            T2_pulse(inst2,cal2,pi01_power,pi01_pulse_length,0,pi01_frequency,shape01,sigma01,2*pi12_pulse_length+de+50,0,2,3)
            T2_pulse(inst1,cal1,pi12_power,pi12_pulse_length, ex_delay,pi12_frequency,shape12,sigma12,de,0)
    
            
            time.sleep(0.01)
            tmp[i] = readout_pulse(inst1,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ro_delay+2*pi12_pulse_length+de)
            
            if plot is True:
                progressive_plot_2d(delay_sweep[:i],tmp[:i],'-o')
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, pi_pulse01 = pi_pulse01,pi_pulse12=pi_pulse12, ro_delay = ro_delay, ex_delay=ex_delay,omega=0)
    
    data.x = delay_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data

        
def TEcho_measurement(inst,HVI,cal,delay_sweep,ave1,ave2,ro_pulse,ex_pulse,hpipulse,ex_delay=0,ro_delay= 0,omega=0,read_temperature=True,plot=True,return_all = False):
    '''This function is used to tune up a pi-pulse, so it should be a sequence of pi-pulse -> readout.
    
    the ro_pulse parameter must be: (frequency,power/amplitude,length)
    the ex_pulse parameter must be: (frequency,amplitude,length,[shape],[sigma])
    
    hampr is the ratio between the half-pi-pulse and the pi-pulse (def 0.5)
    
    omega is the second half-pi-pulse angle variation, pratically the angle of the second half-pi-pulse is changed of an amount equal to omega * pulses_delay
    
    since the excitation frequency should be set through the calibration code, it will be ignored (but checked!)
    
    
    ro_delay is used to increase the pause between the pi pulse and the readout pulse, the user is responsible to set ex_delay and ro_delay
    accurately in order to have the right timings (check wich a high sampling scope).
    
    '''
    
    HVI.load()

    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    if len(ex_pulse) >= 3:
        ex_frequency,ex_power,ex_pulse_length = ex_pulse[0],ex_pulse[1],ex_pulse[2]
        shape,sigma = 'g',0.
    else:
        print('ERROR: ex_pulse must have at least 3 values')
        raise Exception('EXPULERR')
        
    if len(ex_pulse) == 4:
        shape = ex_pulse[3]
    
    elif len(ex_pulse) == 5:
        shape = ex_pulse[3]
    elif len(ex_pulse)>5:
        print('ERROR: ex_pulse must have maximum 5 values')
        raise Exception('EXPULERR')
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    ratio = hpipulse[1]/ex_pulse[1]

    tmp = np.ndarray(len(delay_sweep))
    y= []
    for count in range(ave1):
        
        for i,de in enumerate(delay_sweep):
            #2*hpi-pulse setup
            TEcho_pulse(inst,cal,ex_power,ratio,ex_pulse_length, ex_delay,ex_frequency,shape,sigma,de,omega*de)
    
            
            time.sleep(0.01)
            tmp[i] = readout_pulse(inst,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ro_delay+2*ex_pulse_length+de)
            
            if plot is True:
                progressive_plot_2d(delay_sweep[:i],tmp[:i],'-o')
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, ex_pulse = ex_pulse, ro_delay = ro_delay, ex_delay=ex_delay,omega=omega,hampr=ratio)
    
    data.x = delay_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data
        
        
def Cavity_temperature(inst,HVI,freq_sweep,ave1,ave2,ro_pulse,ex_pulse,cav_ex_pulse,ro_delay=0,ex_delay=0,return_all = False,plot=False,read_temperature=True):
    '''This function performs a two tone scan with a "saturation pulse" and also gives the possibility to put photons in the cavity before performing a readout.
    This means that three signal generators are used in pulsed mode.

    inst must be a tuple with the following initialized generators:
    (dig,sgro,sgLO,sgex,sg_cav_ex,dio)
    
    ro_pulse,ex_pulse and cav_ex_pulse must be a tuple/list: (frequency, power, pulse_length)
    
    
    since the ex_pulse frequency will be swept, the ex_frequency parameter is ignored.
    
    If ro_delay is None, it will be automatically set equal to ex_pulse_length so that the two pulses are subsequent.
    If ro_delay is 0 or less than ex_pulse_length, the pulses will be overlapped!
    ex_delay can be used to delay the ex_pulse-ro_pulse sequence.
    '''
    
    HVI.load()
    
    #dig = inst[0]
    #sgro = inst[1]
    #sgLO = inst[2]
    sgex = inst[3]
    sgcex = inst[4]
    dio = inst[5]
    
    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    ex_frequency,ex_power,ex_pulse_length = ex_pulse
    cav_ex_frequency,cav_ex_power,cav_ex_pulse_length,cav_ex_on = cav_ex_pulse
    
    
    #readout pulse will be prepare during the readout:
    
    
    
    #sgex.frequency(ex_frequency) # the frequency will be sweeped
    """
    sgex.power(ex_power)
    sgex.pulse_triggered(1,ex_pulse_length,0)
    sgex.ALC("OFF")
    sgex.output(1)
    """
    sgcex.frequency(cav_ex_frequency)
    sgcex.power(cav_ex_power)
    sgcex.pulse_triggered(1,cav_ex_pulse_length,0)
    sgcex.ALC("OFF")
    if cav_ex_on is True:
        sgcex.output(1)
    else:
        sgcex.output(0)
    
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    
    
    
    
    dio.writeRegisterByNumber(1,np.int(ex_delay/10))
    dio.writeRegisterByNumber(2,np.int(ro_delay/10))


    tmp = np.ndarray(len(freq_sweep))
    y= []
    for count in range(ave1):
        
        for i,f in enumerate(freq_sweep):
            sgex.frequency(f)
            time.sleep(0.1)
            tmp[i] = readout_pulse(inst,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ex_pulse_length,ex_delay+ro_delay)
            
            if plot is True:
                progressive_plot_2d(freq_sweep[:i],tmp[:i],'-o')
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, ex_pulse = ex_pulse, cav_ex_pulse = cav_ex_pulse,ro_delay = ro_delay, ex_delay=ex_delay)
    
    data.x = freq_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data
    

def pi_amp_tuning(inst,HVI,cal,amp_sweep,ave,ro_pulse, ex_pulse,ex_delay=0,ro_delay= 0, pi_rep_high=10, pi_rep_low=11,  read_temperature=True,plot=True, ch0=0,ch1=1):
    '''This function is used to tune up a pi-pulse, so it should be a sequence of pi-pulse -> readout.
    
    the ro_pulse parameter must be: (frequency,power/amplitude,length)
    the ex_pulse parameter must be: (frequency,amplitude,length,[shape],[sigma])
    
    since the excitation frequency should be set through the calibration code, it will be ignored (but checked)
    since the excitation amplitude (awg) is swept, the ex_power will be ignored.
    
    
    ro_delay is used to increase the pause between the pi pulse and the readout pulse, the user is responsible to set ex_delay and ro_delay
    accurately in order to have the right timings (check wich a high sampling scope).
    
    '''
    
    HVI.load()

    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    if len(ex_pulse) >= 3:
        ex_frequency,ex_power,ex_pulse_length = ex_pulse[0],ex_pulse[1],ex_pulse[2]
        shape,sigma = 'g',0.
    else:
        print('ERROR: ex_pulse must have at least 3 values')
        raise Exception('EXPULERR')
        
    if len(ex_pulse) == 4:
        shape = ex_pulse[3]
    
    elif len(ex_pulse) == 5:
        shape = ex_pulse[3]
    elif len(ex_pulse)>5:
        print('ERROR: ex_pulse must have maximum 5 values')
        raise Exception('EXPULERR')
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()   

    y = np.ndarray(len(amp_sweep))    
    
    for i,ampl in enumerate(amp_sweep):
        
            pi_pulse_n(inst,cal,ampl,ex_pulse_length, ex_delay,ex_frequency,shape,sigma,ch0=ch0,ch1=ch1,rep=pi_rep_high)
            time.sleep(0.01)
            tmp_high = readout_pulse(inst,HVI,ro_frequency,ro_power, ave,ro_pulse_length,ro_delay+ex_pulse_length*pi_rep_high)
            awg_reset(inst[3])
            pi_pulse_n(inst,cal,ampl,ex_pulse_length, ex_delay,ex_frequency,shape,sigma,ch0=ch0,ch1=ch1,rep=pi_rep_low)
            time.sleep(0.01)
            tmp_low = readout_pulse(inst,HVI,ro_frequency,ro_power, ave,ro_pulse_length,ro_delay+ex_pulse_length*pi_rep_low)
            y[i] = tmp_high-tmp_low
            
        
            if plot is True:
                progressive_plot_2d(amp_sweep[:i],y[:i],'-bo')
    
    if read_temperature is True:
        
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()
    
    data.load_var(amp_sweep,y)
    data.insert_par(averages = ave, ro_pulse = ro_pulse, ex_pulse = ex_pulse, ro_delay = ro_delay, ex_delay=ex_delay,pi_rep_high = pi_rep_high, pi_rep_low = pi_rep_low)
    data.time_stop = time.time()   
    
    return data



def flux_tune(inst,HVI,cal,time_sweep,ave1,ave2,ro_pulse,ex_pulse,flux_amp,flux_function,ex_delay=0,ro_delay= 0,flux_delay=0,read_temperature=True,plot=True,return_all = False,ch0=0,ch1=1,flux_ch=3):
    '''This function is used to tune up a pi-pulse, so it should be a sequence of pi-pulse -> readout.
    
    the ro_pulse parameter must be: (frequency,power/amplitude,length)
    the ex_pulse parameter must be: (frequency,amplitude,length,[shape],[sigma])
    
    since the excitation frequency should be set through the calibration code, it will be ignored (but checked)
    since the excitation amplitude (awg) is swept, the ex_power will be ignored.
    
    
    ro_delay is used to increase the pause between the pi pulse and the readout pulse, the user is responsible to set ex_delay and ro_delay
    accurately in order to have the right timings (check wich a high sampling scope).
    
    '''
    
    HVI.load()

    ro_frequency,ro_power,ro_pulse_length = ro_pulse
    if len(ex_pulse) >= 3:
        ex_frequency,ex_power,ex_pulse_length = ex_pulse[0],ex_pulse[1],ex_pulse[2]
        shape,sigma = 'g',0.
    else:
        print('ERROR: ex_pulse must have at least 3 values')
        raise Exception('EXPULERR')
        
    if len(ex_pulse) == 4:
        shape = ex_pulse[3]
    
    elif len(ex_pulse) == 5:
        shape = ex_pulse[3]
    elif len(ex_pulse)>5:
        print('ERROR: ex_pulse must have maximum 5 values')
        raise Exception('EXPULERR')
    
    
    data = dm.data_table()
    
    if read_temperature is True:
        sr = SensorReader()
        sr.update()
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
    
    data.time_start = time.time()
    
    #timing setup
    
    flux_pulse(inst,flux_amp,flux_function,flux_delay,flux_ch)
    tmp = np.ndarray(len(time_sweep))
    y= []
    for count in range(ave1):
        
        for i,delay in enumerate(time_sweep):
            pi_pulse(inst,cal,ex_power,ex_pulse_length, ex_delay+delay,ex_frequency,shape,sigma,ch0=ch0,ch1=ch1)
            time.sleep(0.01)
            tmp[i] = readout_pulse(inst,HVI,ro_frequency,ro_power, ave2,ro_pulse_length,ro_delay+ex_pulse_length+ex_delay+delay)
            
            if plot is True:
                progressive_plot_2d(time_sweep[:i],tmp[:i],'-o')
        
        y.append(tmp.copy())
    
    if read_temperature is True:
        sr.update()
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()

    data.time_stop = time.time()
    
    data.insert_par(averages = str(ave1)+' x '+str(ave2), ro_pulse = ro_pulse, ex_pulse = ex_pulse, ro_delay = ro_delay, ex_delay=ex_delay,flux_amp = flux_amp)
    
    data.x = time_sweep
    data.y = np.sum(y,0)/ave1
    data.select()
    
    if return_all is True:
        return data,y
    else:
        return data
    
#--------------------------------------------------------------------------------------------- Experiment functions -------------------------------------------------------

