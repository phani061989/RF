# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:49:20 2015

@author: oscar


Library created to perform two tone scans with the SDR14
"""

print('SDR14TT v1.0.0')

import numpy as np
#from UtilitiesLib import value_between
import time
from PULSE import *
from SIGGEN import *
from SDR14 import *
from progress_bar import InitBar
from UtilitiesLib import time_difference,progressive_plot_2d
from AWG import *

def initialize_sg(IP):
    
    sg=Siggen(IP)
    #sg.frequency_mode('CW')
    sg.frequency_reference('EXT')
    
    sg.ALC('OFF')
    
    return sg    

def initialize_LO(IP):
    
    sg=Siggen(IP)
    
    sg.frequency_reference('EXT')

    sg.ALC('ON')
    
    return sg    

def initialize_sdr(gain):
    
    sd=SDR14()

    #External Chassis Clock
    sd.clock_ref()

    #AWG
    #setting AWG

    #SW trigger
    sd.awg_trigger(3,'INT')
    

    #Normal trigger mode
    sd.awg_trigin(3,0)
    
    
    sd.awg_enablesegments(3,1)
    
    sd.GainOffset(1,gain)
    
    sd.awg_continuous(3,0)
    sd.awg_autorearm(3,1)
    
    sd.awg_arm(3,1)

    
    
    
    sd.trigger_mode('INT',6400) #n/200 gives the period in us
    

    return sd

def initialize_awg():
    
    awg=Awg(path='/mnt/TekAWG1/')

    #External Chassis Clock
    awg.output_state(1,1)
    awg.output_state(1,3)
    awg.output_state(0,2)

    

    return awg


def SDR14TT(inst,ex_par,ro_par,acquisition_delay=0,pulses_delay=0,LO_power=9,averages=100,dm_freq=16e-3,var_delay=0,ex_pulse_off=False,return_all=False,progress_bar=True):
    '''function used to perform a two tone with pulses using the SDR14 as a trigger for the siggen and as digitizer
    
    Setup:
    - sgLO will be used as a Local Oscillator for the down mixing
    - sgro is the readout generator, fixed frequency, and needs to be down-mixed
    - sgex is used for excitation, his frequency will be varied
    - sdr output will be used to trigger the sig. gen., they need amplification in order to work
        - the input 1 of SDR14 will acquire the signal after down-mixing
        
    Parameters:
    - every instrument must be initialized with the proper function before passing it to this function, the inst argument is a tuple/list with in order:
        (sgLO, sgro, sgex, sdr)
    - ex_par is a tuple/list with the excitation parameters: (array of frequency points,power,pulse_length)
    - ro_par is a tuple/list with the readout parameters: (array of frequency points, power, pulse_length)
    - acquisition_delay is used to center the acquisition on the readout pulse (calibration) (0 def)
    - pulses_delay is used to change the timing between excitation pulse and readout pulse, it can be negative (0 def)
    - LO_power and dm_freq will set the power for the down mixer and the down mixed frequency respect. (9 dbm and 16 MHz def)
    - averages is the number of averages made on the board (100 def)
    - var_delay can be used to change the timing between the end of the excitation pulse and the begin of the readout, the acquisition delay will follow also (0 def)
    - ex_pulse_off: This parameter will turn off the power of the excitation signal generator (can be used to make a direct scan, without two tone) (False def)
    - return_all: This parameter will return the acquired wave (the average is on-board) after the amplitude (False def)
    - progress_bar: This parameter can be used to display the progress bar (True def)
    '''    

    if len(inst) != 4:
        print('ERROR: inst tuple/list doesn\'t contain 4 instruments')
        raise Exception('INSTNUM')
    
    sgLO,sgro,sgex,sdr = inst[0],inst[1],inst[2],inst[3]
    #sgro.frequency(round(ro_par[0],9))
    sgro.power(round(ro_par[1],9))
    sgro.pulse_trigged(ro_par[2],0,'ON')
    sgro.output(1)    
    
    if ex_pulse_off == True:
        if sgex!=None:
            sgex.output(0)
    else:
        sgex.power(ex_par[1])
        #sgex.frequency(round(ex_freq,9))
        sgex.pulse_trigged(ex_par[2],0,'ON')
        sgex.output(1)

            
    
    sgLO.power(LO_power)
        
    
    
    #LOfreq= round(ro_par[0]+dm_freq,9)
    #sgLO.frequency(LOfreq)
    sgLO.output(1)    
    
    """
    #cal LO power
    d=load('12-ft cable.npz')
    xcal,ycal = d['x'],d['y']
    sgLO.power(9+value_between(xcal,ycal,LOfreq))
    """
    """
    #cal power combiner
    d=load('power-combiner-5-7.npz')
    x1cal,y1cal = d['x'],d['raw1']
    x2cal,y2cal = d['x'],d['raw2']
    sgro.power(ro_power+value_between(x1cal,y1cal,ro_freq))
    sgex.power(ex_power+value_between(x2cal,y2cal,ex_freq))
    """
    
    #sgLO.output(1)
    
    #sc=initialize_scope()
    
    
    #pulse samples must be a multiple of 16
    tot_length= ex_par[2]+ro_par[2] +var_delay
    if tot_length%10 != 0:
        print("total pulse time is not a multiple of ten")
        raise Exception('PulseLength')        
    
    dtmp = ex_par[2]+pulses_delay +var_delay
    if dtmp%10 !=0:
        print("total readout delay time is not a multiple of ten")
        raise Exception('ACQDelay')
    
    p1=Pulse(Delay=dtmp,Length=160,Wait=tot_length-160-dtmp,SF=sdr.DACSF)
    
    """
    if ex_pulse_on == True:
        p2=Pulse(Length=160,Wait=p1.gettotallength('t')-160,SF=sdr.DACSF)    
    elif ex_pulse_on == False:
        p2=Pulse(Wait=p1.gettotallength('t'),SF=sdr.DACSF)
    else:
        print('Wrong ex_pulse_on state inserted')
    """
    
    p2=Pulse(Length=160,Wait=p1.gettotallength('t')-160,SF=sdr.DACSF)    
    #return p1,p2
    
    sdr.awg_continuous(3,0)
    sdr.awg_autorearm(3,1)
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,1,9)
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,2,9)

    #sdr.awg_arm(3,0)
    #sdr.trigarm(0)
    
    sdr.awg_writesegment(p1.generate()*0.4,1,1)
    sdr.awg_writesegment(p2.generate()*0.4,2,1)

    samples = int(ro_par[2]*sdr.ADCSF)
    sdr.averaging(averages,samples,3,Hold=ex_par[2]+acquisition_delay +var_delay)
    
    sdr.awg_arm(3,1)
    time.sleep(1)
    
    #sgex.output(1)
    if return_all==True:
        collection=[]
    
    data= np.ndarray((len(ex_par[0]),len(ro_par[0])))
    
    if len(ro_par[0])==1 & len(ex_par[0])==1:
        progress_bar=False
    
    if progress_bar == True:
        expoints,ropoints = len(ex_par[0]),len(ro_par[0])
        pb=InitBar('Measuring:',expoints*ropoints)
        pb(0)
        cycle_start=time.time()
    
    for i,exf in enumerate(ex_par[0]):    
        
        
        sgex.frequency(exf)
        time.sleep(0.001)
        
        for j,rof in enumerate(ro_par[0]):
            step_start = time.time()
            
            sgLO.frequency(round(rof+dm_freq,9))
            sgro.frequency(round(rof,9))
            time.sleep(0.001)
            sdr.averaging_arm(1)
        
            ave_stat,retry = [0,0,0],0
            while ave_stat[0]==0:
                #time.sleep(1)
                ave_stat=sdr.averaging_status()
                #print("acquired: "+str(ave_stat[1]))
                if ave_stat[1]==averages:
                    retry+=1
                    if retry>5:
                        break
                
                
            y = sdr.averages_acquire(samples)/averages
            
            if return_all==True:
                collection.append(y)
            
            ftmp = abs(np.fft.fft(y))/len(y)
            #amplitude=ftmp[int(round(ro_length*dm_freq,10))]
            
            data[i][j] =  2*ftmp[int(round(ro_par[2]*dm_freq,10))]
            
            if progress_bar==True:
                pb(ropoints*i+j-1,' Extimated: '+time_difference(step_start,ropoints*expoints-ropoints*i-j-1)+'               ' )
    
    if progress_bar==True:
        pb(ropoints*expoints, 'Time: '+time_difference(cycle_start)+'                                      ')

    #sdr.awg_arm(3,0)
    sdr.averaging_arm(0)
    
    sgro.output(0)
    sgLO.output(0)
    sgex.output(0)

    if return_all == True:
        return data,collection
    else:
        return data        


def SDR14TTdec(inst,ex_par,ro_par,acquisition_delay=0,pulses_delay=0,LO_power=9,averages=100,dm_freq=16e-3,var_delay=0,return_all=False,progress_bar=True):
    '''function used to perform a two tone with pulses using the SDR14 as a trigger for the siggen and as digitizer
    
    Setup:
    - sgLO will be used as a Local Oscillator for the down mixing
    - sgro is the readout generator, fixed frequency, and needs to be down-mixed
    - sgex is used for excitation, his frequency will be varied
    - SDR14 output will be used to trigger the sig. gen., they need amplification in order to work
        - the input 1 of SDR14 will acquire the signal after down-mixing
        
    Parameters:
    - every instrument must be initialized with the proper function before passing it to this function
    - ex_par is a tuple/list with the excitation parameters: (array of frequency points,power,pulse_length)
    - ro_par is a tuple/list with the readout parameters: (array of frequency points, power, pulse_length)
    
    '''    
    if len(inst) != 4:
        print('ERROR: inst tuple/list doesn\'t contain 4 instruments')
        raise Exception('INSTNUM')
    
    sgLO,sgro,sgex,sdr = inst[0],inst[1],inst[2],inst[3]
    #sgro.frequency(round(ro_par[0],9))

    
    
    if averages<1:
        print('Wrong averages number inserted')
        raise Exception('AveNum')
    
    #sdr,sgro,sgex,sgLO = inst[0], inst[1], inst[2], inst[3]
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,1,9)
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,2,9)
    sdr.awg_arm(3,1)
    
    sgro.power(round(ro_par[1],9))
    sgro.pulse_trigged(ro_par[2],0,'ON')
    sgro.frequency(round(ro_par[0][0],12))
    

    sgex.power(ex_par[1])
    sgex.frequency(round(ex_par[0][0],12))
    sgex.pulse_trigged(ex_par[2],0,'ON')
    
        

    
    sgLO.power(LO_power)
    sgLO.frequency(np.round(ro_par[0][0]+dm_freq,10))
    sgLO.output(1)
    
    
    time.sleep(1)
    sgro.output(1)    
    sgex.output(1)

    data=np.ndarray(len(var_delay))
    
    if progress_bar == True:
        points = len(var_delay)
        pb=InitBar('Measuring:',points)
        pb(0)
        cycle_start=time.time()
    
    collection=[]
    
    for i,vd in enumerate(var_delay):
        step_start = time.time()
    
        #pulse samples must be a multiple of 16
        tot_length= ex_par[2]+vd+ro_par[2]
        if tot_length % 10!=0:
            print('total number of samples  is not a multiple of 16')
            raise Exception('SamplesNum')
        
        
        dtmp = ex_par[2]+pulses_delay+vd
        if dtmp%10 !=0:
            print('Pulse delay is not a multiple of 10')
            raise Exception('ACQDelay')
        
        p1=Pulse(Delay=dtmp,Length=200,Wait=tot_length-200-dtmp,SF=sdr.DACSF)
            
        p2=Pulse(Length=200,Wait=p1.gettotallength('t')-200,SF=sdr.DACSF)    
            
        #return p1,p2
        
        
    
        
        #sdr.trigarm(0)
        
        sdr.awg_writesegment(p1.generate()*0.4,1,1)
        sdr.awg_writesegment(p2.generate()*0.4,2,1)
        
        #sdr.multirecord(averages,int((ro_length+ex_length)*sdr.ADCSF))
        samples= np.int(ro_par[2]*sdr.ADCSF)
        hold=ex_par[2]+acquisition_delay+vd
        if hold%10 != 0:
            print('Acquisition delay is not a multiple of 10')
            raise Exception('ACQDelay')
            
        sdr.averaging(averages,samples,3,Hold=hold)
    
        
        
        sdr.averaging_arm(1)
    
        ave_stat,retry = [0,0,0],0
        while ave_stat[0]==0:
            #time.sleep(1)
            ave_stat=sdr.averaging_status()
            #print("acquired: "+str(ave_stat[1]))
            if ave_stat[1]==averages:
                retry+=1
                if retry>5:
                    break
            
            
        y = sdr.averages_acquire(samples)/averages
        
        if return_all==True:
            collection.append(y)
        
        ftmp = abs(np.fft.fft(y))/len(y)
        #amplitude=ftmp[int(round(ro_length*dm_freq,10))]
        
        data[i] =  2*ftmp[int(round(ro_par[2]*dm_freq,10))]
        
        if progress_bar==True:
            pb(i+1,' Extimated: '+time_difference(step_start,points-i-1)+'               ' )

    if progress_bar==True:
        pb(points, 'Time: '+time_difference(cycle_start)+'                                      ')
    
    sdr.averaging_arm(0)
    sgro.output(0)
    sgex.output(0)
    sgLO.output(0)
    
    if return_all == True:
        return data,collection
    else:
        return data


def SDR14TTram(inst, ex_par, ro_par, averages, fixed_delay_exex, fixed_delay_exro, fixed_delay_acquisition, LO_pwr=9, dm_freq=16e-3, var_delay = (0,0), return_all=False,progress_bar=True):
    """
    function parameters:

    arg1: tuple/list with initialized instruments (sdr14,readout generator, excitation generator, Local Oscillator generator)
    arg2: Readout parameters tuple/list: readout power (dbm), readout frequency (GHz), readout pulse length (ns)
    arg3: Excitation parameters tuple/list: excitation power (dbm), frequency (GHz), pulse length (ns) and state on (True/False)
    arg4: Number of averages (integer >0)   
    arg5: delay between excitation pulses, can be negative (ns)
    arg6: delay between 2nd excitation pulse and readout pulse, can be negative (ns)
    arg7: delay of the acquisition, it is a fix needed because of the propagation trough the system    
    arg7: Local oscillator generator output power (dbm, def is 9)
    arg8: down mixed frequency (MHz, def is 16). Must be a multiple of 16
    arg9: a tuple/list of 2 elements. It is a variable delay between excitation pulses and between 2nd excitation pulse and readout pulse, used for TRabi or T1 measurements (ns, def is (0,0) )
    arg10: if it is True, the function will return (amplitude, acquired signals), if it is False (def) only the amplitude
    """
    
    if len(inst)!=4:
        print('Not enough instruments')
        raise Exception('InstNum')
    
    if averages<1:
        print('Wrong averages number inserted')
        raise Exception('AveNum')
    
    sgLO,sgro,sgex,sdr = inst[0],inst[1],inst[2],inst[3]
    
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,1,9)
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,2,9)

    sdr.awg_arm(3,1)    
    
    sgro.power(round(ro_par[1],9))
    sgro.frequency(round(ro_par[0][0],9))
    sgro.pulse_trigged(ro_par[2],0,'ON')
    sgro.output(1)
    
    
    sgex.power(round(ex_par[1],9))
    sgex.frequency(round(ex_par[0][0],9))
    sgex.pulse_trigged(ex_par[2],0,'ON')
    sgex.output(1)
    
    sgLO.power(LO_pwr)
    
    LOfreq= round(ro_par[0][0]+dm_freq,LO_pwr)
    sgLO.frequency(LOfreq)
    sgLO.output(1)
    
    """
    #cal LO power
    d=load('12-ft cable.npz')
    xcal,ycal = d['x'],d['y']
    sgLO.power(9+value_between(xcal,ycal,LOfreq))
    """
    """
    #cal power combiner
    d=load('power-combiner-5-7.npz')
    x1cal,y1cal = d['x'],d['raw1']
    x2cal,y2cal = d['x'],d['raw2']
    sgro.power(ro_power+value_between(x1cal,y1cal,ro_freq))
    sgex.power(ex_power+value_between(x2cal,y2cal,ex_freq))
    """
    
    
    
    data=np.ndarray(( len(var_delay[0]), len(var_delay[1]) ))
    if return_all == True:
        collection = []
        
    if progress_bar == True:
        pointsex,pointsro = len(var_delay[0]),len(var_delay[1])
        pb=InitBar('Measuring:',pointsex*pointsro)
        pb(0)
        cycle_start=time.time()
    
    for i,exd in enumerate(var_delay[0]):
    #pulse samples must be a multiple of 16
        for j,rod in enumerate(var_delay[1]):
            step_start= time.time()            
            
            tot_length= 2*ex_par[2]+exd+rod+fixed_delay_exex+fixed_delay_exro+ro_par[2]+10e3
    
    
            dtmp = 2*ex_par[2]+fixed_delay_exex+fixed_delay_exro+exd+rod
    
            p1=Pulse(Delay=dtmp,Length=100,Wait=tot_length-100-dtmp,SF=sdr.DACSF)
        
    
            p2=Pulse(Length=200,Wait=ex_par[2]+fixed_delay_exex+exd-200,SF=sdr.DACSF)    
            p2b=Pulse(Length=200,Wait=tot_length-200-p2.gettotallength(),SF=sdr.DACSF)
            p2.insert(p2b)
        
            
            #return p1,p2
    
    
            sdr.awg_writesegment(p1.generate()*0.4,1,1)
            sdr.awg_writesegment(p2.generate()*0.4,2,1)

            samples= np.int(ro_par[2]*sdr.ADCSF)
            hold=dtmp+fixed_delay_acquisition
            if hold%10 != 0:
                print('Acquisition delay is not a multiple of 10')
                raise Exception('ACQDelay')
            
            sdr.averaging(averages,samples,3,Hold=hold)
    
        
        
            sdr.averaging_arm(1)
    
            ave_stat,retry = [0,0,0],0
            while ave_stat[0]==0:
                #time.sleep(1)
                ave_stat=sdr.averaging_status()
                #print("acquired: "+str(ave_stat[1]))
                if ave_stat[1]==averages:
                    retry+=1
                    if retry>5:
                        break
            
            
            y = sdr.averages_acquire(samples)/averages
        
            if return_all==True:
                collection.append(y)
        
            ftmp = abs(np.fft.fft(y))/len(y)
        
        
            data[i][j] =  2*ftmp[int(round(ro_par[2]*dm_freq,10))]
        
            if progress_bar==True:
                pb(pointsro*i+j+1,' Extimated: '+time_difference(step_start,pointsro+pointsex-pointsro*i-j-1)+'               ' )

    if progress_bar==True:
        pb(pointsro*pointsex, 'Time: '+time_difference(cycle_start)+'                                      ')
    
    
    
    
    sdr.averaging_arm(0)
    
    
    sgLO.output(0)
    sgro.output(0)
    sgex.output(0)

    
    
    if return_all == True:
        return data,collection
    else:
        return data
        

def SDR14TTram2(inst,ex_par,ro_par,acquisition_delay=440,ro_pulse_delay=1200,LO_power=9,averages=100,dm_freq=16e-3,var_delay=[0,],deltaphi=0,return_all=False,progress_bar=True,create_pulses=True,plotting=False):
    
    
    if ex_par[1] == 'g' or ex_par[1]=='G':
        pulse_shape='g'
    else:
        pulse_shape='s'
    
    #INSTRUMENT INIT
    if len(inst) != 4:
        print('ERROR: inst tuple/list doesn\'t contain 4 instruments')
        raise Exception('INSTNUM')
    
    sgLO,sgro,awg,sdr = inst[0],inst[1],inst[2],inst[3]
    #sgro.frequency(round(ro_par[0],9))

    
    
    if averages<1:
        print('Wrong averages number inserted')
        raise Exception('AveNum')
    
    #sdr,sgro,sgex,sgLO = inst[0], inst[1], inst[2], inst[3]
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,1,9)
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,2,9)
    sdr.awg_arm(3,1)
    
    sgro.power(round(ro_par[1],9))
    sgro.pulse_trigged(ro_par[2],0,'ON')
    sgro.frequency(round(ro_par[0][0],12))
    sgro.output(1)    


    sgLO.power(LO_power)
    sgLO.frequency(np.round(ro_par[0][0]+dm_freq,10))
    
    
    points = len(var_delay)

    if create_pulses==True:
        if progress_bar == True:
            points = len(var_delay)
            pb=InitBar('Creating pulses:',points)
            pb(0)
            cycle_start=time.time()
        
        #AWG pulses preparation
        for i,vd in enumerate(var_delay):
            step_start=time.time()
            
            exf= np.round(ex_par[0][0],10)
            p1=Pulse(exf,Length=ex_par[2],Wait=vd,Shape=pulse_shape)
            # normal code            
            #f2= 2*np.pi*exf*p1.gettotallength()+2*np.pi*deltaphi*vd
            #f2-=2*pi*np.int(f2/2*pi)
            p2=Pulse(exf,2.*np.pi*deltaphi*vd,Length=ex_par[2],Shape=pulse_shape)
            p1.insert(p2)
            
            
            """
            # debug code            
            f2= 2*np.pi*np.random.rand()
            #f2-=2*pi*int(f2/2*pi)
            p2=Pulse(exf,f2,Length=ex_par[2])
            pf=p2.generate()
            """
            
            
            
            #m1=np.zeros(len(pf))
            #m1[:2500]=1
            
            awg.create_file('pulse'+str(i),p1.generate())
            
            if progress_bar==True:
                pb(i+1,' Extimated: '+time_difference(step_start,points-i-1)+'               ' )

        if progress_bar==True:
            pb(points, 'Time: '+time_difference(cycle_start)+'                                      ')
    
    sgLO.output(1)    
    #SDR14 Pulses preparation
    data=np.ndarray(len(var_delay))
    if return_all==True:
        collection=[]
    
    if progress_bar == True:
        
        pb=InitBar('Measuring:',points)
        pb(0)
        cycle_start=time.time()
        
    for i,vd in enumerate(var_delay):
        step_start= time.time()
        
        awg.run(0)
        awg.load_signal('pulse'+str(i),10)
        awg.run(1)
                
        dtmp = 2*ex_par[2]+ro_pulse_delay+vd
        if dtmp%10 !=0:
            print('Pulse delay is not a multiple of 10')
            raise Exception('ACQDelay')
        
         #pulse samples must be a multiple of 16
        tot_length= dtmp+ro_par[2]+10e3
        if tot_length % 10!=0:
            print('total number of samples  is not a multiple of 16')
            raise Exception('SamplesNum')
        
        
        
        p1=Pulse(Delay=dtmp,Length=200,Wait=tot_length-200-dtmp,SF=sdr.DACSF)
        p2=Pulse(Length=200,Wait=p1.gettotallength('t')-200,SF=sdr.DACSF)
        
        sdr.awg_writesegment(p1.generate()*0.4,1,1)
        sdr.awg_writesegment(p2.generate()*0.4,2,1)
        

        samples= np.int(ro_par[2]*sdr.ADCSF)
        hold=dtmp+acquisition_delay
        if hold%10 != 0:
            print('Acquisition delay is not a multiple of 10')
            raise Exception('ACQDelay')
            
        sdr.averaging(averages,samples,3,Hold=hold)
    
        
        time.sleep(1)
        sdr.averaging_arm(1)
    
        ave_stat,retry = [0,0,0],0
        while ave_stat[0]==0:
            #time.sleep(1)
            ave_stat=sdr.averaging_status()
            #print("acquired: "+str(ave_stat[1]))
            if ave_stat[1]==averages:
                retry+=1
                if retry>5:
                    break
            
            
        y = sdr.averages_acquire(samples)/averages
        
        if return_all==True:
            collection.append(y)
        
        ftmp = abs(np.fft.fft(y))/len(y)
        #amplitude=ftmp[int(round(ro_length*dm_freq,10))]
        
        data[i] =  2*ftmp[int(round(ro_par[2]*dm_freq,10))]
        
        if plotting == True:
            progressive_plot_2d(var_delay[i],data[i],'-bo',clear=False)
        
        if progress_bar==True:
            pb(i+1,' Extimated: '+time_difference(step_start,points-i-1)+'               ' )

    if progress_bar==True:
        pb(points, 'Time: '+time_difference(cycle_start)+'                                      ')
    
    sdr.averaging_arm(0)
    sdr.awg_arm(3,0)
    sgro.output(0)
    awg.run(0)
    sgLO.output(0)
    
    if return_all == True:
        return data,collection
    else:
        return data


def SDR14TTawg(inst,ex_par,ro_par,acquisition_delay=440,ro_pulse_delay=1210,LO_power=9,averages=100,dm_freq=16e-3,var_delay=[0,],return_all=False,progress_bar=True):
    
    
    
    
    #INSTRUMENT INIT
    if len(inst) != 4:
        print('ERROR: inst tuple/list doesn\'t contain 4 instruments')
        raise Exception('INSTNUM')
    
    sgLO,sgro,awg,sdr = inst[0],inst[1],inst[2],inst[3]
    #sgro.frequency(round(ro_par[0],9))

    
    
    if averages<1:
        print('Wrong averages number inserted')
        raise Exception('AveNum')
    
    #sdr,sgro,sgex,sgLO = inst[0], inst[1], inst[2], inst[3]
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,1,9)
    sdr.ADQAPI.SDR14_AWGSetTriggerEnable(sdr._sdr,1,2,9)
    sdr.awg_arm(3,1)
    
    sgro.power(round(ro_par[1],9))
    sgro.pulse_trigged(ro_par[2],0,'ON')
    sgro.frequency(round(ro_par[0][0],12))
    sgro.output(1)    


    sgLO.power(LO_power)
    sgLO.frequency(np.round(ro_par[0][0]+dm_freq,10))
    sgLO.output(1)
    
    points = len(var_delay)

    p1=Pulse(round(ex_par[0][0],10),Length=ex_par[2])
    m1=Pulse(Length=100,Wait=ex_par[2]-100)
    awg.create_file('ex_pulse',p1.generate(),m1.generate())
            
            
    #SDR14 Pulses preparation
    data=np.ndarray(len(var_delay))
    if return_all==True:
        collection=[]
    
    if progress_bar == True:
        
        pb=InitBar('Measuring:',points)
        pb(0)
        cycle_start=time.time()
    
    awg.run(0)
    awg.load_signal('ex_pulse',9)
    awg.run(1)
    time.sleep(2)
    
    for i,vd in enumerate(var_delay):
        step_start= time.time()
        
        
        dtmp = ex_par[2]+ro_pulse_delay+vd
        if dtmp%10 !=0:
            print('Pulse delay is not a multiple of 10')
            raise Exception('ACQDelay')
        
         #pulse samples must be a multiple of 16
        tot_length= dtmp+ro_par[2]+10e3
        if tot_length % 10!=0:
            print('total number of samples  is not a multiple of 16')
            raise Exception('SamplesNum')
        
        
        
        p1=Pulse(Delay=dtmp,Length=200,Wait=tot_length-200-dtmp,SF=sdr.DACSF)
        p2=Pulse(Length=200,Wait=p1.gettotallength('t')-200,SF=sdr.DACSF)
        
        sdr.awg_writesegment(p1.generate()*0.4,1,1)
        sdr.awg_writesegment(p2.generate()*0.4,2,1)
        

        samples= np.int(ro_par[2]*sdr.ADCSF)
        hold=dtmp+acquisition_delay
        if hold%10 != 0:
            print('Acquisition delay is not a multiple of 10')
            raise Exception('ACQDelay')
            
        sdr.averaging(averages,samples,3,Hold=hold)
    
        
        
        sdr.averaging_arm(1)
    
        ave_stat,retry = [0,0,0],0
        while ave_stat[0]==0:
            #time.sleep(1)
            ave_stat=sdr.averaging_status()
            #print("acquired: "+str(ave_stat[1]))
            if ave_stat[1]==averages:
                retry+=1
                if retry>5:
                    break
            
            
        y = sdr.averages_acquire(samples)/averages
        
        if return_all==True:
            collection.append(y)
        
        ftmp = abs(np.fft.fft(y))/len(y)
        #amplitude=ftmp[int(round(ro_length*dm_freq,10))]
        
        data[i] =  2*ftmp[int(ro_par[2]*dm_freq)]
        
        if progress_bar==True:
            pb(i+1,' Extimated: '+time_difference(step_start,points-i-1)+'               ' )

    if progress_bar==True:
        pb(points, 'Time: '+time_difference(cycle_start)+'                                      ')
    
    sdr.averaging_arm(0)
    sdr.awg_arm(3,0)
    sgro.output(0)
    awg.run(0)
    sgLO.output(0)
    
    if return_all == True:
        return data,collection
    else:
        return data