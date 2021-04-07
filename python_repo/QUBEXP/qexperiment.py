# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:00:48 2018

@author: User
"""


from .UtilityFunctions import *
import numpy as np
import DataModule as dm
import time
from UtilitiesLib import progressive_plot_2d,filetodictionary
import matplotlib.pyplot as plt
import IQCALIBRATION
from PULSE import Pulse,Sequence


class Experiment():
    def __init__(self,CryoID,exp_dict=None):
        if exp_dict is None:
            self.exp_dict = dict(CryostatID=CryoID,
                             sgro_frequency = 6., #GHz
                             sgro_power = -40, #dBm
                             sgro_pulse_length= 1000, #ns
                             repetition_rate = 500, #mus
                             averages = 1000,
                             samples = 1000, #1 samples = 1 ns
                             sgro_pulse_delay = 0, #ns
                             dm_freq = 50, #MHz
                             LOpwr = 8 #dBm
                             )
        else:
            self.exp_dict = exp_dict
            
        self.gen_dict = {}
        #self.gen_pointers={}
        
        self._dev_ids = ['sgro','sgLO','sgex','IQLO','AWG','DIG']
    #------------------------------------------------------------------------------ Custom exceptions
    class __BASEEXC(Exception):
        pass
    
    class _EXPEXC(__BASEEXC):
        def __init__(self,Expression,Message):
            self.Expression = Expression
            self.Message = Message
            
    #------------------------------------------------------------------------------- Parameters functions
    def CryoID(self,CryoID=None):
        if CryoID is None:
            return self.exp_dict['CryostatID']
        else:
            if type(CryoID)!=str:
                self._EXPEXC('CryoID must be a string\n','TypeError')
            self.exp_dict['CryostatID']=CryoID
            
    
    
    def list_dev_ids(self):
        print(self._dev_ids)
    
    def __check_dev_id(self,dev_id):
        dev_id = dev_id#.lower()
        try:
            self._dev_ids.index(dev_id)
            return dev_id
            
        except ValueError:
            self._dev_ids.append(dev_id)
            print('dev_id inserted\n')
            return dev_id
    
    def assign_generator(self,dev_id,dev,**k):
        
        dev_id=self.__check_dev_id(dev_id)
            
        
        tmp = dict(devid=dev.id,channel=dev.ch)
        
        
        #dev.power(-80) #safe
        
        tmp.update({'device':dev})
        tmp.update(**k)
        
        
        self.gen_dict.update({dev_id:tmp})

    def assign_digitizer(self,dev_id,dev,**k):
        
        dev_id = self.__check_dev_id(dev_id)
            
        
        if dev.id == 'ADQ14':
            tmp = dict(devid=dev.id,boardnum=dev.devnum)
            tmp.update(**k)
        elif dev.id =='DIGKEY':
            tmp = dict(devid='DIGKEY',slot=dev._slot,chassis=dev._chassis)
            tmp.update(**k)
        else:
            print('WRONG digitizer ID: {}\n'.format(dev.id))
            return
        
        tmp.update({'device':dev})
        
        
        self.gen_dict.update({dev_id:tmp})

    def assign_awg(self,dev_id,dev,**k):
        
        dev_id = self.__check_dev_id(dev_id)
            
        
        if dev.id == 'AWGKEY':
            tmp = dict(devid='AWGKEY',slot=dev._slot,chassis=dev._chassis)
            tmp.update(**k)
        
        else:
            print('WRONG awg ID: {}\n'.format(dev.id))
            return
        
        tmp.update({'device':dev})
        
        
        self.gen_dict.update({dev_id:tmp})    
 
    
    def _setup_rogen(self):
        sgro = self.gen_dict['sgro']['device']
        
        
        sgro.power(self.exp_dict['sgro_power'])
        sgro.frequency(self.exp_dict['sgro_frequency'])
        
        sgro.reference()
        sgro.instr.alc(0)
        delay = self.exp_dict['sgro_pulse_delay']
        try:
            delay += self.gen_dict['sgro']['trig_delay']
        except:
            pass
        sgro.instr.pulse_triggered(1,self.exp_dict['sgro_pulse_length'],delay)
        
        sgro.output(1)

        sgLO = self.gen_dict['sgLO']['device']
        sgLO.power(self.exp_dict['LOpwr'])
        sgLO.reference()
        try:
            sgLO.instr.alc(1)
            sgLO.instr.pulse_triggered(0)
        except:
            pass
        sgLO.output(1)

        return sgro,sgLO
    
    def _setup_exgen(self,expid,mode='CW'):
        sg = self.gen_dict[expid]['device']
        
        
        sg.power(self.exp_dict[expid+'_power'])
        
        try:
            sg.frequency(self.exp_dict[expid+'_frequency'])
        except:
            sg.frequency(6.)
        
        sg.reference()
        if mode.upper() == 'CW':
            try:
                sg.instr.alc(1)
                sg.instr.pulse_triggered(0,0,0)
            except:
                pass
        elif mode.upper() == 'PULSE':
            sg.instr.alc(0)
            delay=self.exp_dict[expid+'_pulse_delay']
            try:
                delay += self.gen_dict[expid]['trig_delay']
            except:
                pass
            
            sg.instr.pulse_triggered(1,self.exp_dict[expid+'_pulse_length'],delay)
        else:
            print('Wrong mode inserted\n')
            raise TypeError
            
        sg.output(1)

        return sg
    
    def _setup_dig(self):
        tmp = self.gen_dict['DIG']
        
        
        if tmp['devid']=='ADQ14':
            
            adq = tmp['device']
            adq.Trigger_mode('INT',np.round(self.exp_dict['repetition_rate']*1e-6,9))
            adq.trigger_output_setup('INT','RISE')
            delay = self.exp_dict['sgro_pulse_delay']
            try:
                delay += self.gen_dict['DIG']['trig_delay']
            except:
                pass            
            self.exp_dict['samples']=adq.Acquisition_setup(self.exp_dict['averages'],self.exp_dict['samples'],delay)
            return adq
        else:
            print('Not implemented yet')
            raise TypeError
        

    def measure_wave(self,x,sgro,sgLO,adq,conversion=True,amp=True):
        if x is not None:
            sgro.frequency(np.round(x,9))
            sgLO.frequency(np.round(x+self.exp_dict['dm_freq']/1000,9))
    
        
        data = adq.Acquire_data(1,conversion)
        if amp==True:
            SF = adq.SF
            ft = np.abs(np.fft.fft(data))/len(data)
            
            index = int(len(data)*self.exp_dict['dm_freq']/1000/SF)
            
            return 2*np.max(ft[index-2:index+2])
        else:
            return data
        
    def test_readout(self,freq,engine='p'):
        import bokeh.plotting as bp
        
        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()
        test = self.measure_wave(freq,sgro,sgLO,adq,0,0)
        
    
        xaxis = np.linspace(0,1,len(test))
        ft = np.abs(np.fft.fft(test))/len(test)
        
        if engine=='p':
            plt.plot(test)
            plt.figure()
            plt.plot(xaxis,ft)
        elif engine=='b':
            tools = ['box_zoom', 'pan', 'wheel_zoom', 'reset',
                         'save', 'hover']
            
            fig1 = bp.figure(tools=tools,
                                height=300,
                                sizing_mode='scale_width',
                                title= 'Signal')
            fig1.line(xaxis*1e4, test)
            fig2 = bp.figure(tools=tools,
                                height=300,
                                sizing_mode='scale_width',
                                title= 'Fourier Transform')
            fig2.line(xaxis*1e3,ft)
            
            for fig in [fig1,fig2]:
                bp.show(fig)
        else:
            print('Wrong engine selected: {p,b}')
    
    def do_readout_sweep(self,freq_sweep,power=None):
        if power != None:
            self.exp_dict['sgro_power']=power
            
        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()
        
        
        tmp = np.ndarray(len(freq_sweep))
        
        time_start = time.time()

        temp_start, temp_start_time= read_temperature(self.CryoID())
        try:
            for i,f in enumerate(freq_sweep):
                tmp[i]= self.measure_wave(f,sgro,sgLO,adq,1,1)*1e3
                progressive_plot_2d(freq_sweep[:i],tmp[:i],'-o')
        except KeyboardInterrupt:
            freq_sweep = freq_sweep[:i]
            tmp = tmp[:i]
            
        #creating data module    
        data = dm.data_table((freq_sweep,tmp),('Frequency (GHz)','Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time     
        
        
        data.time_stop = time.time()
        data.temp_stop,data.temp_stop_time= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        return data

    def do_spectroscopy_sweep(self,freq_sweep,pulse_length=None,devid='sgex',power=None,ro_delay=None):
        if power != None:
            self.exp_dict[devid+'_power']=power

        if pulse_length is not None:
            self.exp_dict[devid+'_pulse_length']=pulse_length
        
        try:
            self.exp_dict[devid+'_pulse_delay']
        except KeyError:
            print(devid+'_pulse_delay set to 0\n')
            self.exp_dict[devid+'_pulse_delay']=0

        sgex = self._setup_exgen(devid,'PULSE')        
        if ro_delay is None:
            self.exp_dict['sgro_pulse_delay']=sgex.instr.pulse_delay()+sgex.instr.pulse_width()
        else:
            self.exp_dict['sgro_pulse_delay']=ro_delay
        sgro,sgLO = self._setup_rogen()

        adq = self._setup_dig()
        
        
        tmp = np.ndarray(len(freq_sweep))
        
        time_start = time.time()

        temp_start, temp_start_time= read_temperature(self.CryoID())
        try:
            for i,f in enumerate(freq_sweep):
                sgex.frequency(f)
                tmp[i]= self.measure_wave(self.exp_dict['sgro_frequency'],sgro,sgLO,adq,1,1)*1e3
                progressive_plot_2d(freq_sweep[:i],tmp[:i],'-o')
        except KeyboardInterrupt:
            freq_sweep = freq_sweep[:i]
            tmp = tmp[:i]
            
        #creating data module    
        data = dm.data_table((freq_sweep,tmp),('Frequency (GHz)','Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time        

        
        data.time_stop = time.time()
        data.temp_stop,data.temp_stop_time= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        
        return data

    def do_calibration(self,freq_to_cal=None,awg_freq=123,qubit_id='q1', cal_amp=0.5,chI=0,SAID= 'SH',freq_sb = 'L', fitpoints=7,LO_pwr=13,show_steps=False):
        """Perform an IQ mixed calibration
        
        pars:
            - freq_to_cal = frequency to calibrate in GHz (def None = taken from the exp. dictionary)
            - awg_freq = awg frequencty to be used in MHz
            - qubit_id = if frequency is None (def), the qubit_id frequency will be taken
                the calibration will be assigned to this qubit
            - cal_amp = calibration amplitude in V (def 0.5)
            - chI: channel number connected to the I mixer port (def 0)
                NOTE:chQ = chI+1
            - SAID = Spectrum analyzer ID: ['SH','RS'] (def 'SH')
            - freq_sb = which sideband you want to calibrate: ['L','R'], (def 'L')
            - fitpoint: number of points to add around the minimum to improve the fit
                tradeoff between speed and quality (def 7)
            - LO_pwr: LO generator power (def 13)
            - show_steps: shows all the calibration plots (def False)
        """
        
        if freq_to_cal is None:
            freq_to_cal = self.exp_dict[qubit_id]['frequency']
        
        awg = self.gen_dict['AWG']['device']
        IQLO = self.gen_dict['IQLO']['device']
        cal = IQCALIBRATION.IQCal_KEYAWG(awg,IQLO,SAID,AWG_channel_cal_amplitude=cal_amp)
        cal.initialize_calibration([awg_freq,chI],freq_to_cal,freq_sb,LO_pwr)
        cal.measure_SB(1,1)
        cal.do_calibration(fitpoints,show_steps=show_steps,timeout=5,save=False)
        bands=cal.measure_SB(1,1)
        
        cal.calibration.calibration_dictionary['Calibration results']=bands
        print(cal.calibration.calibration_dictionary['Calibration results'])
        
        fpath='calibrations/{}V-cal_amp-freq-{}GHz-LO-{}GHz-AWG-{}MHz-{}'.format(cal_amp,freq_to_cal,cal.calibration.Sidebands()[1],awg_freq,qubit_id)
        cal.save_calibration(fpath,True)
        self.exp_dict[qubit_id].update({'cal_file_path':fpath})
        cal.close_SA_connection()


    def _setup_IQLO(self,IQLO,upmix_freq,cal_dict):
        IQLO.reference()
        IQLO.power(cal_dict['LO power'])
        if cal_dict['Sideband'] == 'RSB':
            IQLO.frequency(np.round(upmix_freq-cal_dict['AWG frequency']/1e3,9))
        else:
            IQLO.frequency(np.round(upmix_freq+cal_dict['AWG frequency']/1e3,9))
        
        try:
            IQLO.instr.alc(1)
            IQLO.instr.pulse_triggered(0)
        except:
            pass
        
        IQLO.output(1)
        
        
    def do_amplitude_Rabi_experiment(self,amp_sweep,pulse_length,qubit_id='q1',autodelay=True): 
        awg = self.gen_dict['AWG']['device']
        IQLO = self.gen_dict['IQLO']['device']
        
        
        
        cal_dict = filetodictionary(self.exp_dict[qubit_id]['cal_file_path'])
        chI = cal_dict['AWG chI']
        
        awg.apply_correction(cal_dict,0.5)    #arbitrary, to show waves if needed
        #AWG setup
        for i in [chI,chI+1]:
            awg.mode(i,'AWG')
            awg.modulation(i,0)
        
        #ph_corr = cal_dict['Phase correction chQ']
        #awg.register(0,int(awg_pulse_length/10+10))
        #awg.register(1,QUBEXP.set_phase_in_reg_deg(270,ph_corr))
        #awg.register(2,0)
        #awg.register(3,QUBEXP.set_phase_in_reg_deg(270,ph_corr))
        #awg.register(4,0)#awg_pulse_length+200) #safer
        
        
        
            
        awg.clear_waves()
        awg.clear_channel_queue(3<<chI)
        
        
        p0ch0 = Pulse(awg.frequency(chI)/1e3,awg.phase(chI),Width= pulse_length,Wait=10,Shape='g',SF=awg.SF)
        p0ch1 = Pulse(awg.frequency(chI+1)/1e3,awg.phase(chI+1),Width= pulse_length,Wait=10,Shape='g',SF=awg.SF)
        awg.insert_array(p0ch0.generate(),'p0ch0')
        awg.insert_array(p0ch1.generate(),'p0ch1')
        #awg.load_waves_in_AWG_memory()
        awg.queue_in_channel(chI,'p0ch0',6,Repetitions=0)
        awg.queue_in_channel(chI+1,'p0ch1',6,Repetitions=0)
        
        
        #setup LO
        self._setup_IQLO(IQLO,self.exp_dict[qubit_id]['frequency'],cal_dict)
        #setup readout 
        if autodelay:
            self.exp_dict['sgro_pulse_delay']= pulse_length

        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()
        
        #measure
        
        tmp = np.ndarray(len(amp_sweep))
        
        time_start = time.time()

        temp_start, temp_start_time= read_temperature(self.CryoID())
        
        
        awg.start_multiple()
       
        try:
             for i,am in enumerate(amp_sweep):
                awg.apply_correction(cal_dict,am)
                tmp[i]= self.measure_wave(self.exp_dict['sgro_frequency'],sgro,sgLO,adq,1,1)*1e3
                progressive_plot_2d(amp_sweep[:i],tmp[:i],'-o')
        except KeyboardInterrupt:
            amp_sweep = amp_sweep[:i]
            tmp = tmp[:i]
            awg.stop_multiple()
        
        awg.stop_multiple()
        #creating data module    
        data = dm.data_table((amp_sweep,tmp),('AWG Amplitude (V)','Transmitted Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time        
        
        
        data.time_stop = time.time()
        data.temp_stop,data.temp_stop_time= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        data.insert_par(awg_pulse_length = pulse_length, cal_dict = cal_dict)

        return data

    def _change_ro_delay(self,tau):
        sgro = self.gen_dict['sgro']['device']
        adq = self.gen_dict['DIG']['device']
        
        delay = self.exp_dict['sgro_pulse_delay']
        try:
            delay += self.gen_dict['sgro']['trig_delay']
        except:
            pass
        sgro.instr.pulse_triggered(1,self.exp_dict['sgro_pulse_length'],delay+tau)
        
        delay = self.exp_dict['sgro_pulse_delay']
        try:
            delay += self.gen_dict['DIG']['trig_delay']
        except:
            pass            
        self.exp_dict['samples']=adq.Acquisition_setup(self.exp_dict['averages'],self.exp_dict['samples'],delay+tau)
            
    def do_T1_experiment(self,time_sweep,qubit_id='q1',autodelay=True): 
        awg = self.gen_dict['AWG']['device']
        IQLO = self.gen_dict['IQLO']['device']
        
        
        
        cal_dict = filetodictionary(self.exp_dict[qubit_id]['cal_file_path'])
        chI = cal_dict['AWG chI']
        
        
        #AWG setup
        awg.apply_correction(cal_dict,self.exp_dict[qubit_id]['pi_pulse_amp'])    #arbitrary, to show waves if needed
        for i in [chI,chI+1]:
            awg.mode(i,'AWG')
            awg.modulation(i,0)
        
        #ph_corr = cal_dict['Phase correction chQ']
        #awg.register(0,int(awg_pulse_length/10+10))
        #awg.register(1,QUBEXP.set_phase_in_reg_deg(270,ph_corr))
        #awg.register(2,0)
        #awg.register(3,QUBEXP.set_phase_in_reg_deg(270,ph_corr))
        #awg.register(4,0)#awg_pulse_length+200) #safer
        
        
        
        awg.autoset=0
        awg.clear_waves()
        awg.clear_channel_queue(3<<chI)
        
        pulse_length = self.exp_dict[qubit_id]['pulse_length']
        p0ch0 = Pulse(awg.frequency(chI)/1e3,awg.phase(chI),Width= pulse_length,Wait=10,Shape='g',SF=awg.SF)
        p0ch1 = Pulse(awg.frequency(chI+1)/1e3,awg.phase(chI+1),Width= pulse_length,Wait=10,Shape='g',SF=awg.SF)
        awg.insert_array(p0ch0.generate(),'p0ch0')
        awg.insert_array(p0ch1.generate(),'p0ch1')
        #awg.load_waves_in_AWG_memory()
        awg.queue_in_channel(chI,'p0ch0',6,Repetitions=0)
        awg.queue_in_channel(chI+1,'p0ch1',6,Repetitions=0)
        
        
        awg.set_channel(load_waves_in_memory=True)
        awg.autoset=1
        
        #setup LO
        self._setup_IQLO(IQLO,self.exp_dict[qubit_id]['frequency'],cal_dict)
        #setup readout 
        if autodelay:
            self.exp_dict['sgro_pulse_delay']= pulse_length
        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()
        
        
        #measure
        
        tmp = np.ndarray(len(time_sweep))
        
        time_start = time.time()

        temp_start, temp_start_time= read_temperature(self.CryoID())
        
        
        awg.start_multiple()
       
        try:
             for i,tau in enumerate(time_sweep):
                self._change_ro_delay(tau)
                
                tmp[i]= self.measure_wave(self.exp_dict['sgro_frequency'],sgro,sgLO,adq,1,1)*1e3
                progressive_plot_2d(time_sweep[:i],tmp[:i],'-o')
        except KeyboardInterrupt:
            time_sweep = time_sweep[:i]
            tmp = tmp[:i]
            
        
        awg.stop_multiple()
        self._change_ro_delay(0)
        #creating data module    
        data = dm.data_table((time_sweep+pulse_length/2,tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time        
        
        
        data.time_stop = time.time()
        data.temp_stop,data.temp_stop_time= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        data.insert_par(awg_pulse_length = pulse_length, cal_dict = cal_dict)

        return data


    def do_T2_experiment(self,time_sweep,osc_period,qubit_id='q1',autodelay=True): 
        awg = self.gen_dict['AWG']['device']
        IQLO = self.gen_dict['IQLO']['device']
        
        
        
        cal_dict = filetodictionary(self.exp_dict[qubit_id]['cal_file_path'])
        chI = cal_dict['AWG chI']
        
        
        #AWG setup
        awg.apply_correction(cal_dict,self.exp_dict[qubit_id]['pi_pulse_amp'])    #arbitrary, to show waves if needed
        for i in [chI,chI+1]:
            awg.mode(i,'AWG')
            awg.modulation(i,0)

        awg.clear_waves()
        awg.clear_channel_queue(3<<chI)
        

        pulse_length = self.exp_dict[qubit_id]['pulse_length']
        p0ch0 = Pulse(awg.frequency(chI)/1e3,awg.phase(chI),Width= pulse_length,Wait=10,Shape='g',SF=awg.SF,Amplitude=0.5)
        p0ch1 = Pulse(awg.frequency(chI+1)/1e3,awg.phase(chI+1),Width= pulse_length,Wait=10,Shape='g',SF=awg.SF,Amplitude=0.5)
        awg.insert_array(p0ch0.generate(),'p0ch0')
        awg.insert_array(p0ch1.generate(),'p0ch1')


        def load_waves(tau,osc_period):
            awg.autoset=0
            awg.remove_wave_by_ID('p1ch0')
            awg.remove_wave_by_ID('p1ch1')
            
            
            new_phase = np.degrees(2*np.pi*awg.frequency(chI)/1e3*(pulse_length+10+tau))
            if osc_period>0:
                new_phase += 360/osc_period*tau
            
            
            p1ch0 = Pulse(awg.frequency(chI)/1e3,awg.phase(chI)+new_phase,Width= pulse_length,Wait=10,Shape='g',SF=awg.SF,Amplitude=0.5)
            p1ch1 = Pulse(awg.frequency(chI+1)/1e3,awg.phase(chI+1)+new_phase,Width= pulse_length,Wait=10,Shape='g',SF=awg.SF,Amplitude=0.5)
            
            awg.insert_array(p1ch0.generate(),'p1ch0')
            awg.insert_array(p1ch1.generate(),'p1ch1')
            
            
            awg.clear_channel_queue(3<<chI)
            
            awg.queue_in_channel(chI,'p0ch0',6,Repetitions=1)
            awg.queue_in_channel(chI+1,'p0ch1',6,Repetitions=1)
            awg.queue_in_channel(chI,'p1ch0',0,Delay=tau)
            awg.queue_in_channel(chI+1,'p1ch1',0,Delay=tau)
            awg.autoset=1
            awg.set_channel(load_waves_in_memory=True)
        load_waves(0,0)


        #setup LO
        self._setup_IQLO(IQLO,self.exp_dict[qubit_id]['frequency'],cal_dict)
        #setup readout 
        if autodelay:
            self.exp_dict['sgro_pulse_delay']= 2*(pulse_length+10)
        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()

        #measure
        
        tmp = np.ndarray(len(time_sweep))
        #data = dm.data_table((time_sweep,tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        time_start = time.time()

        temp_start, temp_start_time= read_temperature(self.CryoID())
        
        
        awg.start_multiple(0xf)
       
        try:
            for i,tau in enumerate(time_sweep):
                load_waves(tau,osc_period)#loading waves stop the AWG
                awg.start_multiple()
                self._change_ro_delay(tau)
                #time.sleep(0.01)
                
                tmp[i]= self.measure_wave(self.exp_dict['sgro_frequency'],sgro,sgLO,adq,1,1)*1e3
                progressive_plot_2d(time_sweep[:i]+(pulse_length+10),tmp[:i],'-o')
        except KeyboardInterrupt:
            time_sweep = time_sweep[:i]
            tmp = tmp[:i]
            
        
        awg.stop_multiple()
        self._change_ro_delay(0)
        #creating data module    
        data = dm.data_table((time_sweep+pulse_length+10,tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time
        
        data.time_stop = time.time()
        data.temp_stop,data.temp_stop_time= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        data.insert_par(awg_pulse_length = pulse_length, cal_dict = cal_dict,osc_period=osc_period)

        return data

    def do_time_Rabi_experiment(self,amp,pulse_len_sweep,qubit_id='q1'):
        awg = self.gen_dict['AWG']['device']
        IQLO = self.gen_dict['IQLO']['device']
        
        
        
        cal_dict = filetodictionary(self.exp_dict[qubit_id]['cal_file_path'])
        chI = cal_dict['AWG chI']
        
        #awg.apply_correction(cal_dict,amp)    #arbitrary, to show waves if needed
        #AWG setup
        
        for i in [chI,chI+1]:
            awg.mode(i,'AWG')
            awg.modulation(i,0)

        def rabi_block(t):
            time_pulse=Pulse(Frequency=awg.frequency(chI)/1e3, Phase=0.0, Shape='g', Width=t, SF=awg.SF, ID='time_pulse')
            pulselist=Sequence([time_pulse])
                
            
            
            awg.insert_sequence_object_IQ(
            seq=pulselist,
            cal_dict=cal_dict,
            amplitude=amp,
            clear_channels_queue=True,
            )
            """
            dummy = np.hstack((np.ones(100),np.zeros(10)))
            awg.clear_channel_queue(1<<2)
            awg.insert_array(dummy,'dummy')
            awg.queue_in_channel(2,'dummy',6)
            """
            
            return pulselist.total_width()                         


        #setup LO
        self._setup_IQLO(IQLO,self.exp_dict[qubit_id]['frequency'],cal_dict)
        #setup readout

        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()

        #measure
        
        tmp = np.ndarray(len(pulse_len_sweep))
        #data = dm.data_table((time_sweep,tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        time_start = time.time()
        temp_start, temp_start_time= read_temperature(self.CryoID())                
        #awg.start_multiple(0xf)
        try:
            for i,tau in enumerate(pulse_len_sweep):                
                
                tot_length=rabi_block(tau)#loading waves stops the AWG
                self._change_ro_delay(tot_length)
                print(tot_length)
                awg.start_multiple()
                #time.sleep(0.01)
                tmp[i]= self.measure_wave(self.exp_dict['sgro_frequency'],sgro,sgLO,adq,1,1)*1e3
                
                progressive_plot_2d(pulse_len_sweep[:i],tmp[:i],'-o')
        except KeyboardInterrupt:
            pulse_len_sweep = pulse_len_sweep[:i]
            tmp = tmp[:i]
            
        
        awg.stop_multiple()
        self._change_ro_delay(0)
        #creating data module    
        #data = dm.data_table((pulse_len_sweep+self.exp_dict['sgro_pulse_delay']-self.exp_dict[qubit_id]['pulse_length'],tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        data = dm.data_table((pulse_len_sweep,tmp),('Pulse length (ns)','Transmitted Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time

        data.temp_stop,data.temp_stop_time= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        data.insert_par(amp = amp, cal_dict = cal_dict)
        
        return data  


                

    def do_Spinlock_experiment(self,time_sweep,rab_amp,qubit_id='q1',df=0): #df in MHz

        awg = self.gen_dict['AWG']['device']
        IQLO = self.gen_dict['IQLO']['device']
    

        cal_dict = filetodictionary(self.exp_dict[qubit_id]['cal_file_path'])
        chI = cal_dict['AWG chI']
        
        awg.apply_correction(cal_dict,1.)
        for i in [chI,chI+1]:
            awg.mode(i,'AWG')
            awg.modulation(i,0)
            
        hpi_amp=(self.exp_dict[qubit_id]['pi_pulse_amp'])/2
        hpi_len=self.exp_dict[qubit_id]["pulse_length"]

        awg.clear_waves()
        def block(rab_amp,t,df):
            hpi1=Pulse(Frequency=awg.frequency(chI)/1e3, Phase=0.0, Shape='g', Width=hpi_len, Amplitude=hpi_amp, SF=awg.SF, ID='hpi1')
            
            hpi2=Pulse(Frequency=awg.frequency(chI)/1e3, Phase=0.0, Shape='g', Width=hpi_len, Amplitude=hpi_amp, SF=awg.SF, ID='hpi2')
            
            if t==0:
                pulselist=Sequence([hpi1,hpi2])
            else:
                rab=Pulse(Frequency=awg.frequency(chI)/1e3 +df/1e3, Phase=-90, Shape='g', Width=t, Amplitude=rab_amp, SF=awg.SF, ID='rab')
                pulselist=Sequence([hpi1,rab,hpi2])
                
            
            
            awg.insert_sequence_object_IQ(
            seq=pulselist,
            cal_dict=cal_dict,
            amplitude=1.,
            clear_channels_queue=True,
            )
            """
            dummy = np.hstack((np.ones(100),np.zeros(10)))
            awg.clear_channel_queue(1<<2)
            awg.insert_array(dummy,'dummy')
            awg.queue_in_channel(2,'dummy',6)
            """
            
            return pulselist.total_width()
            
        #setup LO
        self._setup_IQLO(IQLO,self.exp_dict[qubit_id]['frequency'],cal_dict)
        #setup readout

        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()

        #measure
        
        tmp = np.ndarray(len(time_sweep))
        #data = dm.data_table((time_sweep,tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        time_start = time.time()
        temp_start, temp_start_time= read_temperature(self.CryoID())                
        #awg.start_multiple(0xf)
        try:
            for i,tau in enumerate(time_sweep):                
                
                tot_length=block(rab_amp,tau,df)#loading waves stops the AWG
                self._change_ro_delay(tot_length)
                print(tot_length)
                awg.start_multiple()
                #time.sleep(0.01)
                tmp[i]= self.measure_wave(self.exp_dict['sgro_frequency'],sgro,sgLO,adq,1,1)*1e3
                
                progressive_plot_2d(time_sweep[:i],tmp[:i],'-o')
        except KeyboardInterrupt:
            time_sweep = time_sweep[:i]
            tmp = tmp[:i]
            
        
        awg.stop_multiple()
        self._change_ro_delay(0)
        #creating data module    
        #data = dm.data_table((time_sweep+self.exp_dict['sgro_pulse_delay']-self.exp_dict[qubit_id]['pulse_length'],tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        data = dm.data_table((time_sweep ,tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time

        data.temp_stop,data.temp_stop_time= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        data.insert_par(hpi_len = hpi_len, cal_dict = cal_dict,rab_amp=rab_amp)
        
        return data  


    def do_check_pulse(self,amp_pulse,len_pulse,N_sweep,qubit_id='q1'): #df in MHz

        awg = self.gen_dict['AWG']['device']
        IQLO = self.gen_dict['IQLO']['device']
    

        cal_dict = filetodictionary(self.exp_dict[qubit_id]['cal_file_path'])
        chI = cal_dict['AWG chI']
        
        awg.apply_correction(cal_dict,1.)
        for i in [chI,chI+1]:
            awg.mode(i,'AWG')
            awg.modulation(i,0)
            

        awg.clear_waves()
        def pulse(amp_pulse,len_pulse,N):
            check=Pulse(Frequency=awg.frequency(chI)/1e3, Phase=0.0, Shape='g', Width=len_pulse, Amplitude=1., SF=awg.SF, ID='check')
            list1=[check for i in range(N)]
            pulselist=Sequence(list1)             
            
            
            awg.insert_sequence_object_IQ(
            seq=pulselist,
            cal_dict=cal_dict,
            amplitude=amp_pulse,
            clear_channels_queue=True,
            )
            """
            dummy = np.hstack((np.ones(100),np.zeros(10)))
            awg.clear_channel_queue(1<<2)
            awg.insert_array(dummy,'dummy')
            awg.queue_in_channel(2,'dummy',6)
            """
            
            return pulselist.total_width()
            
        #setup LO
        self._setup_IQLO(IQLO,self.exp_dict[qubit_id]['frequency'],cal_dict)
        #setup readout

        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()

        #measure
        
        tmp = np.ndarray(len(N_sweep))
        #data = dm.data_table((time_sweep,tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        time_start = time.time()
        temp_start, temp_start_time= read_temperature(self.CryoID())                
        #awg.start_multiple(0xf)
        try:
            for i,nau in enumerate(N_sweep):                
                
                tot_length=pulse(amp_pulse,len_pulse,nau)#loading waves stops the AWG
                awg.start_multiple()
                
                self._change_ro_delay(tot_length)
                #time.sleep(0.01)
                tmp[i]= self.measure_wave(self.exp_dict['sgro_frequency'],sgro,sgLO,adq,1,1)*1e3
                
                progressive_plot_2d(N_sweep[:i],tmp[:i],'-o')
        except KeyboardInterrupt:
            N_sweep = N_sweep[:i]
            tmp = tmp[:i]
            
        
        awg.stop_multiple()
        self._change_ro_delay(0)
        #creating data module    
        data = dm.data_table((N_sweep,tmp),('Pulses number','Transmitted Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time

        data.temp_stop,data.temp_stop= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        data.insert_par(amp_pulse = amp_pulse, cal_dict = cal_dict, len_pulse=len_pulse)
        
        return data      
    
    def do_TEcho_experiment(self,time_sweep,osc_period,N_pulses=1,qubit_id='q1',autodelay=True): 
        awg = self.gen_dict['AWG']['device']
        IQLO = self.gen_dict['IQLO']['device']
    

        cal_dict = filetodictionary(self.exp_dict[qubit_id]['cal_file_path'])
        chI = cal_dict['AWG chI']
        
        
        #AWG setup
        awg.apply_correction(cal_dict,self.exp_dict[qubit_id]['pi_pulse_amp'])    #arbitrary, to show waves if needed
        for i in [chI,chI+1]:
            awg.mode(i,'AWG')
            awg.modulation(i,0)

        
        pulse_length = self.exp_dict[qubit_id]['pulse_length']
        pi_pulse_amp = self.exp_dict[qubit_id]['pi_pulse_amp']
        
        
        
            
        
        
        def load_waves(tau,osc_period,N_pulses):
            
            if N_pulses<1:
                print('N must be larger than 1\n')
                raise ValueError
            
            if ((tau/N_pulses) % 10)!= 0 or ((tau/N_pulses/2) % 10)!=0:
                print('Time steps must be a multiple of 10 ns\n')
                raise ValueError
            
            
            
            awg.clear_waves()
            
            
            
            
            #hpi-pulse
            hpI = Pulse(awg.frequency(chI)/1e3,0,Width= pulse_length,Wait=10,Shape='g',SF=awg.SF,Amplitude=0.5,ID='hpi1')
            pI = Pulse(awg.frequency(chI)/1e3,0,Width= pulse_length,Delay=tau/2/N_pulses,Wait=10,Shape='g',SF=awg.SF,Amplitude=1,ID='pi1')
            
            if osc_period>0:
                new_phase=360/osc_period*tau
            else:
                new_phase=0
                
            hpI2 = Pulse(awg.frequency(chI)/1e3,new_phase,Delay=tau/2/N_pulses,Width= pulse_length,Wait=10,Shape='g',SF=awg.SF,Amplitude=0.5,ID='hpi2')
            #pi-pulse


            seq = Sequence([hpI,pI])
            if N_pulses>1:
                pIN = Pulse(awg.frequency(chI)/1e3,0,Width= pulse_length,Delay=tau/N_pulses,Wait=10,Shape='g',SF=awg.SF,Amplitude=1,ID='piN')
                for i in range(N_pulses-1):
                        seq.sequence().append(pIN)
                        
            seq.sequence().append(hpI2)
            
            awg.insert_sequence_object_IQ(seq,cal_dict,pi_pulse_amp,True)
            
            #awg.load_waves_in_AWG_memory()
            
            #return total_length
        
        
        
        
        
        
        #setup LO
        self._setup_IQLO(IQLO,self.exp_dict[qubit_id]['frequency'],cal_dict)
        #setup readout
        if autodelay:
            self.exp_dict['sgro_pulse_delay']= (self.exp_dict[qubit_id]['pulse_length']+10)*(N_pulses+2)
        sgro,sgLO = self._setup_rogen()
        adq = self._setup_dig()

        #measure
        
        tmp = np.ndarray(len(time_sweep))
        #data = dm.data_table((time_sweep,tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        time_start = time.time()

        temp_start, temp_start_time= read_temperature(self.CryoID())
        
        
        #awg.start_multiple(0xf)
       
        try:
            for i,tau in enumerate(time_sweep):
                load_waves(tau,osc_period,N_pulses)#loading waves stops the AWG
                awg.start_multiple()
                self._change_ro_delay(tau)
                #time.sleep(0.01)
                
                tmp[i]= self.measure_wave(self.exp_dict['sgro_frequency'],sgro,sgLO,adq,1,1)*1e3
                progressive_plot_2d(time_sweep[:i]+self.exp_dict['sgro_pulse_delay']-self.exp_dict[qubit_id]['pulse_length'],tmp[:i],'-o')
        except KeyboardInterrupt:
            time_sweep = time_sweep[:i]
            tmp = tmp[:i]
            
        
        awg.stop_multiple()
        self._change_ro_delay(0)
        #creating data module    
        data = dm.data_table((time_sweep+self.exp_dict['sgro_pulse_delay']-self.exp_dict[qubit_id]['pulse_length'],tmp),('Pulses delay (ns)','Transmitted Amplitude (mV)'))
        
        data.time_start = time_start
        data.temp_start = temp_start
        data.temp_start_time = temp_start_time

        data.temp_stop,data.temp_stop_time= read_temperature(self.CryoID())
        data.par= self.gen_pars()
        data.insert_par(awg_pulse_length = pulse_length, cal_dict = cal_dict,osc_period=osc_period,N_pulses=N_pulses)

        return data             
    
    def gen_pars(self,dev_id=None):
        if dev_id is None:
            full={}
            for k in self.gen_dict:
                dev = self.gen_dict[k]['device']
                try:
                    tmp = dev.pars_dict
                except:
                    tmp = dev.parameters_dictionary()
                full.update({k:tmp})
            return full
        
        dev = self.gen_dict[dev_id]['device']
        try:
           tmp = dev.pars_dict
        except:
           tmp = dev.parameters_dictionary()
        return tmp
   
    def close_inst_connection(self):
        for d in self.gen_dict:
            try:
                self.gen_dict[d]['device'].close_driver()
            except AttributeError:
                try:
                    self.gen_dict[d]['device'].Disconnect()
                except:
                    pass
            
            
            