# -*- coding: utf-8 -*-
"""



"""

print('E5063A v1.1.0')

import pyvisa as vx
import DataModule as dm
import time
import numpy as np

class VNA(object):
    version = '1.1.0'
    
    def __init__(self,ip='192.168.187.123',CryoID='Freezer'):
        self.version='1.0.5'
        self.v = vx.Instrument(str(ip))
        self.CryoID = CryoID
        
    
    def cmd(self,str1,arg):
        self.v.write(''.join([str(str1),' ',str(arg),'\n']))

    def query(self,str1):
        return self.v.ask(''.join([str(str1),'?\n']))
        
    def cmd_only(self,str1,arg):
        if arg == '?':
            raise ValueError('No queries allowed. This is only a command.')
        else:
            self.cmd(str1,arg)
    
    def query_only(self,str1,arg='?'):
        if arg == '?':
            return (self.query(str1))
        else:
            raise ValueError('No arguments allowed. This is only a query.')
        
    def cmd_query(self,str1,arg='?'):
        if arg == '?':
            return self.query(str1)
        else:
            self.cmd(str1,arg)
    
    # Close connection
    def close(self):
        self.v.close()
    
    
    # Identification
    def identify(self):
        str1 = '*OPT'
        return self.query(str1)
        
    def output(self,arg='?'): # 
        """
            Turns RF output power on/off.

            Arg = ON|OFF|1|0
        """
        str1 = ':OUTP'
        return self.cmd_query(str1, arg)
        
        
    #POWER settings
    def power(self,power='?',channel=''):
        str1 = ''.join([':SOUR', str(channel), ':POW:LEV:IMM:AMPL'])
        return self.cmd_query(str1,power)

    
    # AVERAGE settings
    def average_reset(self,channel=''):
        str1 = ''.join([':SENSe',str(channel),':AVERage:CLEar'])
        return self.cmd_only(str1,'')
    
    def average_count(self,count='?',channel=''):
        str1 = ''.join([':SENSe',str(channel),':AVERage:COUNt'])
        return self.cmd_query(str1,count)
        
    def average_state(self,state='?',channel=''):
        str1 = ''.join([':SENSe',str(channel),':AVERage:STATe'])        
        return self.cmd_query(str1,state)
        
    # FREQUENCY sweep setting
    def freq_start(self,freq='?',channel=''):
        str1 = ''.join([':SENSe',str(channel),':FREQuency:STARt'])
        return self.cmd_query(str1,freq)
    
    def freq_stop(self,freq='?',channel=''):
        str1 = ''.join([':SENSe',str(channel),':FREQuency:STOP'])
        return self.cmd_query(str1,freq)
        
    def freq_center(self,freq='?',channel=''):
        str1 = ''.join([':SENSe',str(channel),':FREQuency:CENTer'])
        return self.cmd_query(str1,freq)
            
    def freq_span(self,freq='?',channel=''):
        str1 = ''.join([':SENSe',str(channel),':FREQuency:SPAN'])
        return self.cmd_query(str1,freq)
        
    def freq_npoints(self,points='?',channel=''):
        str1 = ''.join([':SENSe',str(channel),':SWEep:POINts'])
        return self.cmd_query(str1,points)
        
    # READING data
    def freq_read(self):
        str1 = 'CALCulate:TRACe:DATA:XAXis'
        return self.query(str1).split(',')
    
    def trace_read(self,trace=''):
        str1 = ''.join(['CALC:TRACe',str(trace),':DATA:FDATa'])
        dat = self.query(str1).split(',')
        return dat[0::2],dat[1::2]
        
    # Setting the IF bandwidth
    # This command sets/gets the IF bandwidth of selected channel (Ch).
    def IFBW(self,BW='?',channel=''):
        str1 = ''.join([':SENSe',str(channel),':BANDwidth:RESolution'])
        return self.cmd_query(str1,BW)
        
    def SPAR(self,Par='',Trace=1):
        if type(Par)!= str:
            print('Par must be a string')
            raise Exception('PAREXC')
        
        Par=Par.upper()
        if (Par=='S11') or (Par=='S12') or (Par=='S21') or (Par=='S22'):
                self.cmd_only('CALC1:PAR'+str(np.int(Trace))+':DEF',Par)
        elif Par=='':
            return self.query('CALC1:PAR'+str(np.int(Trace))+':DEF')
        else:
            print('No valid Par inserted')
            raise Exception('PAREXC')

    def traces_number(self,num=1):
        '''This function sets the number of traces (def 1)'''
        
        self.cmd_only('CALC1:PAR:COUN',np.int(num))
        
    def trace_select(self,num=1):
        '''This function will select the trace number (1 def)'''
        
        self.cmd_only('CALC1:PAR'+str(np.int(num))+':SEL','')

    def Format(self,Format=''):
        '''This function can be used to change the data format:
        
        'MLOG' : magntidue in dB
        'PHAS': phase
        'MLIN': linear magnitude
        'REAL': real part of the complex data
        'IMAG' imaginary part of the complex data
        'UPH': Extended phase
        'PPH': Positive phase
        '' (def): the format will be queried
        '''
        
        try:
            Format = Format.upper()
        except:
            print('Format must be a string')
            raise Exception('TypeError')
        
        types= ['MLOG',  'PHAS', 'MLIN', 'REAL',  'IMAG', 'UPH',  'PPH','']
        try:
            num=types.index(Format)
        except:
            print('Wrong format inserted')
            raise Exception('FormatError')
        
        if num==7:
            return self.cmd_query('Calc1:SEL:FORM')
        else:
            self.cmd_only('CALC1:SEL:FORM',Format)
    
    def temp_reading(self):
        ''' This function return the base temperature and the last update of its reading'''
        #import SensorReader as SR
        #a=SR.SensorReader(self.CryoID)
        #return a.base_temp(),a.last_update()
        
        
    def read_settings(self):
        """Returns current state of VNA parameters as dict"""        
        
        par = None
        freq = self.freq_read()
        freq_start = freq[0]
        freq_stop = freq[-1]
        
        freq_span = self.freq_span()
        freq_npoints = self.freq_npoints()
        BW = self.IFBW()
        Spar = self.SPAR()
        format_meas = self.Format()
        power = self.power()
        output = self.output()
        avg = self.average_count()
        #no_of_traces = self.trace_read()
        

        
        par = {'f_start (Hz)':freq_start, 'f_stop (Hz)':freq_stop, \
            'IF - BW (Hz)':BW, 'S_parameter ':Spar, 'Format':format_meas, \
            'freq span (Hz)':freq_span, 'freq npoints':freq_npoints, \
            'power (dbm)': power, 'output (0 off, 1 on)':output, 'averages':avg}
        return par
    
        
    def collect_single(self,f_range,npoints=1601,navg = 999,power=-50,wait=1,Spar = 'S21',Format='MLOG',
                       BW=1e3,Temp_reading=True):
        '''This function reads S-parameter from the VNA using the specified setting:
        f_range: frequency range in GHz. List type: [f_start, f_stop] 
        npoints: the the number of frequency points to measure
        navg = number of averaging measurements (average coefficient)
        power = RF_out power in dB
        wait =  data collection time in seconds
        BW = IF bandwidth'''
        self.IFBW(BW)
        self.freq_npoints(npoints)    
        self.freq_start(f_range[0]*1e9)
        self.freq_stop(f_range[1]*1e9)
        self.power(power)
        self.output(1)
        self.SPAR(Spar)
        self.Format(Format)
        if (navg==0):
            self.average_state(0)
        else:
            self.average_state(1)
            self.average_count(navg)

        dat= dm.data_2d()
        
        dat.time_start=time.strftime(dat.date_format,time.localtime())   
        
        if Temp_reading:
            dat.temp_start,dat.temp_start_time = self.temp_reading()
        
        self.average_reset()
        time.sleep(wait) # delay
        x = np.asarray(self.freq_read(),dtype='float')
        y = np.asarray(self.trace_read()[0],dtype='float') 
    
        if Temp_reading:
            dat.temp_stop,dat.temp_stop_time = self.temp_reading()
        
        dat.load_var(x/1e9,y)
        dat.time_stop=time.strftime(dat.date_format,time.localtime())   
        dat.insert_par(frange=f_range,npoints=npoints,navg=navg,power=power,wait=wait,Spar=Spar,BW=BW)
        self.output(0)  # Turn RF-Power off
        return dat

    def collect_single_ave(self,f_range,npoints=1601,navg = 100,power=-50,Spar = 'S21',Format='MLOG',BW=1e3,Temp_reading=True):
        '''This function reads S-parameter from the VNA using the specified setting:
        f_range: frequency range in GHz. List type: [f_start, f_stop] 
        npoints: the the number of frequency points to measure
        navg = number of averaging measurements (average coefficient)
        power = RF_out power in dB
        wait =  data collection time in seconds
        BW = IF bandwidth'''
        self.IFBW(BW)
        self.freq_npoints(npoints)
        self.freq_start(f_range[0]*1e9)
        self.freq_stop(f_range[1]*1e9)
        self.power(power)
        self.output(1)
        self.SPAR(Spar)
        self.Format(Format)
        if (navg==0):
            self.average_state(0)
        else:
            self.average_state(1)
            self.average_count(navg)
            self.v.write(':TRIG:SEQ:AVER ON')


        self.v.write(':TRIG:SEQ:SOUR BUS')
        self.v.write('INIT:CONT OFF')
        self.v.write('INIT:CONT ON')
        self.average_reset()


        self.v.write(':STAT:OPER:PTR 0')
        self.v.write(':STAT:OPER:NTR 16')
        self.v.write(':STAT:OPER:ENAB 16')
        self.v.write('*SRE 128')
        self.v.write('*CLS')

        dat= dm.data_2d()
        
        dat.time_start=time.strftime(dat.date_format,time.localtime())   
        
        if Temp_reading:
            dat.temp_start,dat.temp_start_time = self.temp_reading()
        

        self.v.write(':TRIG:SEQ:SINGLE')

        while int(self.v.ask('*STB?'))!=192:
            time.sleep(0.5)

        if Temp_reading:
            dat.temp_stop,dat.temp_stop_time = self.temp_reading()
        
        x = np.asarray(self.freq_read(),dtype='float')
        y = np.asarray(self.trace_read()[0],dtype='float')

        
        dat.load_var(x/1e9,y)

        dat.insert_par(frange=f_range,npoints=npoints,navg=navg,power=power,Spar=Spar,BW=BW)

        dat.time_stop=time.strftime(dat.date_format,time.localtime())   

        self.v.write('*CLS')
        self.v.write(':TRIG:SEQ:AVER OFF')
        self.v.write(':TRIG:SEQ:SOUR INT')
        self.output(0)  # Turn RF-Power off
        return dat

    def meas_complex_ave(self, f_range, npoints=1601, navg=100, power=-50,
                         Spar='S21', BW=1e3, autoSave=True,
                         filename='measurements/measurement.dm',
                         Use_Date=True, overwrite=False, Temp_reading=True,
                         parameters={}):
        """
            Measure and save as complex voltage data format.
            Uses the complex data module to directly store data on harddrive

            Parameters:
            f_range:    [[f_start, f_stop]] Frequency range
            npoints:    [INT] Number of Points
            navg:       [INT] Number of averages
            power:      [Float] VNA output power in dBm
            BW:         [Int] IF Bandwidth in Hz
            Spar:       [String, 'S21' default] Which S Parameter to measure
            Temp_reading: [Bool] Append temperature to datamodule file
            autosave:   [Bool] Automatically save data to disk (RECOMMENDED!!)
            filename:   [String] Filepath + Filename
            Use_Date:   [Bool] Date as leading part of filename
            overwrite:  [Bool] Force override
            parameters: [Dict] Additional parameters to save in dm
        """
        # Set VNA parameters
        self.IFBW(BW)
        self.freq_npoints(npoints)
        self.freq_start(f_range[0]*1e9)
        self.freq_stop(f_range[1]*1e9)
        self.power(power)
        self.output(1)
        self.SPAR(Spar)
        self.traces_number(2)
        self.trace_select(1)
        self.Format('REAL')
        self.trace_select(2)
        self.Format('IMAG')
        self.SPAR(Spar, 2)
        self.trace_select(1)
        if navg == 0:
            self.average_state(0)
        else:
            self.average_state(1)
            self.average_count(navg)
            self.v.write(':TRIG:SEQ:AVER ON')

        # Set device to external Control
        self.v.write(':TRIG:SEQ:SOUR BUS')
        self.v.write('INIT:CONT OFF')
        self.v.write('INIT:CONT ON')
        self.average_reset()
        # Tweaks for VNA
        self.v.write(':STAT:OPER:PTR 0')
        self.v.write(':STAT:OPER:NTR 16')
        self.v.write(':STAT:OPER:ENAB 16')
        self.v.write('*SRE 128')
        self.v.write('*CLS')
        # Set electric delay to 0
        self.cmd("CALC:TRAC1:CORR:EDEL:TIME", 0)
        self.cmd("CALC:TRAC2:CORR:EDEL:TIME", 0)
        # Initialize datamodule
        data = dm.data_cplx()
        data.time_start = time.strftime(data.date_format, time.localtime())

        # Get start T
        if Temp_reading:
            data.temp_start, data.temp_start_time = self.temp_reading()

        # Start VNA measurement
        self.v.write(':TRIG:SEQ:SINGLE')

        # Wait for device
        while int(self.v.ask('*STB?')) != 192:
            time.sleep(0.5)

        # Acquire data
        x = np.asarray(self.freq_read(), dtype=np.float)
        re = np.asarray(self.trace_read(1)[0], dtype=np.float)
        im = np.asarray(self.trace_read(2)[0], dtype=np.float)

        # Get stop T
        if Temp_reading:
            data.temp_stop, data.temp_stop_time = self.temp_reading()

        # Format datamodule and save
        data.load_var(x/1e9, re, im)
        data.insert_par(frange=f_range, npoints=npoints, navg=navg,
                        power=power, Spar=Spar, BW=BW, **parameters)

        data.time_stop = time.strftime(data.date_format, time.localtime())
        if (autoSave is True):
            data.save(filename, Use_Date, overwrite)
        else:
            return data

        # Return VNA to initial settings
        self.v.write('*CLS')
        self.v.write(':TRIG:SEQ:AVER OFF')
        self.v.write(':TRIG:SEQ:SOUR INT')
        # Turn of RF-Power
        self.output(0)

    def meas_complex_segm(self, segments, navg = 100, power=-50,Spar = 'S21',BW=1e3, \
                         autoSave = True, filename = 'measurements/measurement.dm', Use_Date=True, \
                         overwrite =False, Temp_reading=True):
        '''This function save datamodule with complex measured data.
        It measures the data according to segments, looking as follows:
        segments: array of dictionarys ( = [segment1, segment2,...])
            segment1 = {'start': x,'stop': x,'npoints': x, 'BW': x (optional), 'power': x (optional)}
        
        npoints: the the number of frequency points to measure
        navg = number of averaging measurements (average coefficient)
        power = RF_out power in dBm
        BW = IF bandwidth
        Spar = 'S21' default
        Temp_reading (True def) will append the temperature of the cryostat to the data
        
        if autoSave = False: datamodule is returned and data not saved
        filename = name under which it get saved (possible to include directory)
        Use_Date = automatically saves data as leading part of filename
        overwrite = measurement file gets overwritten (or not)
        '''
               
        segment_str = '5,0,1,1,0,0,'+str(len(segments))
        
        for d in segments:
            tmp = ','+str(d['start']*1e9)+','+str(d['stop']*1e9)+','+str(d['npoints'])
            
            try:
                tmp += ','+str(d['BW'])
            except:
                tmp += ','+str(BW)
            
            try:
                tmp += ','+str(d['power'])
            except:
                tmp += ','+str(power)
                

            segment_str += tmp
            
        
        
        self.v.write(':SENS:SWE:TYPE SEGM')
        self.v.write(':SENS:SEGM:DATA ' + segment_str)
        self.output(1)
        self.SPAR(Spar)
        self.traces_number(2)
        self.trace_select(1)
        self.Format('REAL')
        self.trace_select(2)
        self.Format('IMAG')
        self.SPAR(Spar, 2)
        self.trace_select(1)

        if (navg==0):
            self.average_state(0)
        else:
            self.average_state(1)
            self.average_count(navg)
            self.v.write(':TRIG:SEQ:AVER ON')

        # Set electric delay to 0
        self.cmd("CALC:TRAC1:CORR:EDEL:TIME", 0)
        self.cmd("CALC:TRAC2:CORR:EDEL:TIME", 0)
        self.v.write(':TRIG:SEQ:SOUR BUS')
        self.v.write('INIT:CONT OFF')
        self.v.write('INIT:CONT ON')
        self.average_reset()


        self.v.write(':STAT:OPER:PTR 0')
        self.v.write(':STAT:OPER:NTR 16')
        self.v.write(':STAT:OPER:ENAB 16')
        self.v.write('*SRE 128')
        self.v.write('*CLS')

        data = dm.data_cplx()
        
        data.time_start=time.strftime(data.date_format,time.localtime())   
        
        if Temp_reading:
            tmp1,tmp2=self.temp_reading()
            data.temp_start,data.temp_start_time = tmp1,tmp2
        

        self.v.write(':TRIG:SEQ:SINGLE')

        while int(self.v.ask('*STB?'))!=192:
            time.sleep(0.5)

        if Temp_reading:
            tmp1,tmp2=self.temp_reading()
            data.temp_stop,data.temp_stop_time = tmp1,tmp2
            
        
        x = np.asarray(self.freq_read(),dtype='float')
        re = np.asarray(self.trace_read(1)[0],dtype='float')
        im = np.asarray(self.trace_read(2)[0],dtype='float')

        
        data.load_var(x/1e9,re,im)

        data.insert_par(segments=segments,navg=navg,power=power,Spar=Spar,BW=BW)

        data.time_stop=time.strftime(data.date_format,time.localtime())   

        self.v.write('*CLS')
        self.v.write(':TRIG:SEQ:AVER OFF')
        self.v.write(':TRIG:SEQ:SOUR INT')
        
        self.v.write(':SENS:SWE:TYPE LIN')
        self.output(0)  # Turn off RF-Power
        if (autoSave is True):
            data.save(filename, Use_Date, overwrite)
        else:
            return data



    def collect_scan(self,f_range_mat,npoints_v=[1601],navg_v = [999],power_v=[-50],wait_v=[1],BW_v=[1e3]):
        x = []
        y = []
        range_mat = np.array(f_range_mat)
        len_loop = len(range_mat[:,0])

        def vector_handling(vector):
            if len(vector) == 1:
                out = vector[0]*np.ones(len_loop)
            else:
                out = vector
            return out            
    
        npoints = vector_handling(npoints_v)
        navg = vector_handling(navg_v)
        power = vector_handling(power_v)
        wait = vector_handling(wait_v)
        BW = vector_handling(BW_v)
    
        for i in np.arange(len_loop):
            f_range = range_mat[i,:]           
            dat_tmp = self.collect_single(f_range,npoints=npoints[i],navg=navg[i],power=power[i],wait=wait[i],BW=BW[i])   
            x = np.hstack((x,dat_tmp.x))
            y = np.hstack((y,dat_tmp.y)) 
        
        dat = dm.data_2d()  
        dat.load_var(x,y)   
        return dat

    def collect_single_correct(self,f_range,npoints=1601,navg=999,power=-50,corr_power=-10,wait=10,BW=1e3,Temp_reading=True):
            dat_cor = self.collect_single_ave(f_range,npoints,2,corr_power,BW=1e3,Temp_reading=Temp_reading)
            self.power(-70)
            time.sleep(1)
            dat_mes = self.collect_single(f_range,npoints,navg,power,wait,BW=BW,Temp_reading=Temp_reading)        
        
            
            dat_mes.y = dat_mes.y-dat_cor.y
            
            if Temp_reading:
                dat_mes.temp_start = dat_cor.temp_start
                dat_mes.temp_start_time = dat_cor.temp_start_time
                
            dat_mes.insert_par(corr_power=corr_power)
            
            return dat_mes

    def collect_single_correct_ave(self,f_range,npoints=1601,navg=100,power=-50,corr_power=-10,BW=1e3,Temp_reading=True):
            dat_cor = self.collect_single_ave(f_range,npoints,2,corr_power,BW=1e3,Temp_reading=Temp_reading)
            self.power(-70)
            time.sleep(1)
            dat_mes = self.collect_single_ave(f_range,npoints,navg,power,BW=BW,Temp_reading=Temp_reading)

            dat_mes.y= dat_mes.y-dat_cor.y

            if Temp_reading:
                dat_mes.temp_start = dat_cor.temp_start
                dat_mes.temp_start_time = dat_cor.temp_start_time
            
            dat_mes.insert_par(corr_power=corr_power)

            return dat_mes

    def collect_scan_correct(self,f_range_mat,npoints_v=[1601],navg_v = [999],power_v=[-50],corr_power_v=[-10],wait_v=[10],corr_wait_v=[1],BW_v=[1e3]):
            dat_cor = self.collect_scan(f_range_mat,npoints_v,navg_v,corr_power_v,corr_wait_v,BW_v)        
            dat_mes = self.collect_scan(f_range_mat,npoints_v,navg_v,power_v,wait_v,BW_v)        
        
            dat = dm.data_2d()
            dat.load_var(dat_mes.x,dat_mes.y-dat_cor.y)
            return dat
            
            
    def get_current_measurement_cplx(self):
        """Gets current complex measurement data and saves to cplx. datamodule"""
        data = dm.data_cplx()        
        
        x = np.asarray(self.freq_read(),dtype='float')
        re = np.asarray(self.trace_read(1)[0],dtype='float')
        im = np.asarray(self.trace_read(2)[0],dtype='float')

        
        data.load_var(x/1e9,re,im)
        
        data.time_stop=time.strftime(data.date_format,time.localtime())   

        self.v.write('*CLS')
        self.v.write(':TRIG:SEQ:AVER OFF')
        self.v.write(':TRIG:SEQ:SOUR INT')
        return data
        
        
        
    def get_current_measurement(self):
        """Gets current measurement data and saves to datamodule"""
        dat = dm.data_2d()        
        
        x = np.asarray(self.freq_read(),dtype='float')
        y = np.asarray(self.trace_read()[0],dtype='float')

        
        dat.load_var(x/1e9,y)
        
        dat.time_stop=time.strftime(dat.date_format,time.localtime())   

        self.v.write('*CLS')
        self.v.write(':TRIG:SEQ:AVER OFF')
        self.v.write(':TRIG:SEQ:SOUR INT')
        return dat
        






