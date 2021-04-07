# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:36:03 2017

@author: Oscar

v1.2.0 - OSC:
    - changed the lib to connect to multiple devices, SN inserted in IPList
    
v1.1.1 - OSC:
    - The instrument will automatically re-try to measure when the errorcode -11 is returned

v1.1.0 - OSC:
    - When initializing the class, it is now possible to specify the OS
    - installed linux libraries but it is under test
    - renamed the function GetSerialNumbersList to GetDevicesList to avoid confusion

v1.0.1 - OSC
- the function bandwidth has been modifies so that vbw is equal to rbw as default
- replace the function debug settings with default settings
"""

import DataModule as dm
import numpy as np
import time
from IPLIST import IPList
#from UtilitiesLib import time_difference
        

class SignalHound(object):
    import ctypes as ct
    import numpy as np
    import os
    
    version='1.2.0'
    def __init__(self,ID,OS='W'):
        """ID can be:
            - a string with the device name inserted in the IPList
            - an integer representing the device serial number
            - None -> no connection will be done automatically, call the 
                OpenDevice function later
        
        
        
        if OS is 'W', the WINDOWS 64bit library will be used (def),
        if it is 'L' the LINUX 64bit library will be used"""
        
        if type(ID)==str:
            ID = IPList[ID]
        
        if OS.upper() == 'W':
            self.API = self.ct.cdll.LoadLibrary(self.os.environ['PYTHONPATH']+"/sa_api.dll")
        elif OS.upper() == 'L':
            self.API = self.ct.cdll.LoadLibrary("/usr/local/lib/libsa_api.so")
        else:
            print('ERROR: Wrong OS inserted')
            raise Exception('OSERR')
        
        
        
        #set some restype
        self.API.saGetErrorString.restype = self.ct.c_char_p
        self.API.saGetAPIVersion.restype = self.ct.c_char_p
    
        #set some device parameters for SA124B
        self.MIN_FREQ = 100.0e3 #Hz
        self.MAX_FREQ = 13.0e9 #Hz
        self.MIN_SPAN = 1.0 #Hz
        self.MAX_REF = 20 #dBm
        self.MAX_ATTEN = 3
        self.MAX_GAIN = 2
        self.MIN_RBW = 0.1 #Hz
        self.MAX_RBW = 6.0e6 #Hz
        self.MIN_RT_RBW = 100.0 #Hz
        self.MAX_RT_RBW = 10000.0 #Hz
        self.MIN_IQ_BANDWIDTH = 100.0 #Hz
        self.MAX_IQ_DECIMATION = 7

        self.IQ_SAMPLE_RATE = 486111.111 
    
        #set some variables:
        self.modes_list = ['IDLE','SWEEP','RT','IQ']
        self._IQ_BW = [250e3,225e3,100e3,50e3,20e3,12e3,5e3,2e3]
        self.RTP = None
        self.detector = None
        self.mode = None
        
        if ID is not None:
            status = self.OpenDevice(ID)
                        
            
            if status !=0 :
                    print('An error occurred, code: '+str(status))
                    raise Exception("DEVERR")
        else:
            self.device_id = None
            
    
    
    
    def GetDevicesList(self):
        '''
This function returns the devices that are unopened in the current process. 
Up to 8 devices max will be returned. The serial numbers of the unopened 
devices are returned. The array will be populated starting at index 0 of the 
provided array. The integer pointed to by deviceCount will equal the number of 
devices reported by this function upon returning.'''

   
        """
        col= self.ct.c_int32*1
        row=self.ct.POINTER(self.ct.c_int32)*8 #max 8 devices
        buffer = row()
        for i in range(8):
            buffer[i]=col(0) #initialized to 0
        """    
        buffer = (self.ct.c_int*8)(0)
        device = self.ct.c_int(0)
        
        status = self.API.saGetSerialNumberList(buffer,self.ct.byref(device))#)self.ct.cast( device,self.ct.POINTER( self.ct.c_int) ))
        
        return np.array(buffer),np.int(device.value),status
    
    def OpenDevice(self,device_num):
        '''This function attempts to open the SA with the specifed serial 
        number. If a device is opened successfully, a handle to the device will 
        be  recorded in the class.device_id.'''
        
        tmp = self.ct.c_int32()
        status = self.API.saOpenDeviceBySerialNumber(self.ct.pointer(tmp),self.ct.c_int(device_num) )
        
        if status == 0:
            self.device_id = tmp.value
        else:
            self.device_id = None
        
        return status
    
    def CloseDevice(self):
        '''This function is called when you wish to terminate a connection with
        a device. Any resources the device has allocated will be freed and the 
        USB 2.0 connection to the device is terminated. The device closed will 
        be released and will become available to be opened again.
        Any activity the device is performing is aborted automatically before 
        closing'''
        
        return self.API.saCloseDevice(self.device_id)
    
    def Preset(self):
        """This function exists to invoke a hard reset of the device. This will 
        function similarly to a power cycle(unplug/re-connect the device). This 
        might be useful if the device has entered an undesirable or 
        unrecoverable state. This function might allow the software to perform 
        the reset rather than ask the user perform a power cycle.

        Functionally, in addition to a hard reset, this function closes the 
        device as if CloseDevice() was called. 
        
        This means the device handle becomes invalid and the device must be 
        reopened for use. This function is a blocking call and takes about 2.5 
        seconds to return. """
        
        return self.API.saPreset(self.device_id)
    
    def GetSerialNumber(self):
        ''' returns the opened device SN'''
        
        tmp = self.ct.c_int(0)
        
        status = self.API.saGetSerialNumber(self.device_id,self.ct.byref(tmp))
        if status != 0:
            print('An Error occurred, code: '+str(status))
            raise Exception('DEVERR')
        
        return tmp.value
    
    def error_code(self,status_number):
        '''This function converts a status number in the error string'''
        
        tmp = self.API.saGetErrorString(self.ct.c_int(status_number)) 
        
        return tmp
    
    def QueryTemperature(self):
        """Requesting the internal temperature of the device cannot be 
        performed while the device is currently active. To receive the 
        absolute current internal device temperature, ensure the device is 
        inactive by calling Abort() before calling this function. If the 
        device is active, the temperature returned will be the last temperature 
        returned from this function."""

        temp = self.ct.c_float(0)
        status = self.API.saQueryTemperature(self.device_id,self.ct.byref(temp))
        if status != 0:
            print('An Error occurred, code: '+str(status))
            raise Exception('DEVERR')
            
        return float(temp.value)
    
    def QueryVoltage(self):
        """A USB voltage below 4.55V may cause readings to be out of spec. 
        Check your cable for damage and USB connectors for damage or oxidation.
        """
    
        voltage = self.ct.c_float(0)
        
        status = self.API.saQueryDiagnostics(self.device_id, self.ct.byref(voltage))
        if status != 0:
            print('An Error occurred, code: '+str(status))
            raise Exception('DEVERR')
            
        return float(voltage.value)
    
    def QueryAPIVersion(self):
        
        return self.API.saGetAPIVersion()

    
    #------------------------------------------------------------------------------- Setup functions
    
    def ConfigAcquisition(self,detector,scale):
        """In this function, detector specifies how to produce the results of 
        the signal processing for the final sweep. Depending on settings, 
        potentially many overlapping FFTs will be performed on the input time 
        domain data to retrieve a more consistent and accurate final result. 
        When the results overlap the detector chooses whether to average the 
        results together, or maintain the minimum and maximum values. If 
        averaging is chosen, the min and max sweep arrays will contain the same
        averaged data. 
        
        The scale parameter will change the units of returned sweeps. 
        If LOG_SCALE is provided sweeps will be returned in amplitude unit dBm.
        If LIN_SCALE is chosen, the returned units will be in millivolts. 
        If the full scale units are specified, no corrections are applied to 
        the data and amplitudes are taken directly from the full scale input.
        
        
        Parameters:
        - detector: 'MIN_MAX' or 0
                    'AVERAGE' or 1
        - scale:    'LOG_SCALE' or 0
                    'LIN_SCALE' or 1
                    'LOG_FULL_SCALE' or 2
                    'LIN_FULL_SCALE' or 3
        
            
        
        """
        
        if type(detector) is str:
            if detector.upper() == 'MIN_MAX':
                detector = 0
                
            elif detector.upper() == 'AVERAGE':
                detector = 1
                
            else:
                print('Wrong detector specified')
                raise Exception('PARERR')

        detector = int(detector)   
        if detector<0 or detector>1:
            print('Wrong detector inserted')
            raise Exception('PARERR')
        
        self.detector = detector
                
        
        if type(scale) is str:
            if detector.upper() == 'LOG_SCALE':
                detector = 0
            elif detector.upper() == 'LIN_SCALE':
                detector = 1
            elif detector.upper() == 'LOG_FULL_SCALE':
                detector = 2
            elif detector.upper() == 'LIN_FULL_SCALE':
                detector = 3
            else:
                print('Wrong scale specified')
                raise Exception('PARERR')
                
        scale = int(scale)   
        if scale<0 or scale>3:
            print('Wrong scale inserted')
            raise Exception('PARERR')
        
        
        return self.API.saConfigAcquisition(self.device_id, self.ct.c_int(detector), self.ct.c_int(scale))
    
    def CenterSpan(self,center,span):
        '''This function sets the frequency range using Center frequency and 
        Span, everything in GHz'''
        
        
        return self.API.saConfigCenterSpan(self.device_id,self.ct.c_double(center*1e9),self.ct.c_double(span*1e9))
    
    def StartStop(self,start,stop):
        '''This function sets the frequency range using Start frequency and 
        Stop frequency, everything in GHz.
        
        If start is less than the minimum allowed it will be set to the minimum.
        If stop is greater than the maximum allowed it will be set to the maximum.
        
        '''
        
        if start<self.MIN_FREQ/1e9:
            start=self.MIN_FREQ/1e9
        
        if stop > self.MAX_FREQ/1e9:
            stop = self.MAX_FREQ/1e9
        
        center = (start+stop)/2
        span = (stop-start)
        if span*1e9 < self.MIN_SPAN:
            print('ERROR, span too small')
        
        return self.CenterSpan(center,span)
    
    def ConfigLevel(self,ref):
        """This function is best utilized when the device attenuation and gain 
        is set to automatic(default). When both attenuation and gain are set to 
        AUTO, the API uses the reference level to best choose the gain and 
        attenuation for maximum dynamic range. The API chooses attenuation and 
        gain values best for analyzing signal at or below the reference level. 
        For this reason, to achieve the best results, ensure gain and 
        attenuation are set to AUTO and your reference level is set at or 
        slightly about your expected input power for best sensitivity. 
        
        Reference level is specified in dBm units."""
        
        return self.API.saConfigLevel(self.device_id,self.ct.c_double (ref))
    '''
    def ConfigGain(self,atten,gain,preamp):
        """To set attenuation or gain to automatic, pass AUTO_GAIN and 
        AUTO_ATTEN as parameters. The preamp parameter is ignored when gain 
        and attenuation are automatic and is chosen automatically.
        
        PLEASE REFER TO THE MANUAL BEFORE USING THIS FUNCTION"""
        
        return self.API.saConfigGain(self.device_id,self.ct.c_int (atten),self.ct.c_int(gain),self.ct.c_bool(preamp))
    '''
    
    
    def BandWidth(self,rbw,vbw=None,reject=True):
        """The resolution bandwidth, or RBW, represents the bandwidth of 
        spectral energy represented in each frequency bin. For example, with 
        a RBW of 10 kHz, the amplitude value for each bin would represent 
        the total energy from 5 kHz below to 5 kHz above the bin’s center.
        The video bandwidth, or VBW, is applied after the signal has been 
        converted to frequency domain as power, voltage, or log units. It is 
        implemented as a simple rectangular window, averaging the amplitude 
        readings for each frequency bin over several overlapping FFTs. A signal 
        whose amplitude is modulated at a much higher frequency than the VBW 
        will be shown as an average, whereas amplitude modulation at 
        a lower frequency will be shown as a minimum and maximum value.   
        
        Available RBWs are [0.1Hz – 100kHz] and 250kHz. For the SA124 devices, 
        a 6MHz RBW is available as well. Not all RBWs will be available 
        depending on span, for example the API may restrict RBW when a 
        sweep size exceeds a certain amount. Also there are many hardware 
        limitations that restrict certain RBWs, for a full list of these 
        restrictions, see the manual.

        The parameter reject determines whether software image reject 
        will be performed.
        
        The SA-series spectrum analyzers do not have hardware based image 
        rejection, instead relying on a software algorithm to reject image 
        responses. See the manual for additional details. 
        
        Generally, set reject to true for continuous signals, and false to 
        catch short duration signals at a known frequency.
        To capture short duration signals with an unknown frequency, consider 
        the Signal Hound BB60C
        
        Parameters:
            - rbw, vbw in Hz. If vbw if None (def), it will be set equal to rbw
            - reject: True (def) or False
        """
        if reject is 0:
            reject=False
        if reject is 1:
            reject=True
        
        if type(reject) is not bool:
            print('Wrong reject value inserted')
            raise Exception('PARERR')
            
        if (rbw<self.MIN_RBW or rbw>100e3) and rbw!=250e3 and rbw != 6e6:
            print('Wrong rbw inserted')
            raise Exception
        
        if vbw is None:
            vbw = rbw
        
        
        return self.API.saConfigSweepCoupling(self.device_id,self.ct.c_double(rbw),self.ct.c_double(vbw),self.ct.c_bool(reject) )
        
        
    def ConfigRBWShape(self,rbwShape=1):
        """Specify the RBW filter shape, which is achieved by changing the 
        window function. When specifying RBW_SHAPE_FLATTOP, a custom 
        bandwidth flat-top window is used measured at the 3dB cutoff 
        point. When specifying RBW_SHAPE_CISPR, a Gaussian window with 
        zero-padding is used to achieve the specified RBW. The Gaussian 
        window is measured at the 6dB cutoff point"""
        
        if type(rbwShape) is str:
            if rbwShape.upper() == 'RBW_SHAPE_FLATTOP':
                rbwShape = 1
            elif rbwShape.upper() == 'RBW_SHAPE_CISPR':
                rbwShape = 2
            else:
                print('Wrong rbwShape inserted')
                raise Exception('PARERR')
        
        rbwShape = int(rbwShape)                        
        if rbwShape<1 or rbwShape>2:
            print('Wrong rbwShape inserted')
            raise Exception('PARERR')
        
        return self.API.saConfigRBWShape(self.device_id,self.ct.c_int(rbwShape))
    
    
    
    def ConfigProcUnits(self,units=0):
        """The units provided determines what unit type video processing occurs 
        in. 
        
        The chart below shows which unit types are used for each units 
        selection.
            
        For “average power” measurements, SA_POWER_UNITS should be selected. 
        For cleaning up an amplitude modulated signal, SA_VOLT_UNITS would be 
        a good choice. 
        
        To emulate a traditional spectrum analyzer, select SA_LOG_UNITS. 
        To minimize processing power and bypass video bandwidth processing, 
        select SA_BYPASS
        
        Parameters:
            - units: 0 or 'SA_LOG_UNITS' - dBm (def)
                     1 or 'SA_VOLT_UNITS' - mV
                     2 or 'SA_POWER_UNITS' - mW
                     3 or 'SA_BYPASS' - No processing
        """
        
        if type(units) is str:
            units=units.upper()
            if units == 'SA_LOG_UNITS':
                units = 0
            elif units == 'SA_VOLT_UNITS':
                units = 1
            elif units == 'SA_LOG_UNITS':
                units = 2
            elif units == 'SA_LOG_UNITS':
                units = 3
            else:
                print('Wrong units inserted'  )
                raise Exception('PARERR')
                
        units = int(units)
        if units<0 or units>3:
            print('Wrong units inserted'  )
            raise Exception('PARERR')
        
        return self.API.saConfigProcUnits(self.device_id,self.ct.c_int(units))
    
    def ConfigIQ(self,decimation,BW):
        """This function is used to configure the digital IQ data stream. A 
    decimation factor and filter bandwidth are able to be specified. 
    The decimation rate divides the IQ sample rate directly while the bandwidth 
    parameter further filters the digital stream. For any given decimation 
    rate, a minimum filter bandwidth must be applied to account for sufficient 
    filter roll off. If a bandwidth value is supplied above the maximum for a 
    given decimation, the bandwidth will be clamped to the maximum value. For 
    a list of possible decimation values and associated bandwidth values, see 
    the table below.
    
    The base sample rate of the SA44 and SA124 spectrum analyzers is 
    486.111111 (repeating) kS/s. To get a precise sample rate given a 
    decimation value, use this equation:

        Sample_rate = 486111.11111 / 2^decimation

    parameters: 
        - decimation: integer from 0 to 7
        - BW: a value of BW in Hz up to the maximum (function of the decimation),
        check the table class._IQ_BW
    """    

        decimation = int(decimation)
        if decimation <0 or decimation>7:
            print('Wrong decimation number inserted')
            raise Exception('PARERR')
            
        if BW > self._IQ_BW[decimation]:
            print('Warning: BW inserted is too large, max: '+str(self._IQ_BW[decimation]))
        
        return self.API.saConfigIQ(self.device_id,self.ct.c_int(2**decimation), self.ct.c_double(BW))
    
    def Reference(self,Reference=2):
        """Configure the time base reference port for the device. By passing a 
        value of SA_REF_INTERNAL_OUT you can output the internal 10MHz time 
        base of the device out on the reference port. By passing a value of 
        SA_REF_EXTERNAL_IN the API attempts to enable a 10MHz reference on the 
        reference BNC port. 
        
        If no reference is found, the device continues to use the internal 
        reference clock. 
        
        Once a device has successfully switched to an external reference it 
        must remain using it until the device is closed, and it is undefined 
        behavior to disconnect the reference input from the reference BNC port.
        
        Parameters:
            - Reference: 0 or SA_REF_UNUSED - to disable the port
                         1 or SA_REF_INTERNAL_OUT - for 10 MHz ref output
                         2 or SA_REF_EXTERNAL_IN - for 10MHz ref in (def)
        """
        
        if type(Reference) is str:
            Reference = Reference.upper()
            
            if Reference == 'SA_REF_UNUSED':
                Reference = 0
            elif Reference == 'SA_REF_INTERNAL_OUT':
                Reference = 1
            if Reference == 'SA_REF_EXTERNAL_IN':
                Reference = 2
            else:
                print('Wrong reference inserted')
                raise Exception('PARERR')
        
        Reference = int(Reference)
        if Reference<0 or Reference>2:
            print('Wrong reference inserted')
            raise Exception('PARERR')
            
        return self.API.saSetTimebase(self.device_id,self.ct.c_int(Reference))
                
        
#------------------------------------------------------------------------------------- Acquisition functions

    def Initiate(self,mode=0):
        """This function configures the device into a state determined by the 
        mode parameter. For more information regarding operating states, refer 
        to the Manual.

        This function calls Abort() before attempting to reconfigure. It
        should be noted, if an error occurs attempting to configure the device, 
        any past operating state will no longer be active and the device will 
        become idle.
        
        Parameters:
            - mode: -1 or 'IDLE' - standby the device
                    0 or 'SWEEP' (def) - performs a single sweep
                    1 or 'RT' - real time and continuous acquisition
                    2 or 'IQ' - IQ measurement
        """
        
        
        
        if type( mode) is str:
            mode = mode.upper()
            
            try:
                mode = self.modes_list.index(mode)-1
            except  ValueError:
                print('Wrong mode inserted')
                raise Exception('PARERR')
        
        mode = int(mode)
        if mode <-1 or mode >2:
                print('Wrong mode inserted')
                raise Exception('PARERR')
            
        self.mode = mode
        
        self.RTP = None
            
        return self.API.saInitiate(self.device_id,self.ct.c_int(mode),self.ct.c_int(0) )
    
    def Abort(self):
        """Stops the device operation and places the device into an idle state. 
        If the device is currently idle, then the function returns normally 
        and returns saNoError."""
        self.RTP = None
        return self.API.saAbort(self.device_id)
    
    def QuerySweepInfo(self):
        """This function should be called to determine sweep characteristics 
        after a device has been configured and initiated. 
        
        it returns [sweep length, start frequency, bins size]
        """
        
        sweepLength = self.ct.c_int(0)
        startFreq = self.ct.c_double(0)
        binSize = self.ct.c_double(0)
        
        status = self.API.saQuerySweepInfo(self.device_id,self.ct.byref(sweepLength),self.ct.byref(startFreq),self.ct.byref(binSize))
        if status != 0:
            print('An error occurred: '+str(status))
            raise Exception('DEVERR')
        
        return int(sweepLength.value),float(startFreq.value),float(binSize.value)
    
    def QueryIQInfo(self):
        """Use this function to get the parameters of the IQ data stream
        
            it returns: [number of IQ sample pairs,bandpass filter bandwidth (Hz), sample rate (samples/s)]
        """
        returnLen = self.ct.c_int(0)
        bandwidth = self.ct.c_double(0)    
        samplesPerSecond = self.ct.c_double(0)
    
        status = self.API.saQueryStreamInfo(self.device_id,self.ct.byref(returnLen),self.ct.byref(bandwidth), self.ct.byref(samplesPerSecond))
        if status != 0:
            print('An error occurred: '+str(status))
            raise Exception('DEVERR')
        
        return int(returnLen.value),float(bandwidth.value),float(samplesPerSecond.value)
    
    def QueryRealTime(self):
        """This function should be called after initializing the device for 
        Real Time mode. 
        
        It returns [frame width, frame height]
        """
        
        frameWidth = self.ct.c_int(0)
        frameHeight = self.ct.c_int(0)
        
        status = self.API.saQueryRealTimeFrameInfo(self.device_id,self.ct.byref(frameWidth),self.ct.byref(frameHeight))
        if status != 0:
            print('An error occurred: '+str(status))
            raise Exception('DEVERR')
        
        return int(frameWidth.value),int(frameHeight.value)
    
    def QueryRealTimePoi(self):
        """When this function returns successfully, the value poi points to 
        will contain the 100% probability of intercept duration in seconds of 
        the device as currently configured in real-time spectrum analysis. The 
        device must actively be configured and initialized in the real-time 
        spectrum analysis mode
        
        the pointer will be stored in class.RTP
        """
        poi=self.ct.c_double(0)
        
        status = self.API.saQueryRealTimePoi(self.device_id,self.ct.byref(poi))
        if status == 0:
            self.RTP = poi
        else:
            self.RTP = None
            print('An Error occurred: '+str(status))
            raise Exception('DEVERR')
        
    def GetSweep(self):
        """Upon returning successfully, this function returns:
            
        - 2 data modules with the minimum and maximum values of one full sweep. If the 
        detector is in MIN_MAX mode.
        
        - 1 data module with the average if the detector is in AVERAGE mode.
        
        """
        
        Length,start,bins_size = self.QuerySweepInfo()
        
        x = np.linspace(start,start+Length*bins_size,Length)
 
        mins = (self.ct.c_double*Length)(0)       
        maxs = (self.ct.c_double*Length)(0)       
        
        start = time.time()
        while(1):
            status = self.API.saGetSweep_64f(self.device_id,self.ct.byref(mins),self.ct.byref(maxs))
            if status != 0:
                if time.time()- start > 20:
                    print('TIMEOUT')
                    raise Exception('TIMEOUT')
                self.Initiate()
                
            if status != 0 and status != -11:
                print('An error occurred: '+str(status))
                raise Exception('DEVERR')
            
            if status == 0:
                break
            
        y1 = [mins[i] for i in range(Length)]
        y2 = [maxs[i] for i in range(Length)]
        
        if self.detector == 0:
            return dm.data_table([x / 1e9, y1], ['Frequency (GHz)', 'PSD (dB)']),\
                   dm.data_table([x / 1e9, y2], ['Frequency (GHz)', 'PSD (dB)'])
        elif self.detector == 1:
            return dm.data_table([x / 1e9, y1], ['Frequency (GHz)', 'PSD (dB)'])
        
        else:
            print("Configuration ERROR")
            raise Exception('CONFIGERROR')
            
    
    def GetPartialSweep(self):
        """This function is similar to the saGetSweep functions except it can 
        return in certain contexts before the full sweep is ready. This 
        function might return for sweeps which are going to take multiple 
        seconds, providing you with only a portion of the results. This might 
        be useful if you want to perform some signal analysis on portions of 
        the sweep as it is being received,or update other portions of your 
        application during a long acquisition process.
        
        Subsequent calls will always provide the next contiguous portion of 
        spectrum.

        It returns:
        [data module with the partial sweep, current_index]
        
        When the final portion of the sweep has been updated, current_index 
        will be equal the sweep length. 
        Calling this function again will request and begin the next sweep."""
        
        Length,start,bins_num = self.QuerySweepInfo()
        x = np.linspace(start,start+Length,Length)
        
 
        mins = (self.ct.c_double*Length)(0)       
        maxs = (self.ct.c_double*Length)(0)       
        start = self.ct.c_int(0)       
        stop = self.ct.c_int(0)       
        
        status = self.API.saGetPartialSweep_64f(self.device_id,self.ct.byref(mins), self.ct.byref(maxs),self.ct.byref(start),self.ct.byref(stop))
        if status != 0:
            print('An error occurred: '+str(status))
            raise Exception('DEVERR')
        
        start = int(start.value)
        stop = int(stop.value)-1
        
        y1 = [mins[i] for i in range(stop-start)]
        y2 = [maxs[i] for i in range(stop-start)]
        
        if self.mode == 0:
            return dm.data_table([x[start:stop], y1], ['Frequency (Hz)', 'PSD (dB)']),\
                   dm.data_table([x[start:stop], y2], ['Frequency (Hz)', 'PSD (dB)'])
        elif self.mode == 1:
            return dm.data_table([x[start:stop], y1], ['Frequency (Hz)', 'PSD (dB)'])
        
        else:
            print("UNKNOWN ERROR")
            raise Exception('UNKNOWNERROR')
            
    def GetRealTimeFrame(self):
        """This function is used to retrieve one real-time frame and sweep. 
        
        The frame should be WxH values long where W and H are the values 
        returned from saQueryRealTimeFrameInfo. 
        
        The function will return [frequency sweep, frame].
        
        For more information see the manual section."""

        Length,start,bins_num = self.QuerySweepInfo()
        W,H = self.QueryRealTime()
        #x = np.linspace(start,start+Length,Length)
        
        
        sweep = (self.ct.c_float*Length)(0)
        frame = (self.ct.c_float*(W*H))(0)
        
        status = self.API.saGetRealTimeFrame(self.device_id,self.ct.byref(sweep),self.ct.byref(frame) )
        if status != 0:
            print('An error occurred: '+str(status))
            raise Exception('DEVERR')
        
        
        sweep = [sweep[i] for i in range(Length)]
        frame = [frame[i] for i in range(W*H)]
        
        return sweep,frame
    
    def GetIQData(self,count,purge):
        """This function retrieves one block of IQ data.
        
        Parameters:
            - count: specifies the number of IQ data pairs to return.
            - purge: specify whether to discard any samples acquired by the API
            since the last time the GetIQData function was called. Set to 1 if 
            you wish to discard all previously acquired data, and 0 if you wish
            to retrieve the contiguous IQ values from a previous call to this 
            function.
        
        It returns:
        [I,Q,Remaining data to acquire, 1 if buffer was full and samples were 
        lost, seconds when the first sample was acquired, msec when the first 
        sample was acquired]
        """
        
        count = int(count)
        if count<0:
            print('wrong count inserted')
            raise Exception('PARERR')
        
        purge = int(purge)
        if purge<0 or purge>1:
            print('wrong purge value inserted')
            raise Exception('PARERR')
        
       
        
        iqData = (self.ct.c_float*(2*count))(0)
        dataRemaining = self.ct.c_int(0)
        sampleLoss =  self.ct.c_int(0)
        sec =  self.ct.c_int(0)
        milli  = self.ct.c_int(0)
        
        
        status = self.API.saGetIQDataUnpacked(self.device_id, self.ct.byref(iqData),self.ct.c_int(count), self.ct.c_int(purge),self.ct.byref(dataRemaining),self.ct.byref(sampleLoss),self.ct.byref(sec),self.ct.byref(milli))
        if status != 0:
            print('An error occurred: '+str(status))
            raise Exception('DEVERR')
        
        y = [iqData[i] for i in range(2*count)]
        
        return y[::2],y[1::2],int(dataRemaining.value),int(sampleLoss.value),int(sec.value),int(milli.value)
    
    
    def default_settings(self):
        self.ConfigAcquisition(1,0)
        self.StartStop(0,12)
        self.ConfigLevel(0)
        self.BandWidth(100e3)
        self.ConfigRBWShape()
        self.ConfigProcUnits(0)
        