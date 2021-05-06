# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:32:12 2014

@author: Oscar

wrapper class for the SDR14 library
v2.1.0 - OSC:
    - implemented multi-board connection when more boards are in the same chassis
    - renamed some functions to have less differences between this library 
    and the ADQ14 one

v2.0.2 - OSC:
    
    - bugfix in trigger edge setting
    
v2.0.1 - OSC:
    - bugfix in edge setting for ext trigger in dig class
    
v2.0.0 - OSC:
    - adapted to python dictionary

v1.1.2 - OSC:
inserted conv_factor to convert digits to V (it is a constant, must multiply digits for it)

v1.1.1 - OSC:
changed something that should give more compatibility

v1.1.0 - OSC:
modified the function used to setup the averaging mode

"""



class SDR14(object):
    import ctypes as ct
    import numpy as np
    version='2.1.0'
    
    def __init__(self,board_num = 1,OS='WINDOWS'):
        if OS.upper()=='LINUX':
            self.ADQAPI = self.ct.cdll.LoadLibrary("libadq.so")
        elif OS.upper()=='WINDOWS':
            self.ADQAPI = self.ct.cdll.LoadLibrary("ADQAPI.dll")
        else:
             print('Wrong OS inserted: {LINUX,WINDOWS}\n')
             return None

        self._PXITrig0=0
        self._PXITrig1=0

        #some c language configuration
        self.ADQAPI.CreateADQControlUnit.restype = self.ct.c_void_p
        self.ADQAPI.ADQ_GetRevision.restype = self.ct.c_void_p
        self.ADQAPI.ADQControlUnit_FindDevices.argtypes = [self.ct.c_void_p]
        self.ADQAPI.ADQ_GetPtrStream.restype = self.ct.c_void_p#self.ct.POINTER(self.ct.c_int16)
        self.Connect(board_num)
        
    
    def Connect(self,board_num=1):
        
        class stwrapper(self.ct.Structure):
            """ creates a struct to match emxArray_real_T """
        
            _fields_ = [('HWIFType', self.ct.c_int),
                        ('ProductID', self.ct.c_int),
                        ('VendorID', self.ct.c_uint),
                        ('AddressField1', self.ct.c_int),
                        ('AddressField2', self.ct.c_int),
                        ('DevFile', (self.ct.c_char*64)),
                        ('DeviceInterfaceOpened', self.ct.c_uint),
                        ('DeviceSetupCompleted', self.ct.c_uint)]
                
        self._sdr = self.ct.c_void_p(self.ADQAPI.CreateADQControlUnit())

        nof_devices = self.ct.c_int(0)
        e = stwrapper()
        
        success = self.ADQAPI.ADQControlUnit_ListDevices(self._sdr,self.ct.pointer(e),self.ct.pointer(nof_devices))
        nof_devices = nof_devices.value


        if nof_devices==0:
            print("WARNING: No device has been found")
            return 0
        
        
        board_num = int(board_num)
        if board_num <0:
            print('Error: board_num can\'t be negative!')
            return 0
        
        if board_num > nof_devices:
            print("Erro: Selected board number {}, but only {} boards have been found!".format(board_num,nof_devices))
            return 0
        
        success = success and self.ADQAPI.ADQControlUnit_OpenDeviceInterface(self._sdr,board_num-1)
        success = success and self.ADQAPI.ADQControlUnit_SetupDevice(self._sdr, board_num-1)
        
        self.devnum = 1
        self.id = 'SDR14' #to implement better
        
        # Print ADQAPI revision
        print('ADQAPI loaded, revision {:d}.'.format(self.ADQAPI.ADQAPI_GetRevision()))   
        rev=self.Print_adq_device_revisions()
        
        if not (rev>20e3 and rev<30e3):
            print('WARNING: FPGA revision is not in the correct range, are you sure you are using the right board?\n')
            
        
        self.Clock_ref() #ext ref activated by def
        
        self.dig = self._DIG(self.ADQAPI,self._sdr,self.devnum)
        self.awg = self._AWG(self.ADQAPI,self._sdr,self.devnum)
        
        #self.dig.gainandoffset(1,1,301) #ch1 offset fix (calibration)
        #self.dig.gainandoffset(2,1,361) #ch2 offset fix (calibration)
        
        return 1
    
    def Disconnect(self):
        self.ADQAPI.DeleteADQControlUnit(self._sdr)
        self._sdr= None
        self.dig = None

    def Print_adq_device_revisions(self):
        # Get revision info from ADQ
        rev = self.ADQAPI.ADQ_GetRevision(self._sdr, self.devnum)
        revision = self.ct.cast(rev,self.ct.POINTER(self.ct.c_int))
        print('\nConnected to ADQ #{:d}\n'.format(self.devnum))
        # Print revision information
        print('FPGA Revision: {}\n'.format(revision[0]))
        if (revision[1]):
            print('Local copy\n')
        else:
            print('SVN Managed\n')
            if (revision[2]):
                print('Mixed Revision\n')
            else :
                print('SVN Updated\n')
        
        return revision[0]

    def Blink(self):
        self.ADQAPI.ADQ_Blink(self._sdr,self.devnum)

        
    def reset(self,Type='COMM'):
        '''function reset([Type='COMM']):
        
        this function will reset the communication as default.
        
        reset('TOT') or reset(1) will reset the board at default settings'''
        
        if type(Type) is str:
            if Type.upper()=='COMM':
                return self.ADQAPI.ADQ_ResetDevice(self._sdr,self.devnum,8)
            elif Type.upper()=='TOT':
                return self.ADQAPI.ADQ_ResetDevice(self._sdr,self.devnum,2)
            else:
                print('Wrong Type inserted')
                return 0
        else:
            if Type==1:
                return self.ADQAPI.ADQ_ResetDevice(self._sdr,self.devnum,2)
            else:
                print('Wrong Type inserted')
                return 0

    def __typecheck(self,Type,Typelist,Min,Max):
    
        if type(Type) is str:
            try:
                num=Typelist.index(Type.upper() )+1
            except:
                print('Wrong Type inserted')
                return None
        else:
            num=int(Type)
            if num<Min or num>Max:
                print('Wrong Type number inserted')
                return None

        return num   

            
    def DIGSF(self):
        """Returns the digitizer sampling frequency in GSamples/s"""
        return 0.8 #GSamples/s

    def AWGSF(self):
        """Returns the AWG sampling frequency in GSamples/s"""
        return 1.6 #GSamples/s

    def Clock_ref(self,Type='EXTC'):
        ''' function clock_ref(self,Type='EXTC'):
        
        This function will set the clock reference, Type can be:
        
        - 'INT' or 1 - for the internal clock
        - 'EXT' or 2 - for an external clock
        - 'EXTC' or 3 (def) - for a 10 MHZ clock sync. signal
        '''
        
        ref_types=['INT','EXT','EXTC']
        
        num=self.__typecheck(Type,ref_types,0,2)-1
        if num is None:
            raise Exception(ValueError)
            
        if num==2:
            num=3
            
        if self.ADQAPI.ADQ_SetClockSource(self._sdr,self.devnum,num)==0:
            print('ERROR in setting the clock reference')
            raise Exception('ClockRef')





    
#----- DIGCL00 --------------------------------------------------------------------------------------------


    class _DIG(object):
        import ctypes as ct
        import numpy as np
        def __init__(self,library,board,devnum):
            
            self.__ADCSF= 0.8 #Gigasamples/s
            self.__conv_factor = 2.2/(2**16-1)
            self.__max_digfreq = 400 #MHz
            self.__max_digamp = 1.1 #Vp
            
            self._trig_type=['SW','EXT','LVL','INT','','PXI']
            self._trig_edge=['FALLING','RISING']
            self._acq_mode=['MR','AVE']
            
            
            self.__lib = library
            self.__board = board
            self.devnum = devnum
            
            self.pars_dict= {'TriggerMode':'SW',
                             'TriggerOptions': None,
                             'AcquisitionDelay': 0,
                             'AcquisitionMode': None,
                             'Records': 1,
                             'Samples': 1024,
                             'ChannelUsed': 3,
                             'Autoset':True
                             }
            
            
#----- DIGCL01 ----------------------------------------------------------------------- Checks and Exceptions            
        def __typecheck(self,Type,Typelist,Min,Max):
        
            if type(Type) is str:
                Type= Type.upper()
                try:
                    num=Typelist.index(Type)+Min
                except:
                    return None
            else:
                num=int(Type)
                if num<Min or num>Max:
                    return None
    
            return num      
        
        class __BASEEXC(Exception):
            pass
        
        class __DIGEXC(__BASEEXC):
            def __init__(self, expression, message):
                self.expression = expression
                self.message = message
#------- DIGCL02 ---------------------------------------------------------------------- Gain and offset
        def gainandoffset(self,Channel=1,Gain=None,Offset=None):
            """function GainOffset([Gain=None],[Offset=None])
            
            This function can be used to set the ADC Gain and Offset of the 
            specified Channel {1,2}. 
            If no argument is given, the value will be returned.
            Gain is [-30,30]
            """
            
            """because of a bug, gain and offsets for channel 1 must be executed on channel 1 and 2.
               gain and offsets for channel 2 must be executed on channel 3 and 4"""
               
            if Channel<1 or Channel>2:
                raise self.__DIGEXC('ChannelNum','Error: Wrong channel inserted')
                
    
            if Gain == None or Offset == None:
                    value_type = self.ct.c_int32*1
                    t1,t2 = value_type(0), value_type(0)
                    g,o= [],[]
                    if self.__lib.ADQ_GetGainAndOffset(self.__board,self.devnum,Channel,t1,t2)==0:
                        raise self.__DIGEXC('GainOffsetRead','Error in getting Gain and Offset values')
                    g,o = int(t1[0]),int(t2[0])
                    del t1,t2
                    
            
            loop_indexes = self.np.array([0,1])+2*(Channel-1)+1 #1 and 2 for Channel 1, 3 and 4 for Channel 2
    
            if Gain != None:
                if Gain<-30 or Gain>30:
                    raise self.__DIGEXC("GainValue","Error, wrong gain: -30 < Gain < 30")
                
                Gain= int(Gain*1024)
                
                if Offset != None:
                    for i in loop_indexes:
                            if self.__lib.ADQ_SetGainAndOffset(self.__board,self.devnum,int(i),Gain,Offset)==0:
                                raise self.__DIGEXC('GainOffsetWrite','Error in setting Gain and Offset values')
                
                else:
                    for i in loop_indexes:
                        if self.__lib.ADQ_SetGainAndOffset(self.__board,self.devnum,int(i),Gain,o)==0: #Offset is unchanged
                                raise self.__DIGEXC('GainOffsetWrite','Error in setting Gain and Offset values')
            else:
                if Offset is None:
                    return g,o
                else:
                    for i in loop_indexes:
                        if self.__lib.ADQ_SetGainAndOffset(self.__board,self.devnum,Channel,g,Offset)==0: #gain is unchanged
                                raise self.__DIGEXC('GainOffsetWrite','Error in setting Gain and Offset values')

#----- DICCL03 ------------------------------------------------------------------ parameters functions ------------
        
        def autoset(self,arg=None):
            if arg is None:
                return self.pars_dict['Autoset']
            
            if arg is True or arg is 1:
                self.pars_dict['Autoset'] = True
            elif arg is False or arg is 0:
                self.pars_dict['Autoset'] = False
            else:
                raise self.__DIGEXC('Autoset','Wrong value inserted: {}'.format(arg)+'/{True,False}')
        
        def trigger_mode(self,Mode=None,arg=None):
            '''function trigger_mode(Type='SFT'):
            
            This function is used to set the trigger mode, Type can be:
            
            - 'SW' or  1 for software controlled trigger
            - 'EXT' or 2 for external trigger 1
                - further options: edge
                    - 0 for falling edge
                    - 1 for rising edge (def)
                    
            - 'LVL' or 3 for level trigger mode
                - further option: [channel number to use as a trigger,edge,level]
                    - chn : 1 or 2 (def)
                    - edge: 0 for falling, 1 for rising (def)
                    - level: threshold level in digits (def 5958 = 0.2V)
                    
            - 'INT' or 4 for internal trigger
                -further option: Period of the internal trigger in s (1e-3 def)
                
            - 'PXI' or 6 for the PXIe trigger (use the proper function to enable it)'''
            
            
                
            if Mode is None: #Query
                tmp= self._trig_type[int(self.__lib.ADQ_GetTriggerMode(self.__board,self.devnum)-1)]
                if self.autoset:
                    if tmp != self.pars_dict['TriggerMode']:
                        raise self.__DIGEXC('TriggerMode','TriggerMode mismatch!')
                    
                    if tmp == 'EXT' or tmp == 'INT' or tmp == 'LVL':
                        return tmp,self.pars_dict['TriggerOptions']
                    else:
                        return tmp
            
            if Mode == '':
                raise self.__DIGEXC('TriggerMode','Mode must be specified')
                
            Mode = self.__typecheck(Mode,self._trig_type,1,6)
            if Mode is None:
                raise self.__DIGEXC('TriggerMode','Wrong mode inserted, check .print_trigger_modes() for help: {}'.format(Mode))
            
            self.pars_dict['TriggerMode']= self._trig_type[Mode-1]
            
            if Mode==2: #EXT
                if arg is None:
                    arg = 'RISING'
                Edge = self.__typecheck(arg,self._trig_edge,0,1)
                if Edge is None:
                    raise self.__DIGEXC('TriggerEdge','Wrong Edge inserted, check .print_trigger_edges() for help: {}'.format(Edge))
                self.pars_dict['TriggerOptions'] = self._trig_edge[Edge]
            
            if Mode == 4: #LVL
                if arg is None:
                    arg = 1e-3
                    
                if arg<0 or 1/arg > self.__ADCSF*1e9:
                    raise self.__DIGEXC('InternalTriggerPeriod','Frequency must be smaller than 800 MHz')
                
                arg = self.np.floor(arg*self.__ADCSF*1e9)/(self.__ADCSF*1e9)
                
                self.pars_dict['TriggerOptions'] = arg
            
            
            if Mode == 3: #INT
                if arg is None:
                    arg = [2,1,5958]
                ch = int(arg[0])
                if ch<1 or ch>2:
                    raise self.__DIGEXC('CHNERR','Wrong channel inserted: {}'.format(ch)+'/{1,2}' )
                
                edge = self.__typecheck(arg[1],self._trig_edge,0,1)
                if edge is None:
                    raise self.__DIGEXC('TriggerEdge','Wrong Edge inserted, check .print_trigger_edges() for help: {}'.format(edge))
                
                lvl = int(arg[2])
                if lvl<=-2**15 or lvl >=2**15:
                    raise self.__DIGEXC('LVLERR','Wrong level inserted: {}/[-2**15,2**15]'.format(lvl))
                
                self.pars_dict['TriggerOptions'] = [ch,edge,lvl]
                
            if self.autoset():
                self.__set_triggermode()
                
        
        def trigdelay(self,arg=None,mode='t'):
            """ 
            This function will set the delay between the trigger and the begin of acquisition, in ns.
            
            -args
                - arg can be the delay time in ns, or in samples, if it is None (def) it will be queried
                - mode can be 't' for time or 's' for samples
            
            NOTE: arg will be modified so that Time*ADCSF will be round at the next upper integer
            
            """
            
            
            
            mode_types = ['T','S']
            mode=self.__typecheck(mode,mode_types,0,1)
            
            if mode is None:
                raise self.__DIGEXC('MODEERR','Wrong mode inserted: {"T","S"}')
            
            if arg is None:
                if mode == 0:
                    return self.pars_dict['AcquisitionDelay']/self.__ADCSF
                else:
                    return self.pars_dict['AcquisitionDelay']
                
            if mode: #samples
                ns = int(arg)
            else: #time
                if arg<0:
                    raise self.__DIGEXC('NegTime','Time cannot be negative')
            
                ns = self.np.floor(arg*self.__ADCSF)
                
            
            if ns>2**31:
                raise self.__DIGEXC('LargeDelayTime','Time delay too large, max: '+ str(2**31/self.__ADCSF))
            
            
            self.pars_dict['AcquisitionDelay'] = int(ns)
            if self.autoset():
                self.__set_trigdelay()
            
        def acqmode(self,Records=1,Samples=1024,Mode=None,Hold=None):
            """
            This function sets the acquisition mode to multirecord or averages.
            
            args:
                - Record (def 1), is the number of tracks to record
                - Samples (def 1024): is the number of points to acquire, suggested a power of 2, the limit is a function of the Records
                - Mode: 'MR' or 1 for multirecord, 'AVE' or 2 for averages
                - Hold: number of samples from the trigger to wait before acquisition
            
            If Mode is None (def) it will be queried.
            """
            
            if Mode is None:
                return self.pars_dict['AcquisitionMode'],self.pars_dict['Records'],self.pars_dict['Samples'],self.trigdelay(mode='s')
            
            Mode = self.__typecheck(Mode,self._acq_mode,1,2)
            if Mode is None:
                raise self.__DIGEXC('ACQMOD','Wrong mode inserted, check print_acqmodes_list() for help: {}'.format(Mode))
            
            self.pars_dict['AcquisitionMode'] = self._acq_mode[int(Mode-1)]
            
            Records = int(Records)
            if Records <1:
                raise self.__DIGEXC('RECERR','Records can\'t be less than 1: {}'.format(Records))
            
            #Evaluates the maximum number of samples in function of the records
            Samples = int(Samples)
            sam_limit = self.ct.c_uint()
            
            self.__lib.ADQ_GetMaxNofSamplesFromNofRecords (self.__board,self.devnum,Records,self.ct.byref( sam_limit))
            
            sam_limit = int(sam_limit.value)
            if Samples >= sam_limit:
                raise self.__DIGEXC('SAMERR','Samples must be in the range: [16,{}]'.format(sam_limit))
            
            
            self.pars_dict['Records'],self.pars_dict['Samples'] = Records,Samples
            
            if Hold is not None:
                self.trigdelay(int(Hold),'s')
                
            if self.autoset():
                self.__set_acqmode()
        
        
        def channel_used(self,arg=None):
            """This function sets the channel used for the acquisition:
                
                arg can be:
                    1 - for channel 1
                    2 - for channel 2
                    3 - for both
                    None - query (def)
                    
                NOTE: this boards always acquires in parallel, using 1 or 2 discards
                the data of the other channel.
                
                """

            if arg is None:
                return self.pars_dict['ChannelUsed']
            
            arg = int(arg)
            if arg<1 or arg>3:
                raise self.__DIGEXC('CHERR','Wrong channel inserted: {}/[1,3]'.format(arg))
                
            self.pars_dict['ChannelUsed']=arg
            
        def _mr_close(self):
            if self.__lib.ADQ_MultiRecordClose(self.__board,self.devnum)==0:
                print('ERROR in setting the singlerecord')
                raise Exception('SingleRecord')
            
            
        
    

            
#------ DIGCL04     ----------------------------------------------------------------------------- Set functions
        def __set_triggermode(self):
            Mode = int(self._trig_type.index(self.pars_dict['TriggerMode'])+1)
            if self.__lib.ADQ_SetTriggerMode(self.__board,self.devnum,Mode)==0:
                raise self.__DIGEXC('TrigMode','ERROR in setting the trigger mode:{}'.format(Mode))
                
            
            if Mode==2: #EXT
                Edge = int(self._trig_edge.index(self.pars_dict['TriggerOptions']))
                if self.__lib.ADQ_SetExternTrigEdge(self.__board,self.devnum,Edge) == 0:
                    raise self.__DIGEXC('TrigEdge','ERROR in setting the external trigger edge:{}'.format(Edge))
            
            if Mode==3: #LVL 
                self.__set_trigger_channel(self.pars_dict['TriggerOptions'][0])
                Edge = int(self._trig_edge.index(self.pars_dict['TriggerOptions'][1]))
                if self.__lib.ADQ_SetLvlTrigEdge(self.__board,self.devnum,Edge) == 0:
                    raise self.__DIGEXC('TrigEdge','ERROR in setting the level trigger edge:{}'.format(Edge))
                lvl = self.pars_dict['TriggerOptions'][2]
                if self.__lib.ADQ_SetLvlTrigLevel(self.__board,self.devnum,lvl) == 0:
                    raise self.__DIGEXC('TrigLevel','ERROR in setting the level trigger level:{}'.format(lvl))
            else:
                self.__set_trigger_channel(0) #Internal/external trigger
                
            if Mode==4: #INT
                per = self.pars_dict['TriggerOptions']
                per = int(per*self.__ADCSF*1e9/4)
                if self.__lib.ADQ_SetInternalTriggerPeriod(self.__board,self.devnum,per)==0:
                    raise self.__DIGEXC('IntTrigPer','Error in setting the internal trigger period: {}'.format(per))

        def __set_trigger_channel(self,Channel=0):
            '''Function trigger_channel([Channel=None]):
            
            This function is used to set the specified channel as a trigger:
            
            Channel can be:
            - 0  (no channel, internal trigger)
            - 1
            - 2
            
            '''
            if self.__lib.ADQ_SetLvlTrigChannel(self.__board,self.devnum,Channel) == 0:
                    raise self.__DIGEXC('TrigChan','ERROR in setting the channel as a trigger')

        def __set_trigdelay(self):
            ns = int(self.np.floor(self.pars_dict['AcquisitionDelay']*self.__ADCSF))
            
            if self.__lib.ADQ_SetTriggerHoldOffSamples(self.__board,self.devnum,ns) == 0:
                raise self.__DIGEXC('TRIGDELAY', 'Error in setting the trigger delay')

        def __set_acqmode(self):
            if self.pars_dict['AcquisitionMode'] == 'MR':
                self.__multirecord()
            elif self.pars_dict['AcquisitionMode'] == 'AVE':
                self.__averaging()
            else:
                raise self.__DIGEXC('ACQERR','Set acquisition mode first')

        def __multirecord(self):
            self.__set_trigdelay()
            
            if self.__lib.ADQ_SetPreTrigSamples(self.__board,self.devnum,0)==0:
                raise self.__DIGEXC('MultiRecord','ERROR in setting the pre-trigger')
            
            if self.__lib.ADQ_MultiRecordSetup(self.__board,self.devnum,self.pars_dict['Records'],self.pars_dict['Samples'])==0:
                raise self.__DIGEXC('MultiRecord', 'ERROR in setting the multirecord')

        
        def __averaging(self,Flags=0x44):
            """NOTE: if you use the averaging, don't use multirecord
            NOTE: Arm and disarm averaging with averaging_arm(1) and not with trigarm()
            NOTE: use averaging_status and average_collect to monitor the averages and take the data
            """
            
            if self.__lib.ADQ_MultiRecordClose(self.__board,self.devnum)==0:
                raise self.__DIGEXC('MRERR',"Error in closing the multirecord")
            if self.__lib.ADQ_WaveformAveragingSetup(self.__board,self.devnum,self.pars_dict['Records'],self.pars_dict['Samples'],0,self.trigdelay(mode='s'),int(Flags)) == 0:
                raise self.__DIGEXC('ErrAve',"Error in setting up the averages")


                    
#------ DICCL05 ----------------------------------------------------------------------------- Acquisition functions
        def acquisition_ended(self):
            if self.pars_dict['AcquisitionMode']=='MR':
                return self.__lib.ADQ_GetAcquiredAll(self.__board,self.devnum)
            elif self.pars_dict['AcquisitionMode']=='AVE':
                return self.averaging_status()[0]
            else:
                raise self.__DIGEXC('ACQERR','Set acquisition mode first')
        
        def averaging_status(self):
            """
            
            This function will return a tuple with the status of the averaging:
            
            - ended = 1 means that the acquisition is completed
            - acquired return the number of records acquired
            - idle means that the board is waiting
            """
            """        
            ended= self.ct.POINTER(self.ct.c_ubyte)*1
            idle= self.ct.POINTER(self.ct.c_ubyte)*1
            acquired= self.ct.POINTER(self.ct.c_uint32)*1
            """
            
            if self.pars_dict['AcquisitionMode'] != 'AVE':
                raise self.__DIGEXC('MODEERR','This function works only in AVE mode')
            
            ended= self.ct.c_ubyte()
            idle= self.ct.c_ubyte()
            acquired= self.ct.c_uint32()
    
            if self.__lib.ADQ_WaveformAveragingGetStatus(self.__board,self.devnum,self.ct.byref(ended),self.ct.byref(acquired),self.ct.byref(idle))==0:
                raise self.__DIGEXC('AveStatus','Error in getting the averaging status')
    
            """        
            ended= self.ct.cast(ended,self.ct.POINTER(self.ct.c_short))[0]  
            acquired= self.ct.cast(acquired,self.ct.POINTER(self.ct.c_short))[0]  
            idle= self.ct.cast(idle,self.ct.POINTER(self.ct.c_short))[0]  
            """
            
            return int(ended.value>>1),int(acquired.value),int(idle.value)
        
        def trigarm(self,Mode='ON'):
            '''This function arms (def) or disarms the trigger:
                
                - 0 or "OFF" to disarm
                - 1 or "ON" to arm
            '''
            
            modtype=['OFF','ON']
            mt=self.__typecheck(Mode,modtype,0,1)
            
            if self.pars_dict['AcquisitionMode'] == 'MR':
                if mt==0:
                    if self.__lib.ADQ_DisarmTrigger(self.__board,self.devnum) ==0:
                        raise self.__DIGEXC('TrigDisarm','ERROR in disarming the MR trigger')
                else:
                    if self.__lib.ADQ_ArmTrigger(self.__board,self.devnum)==0:
                        raise self.__DIGEXC('TrigDisarm','ERROR in arming the MR trigger')
            elif self.pars_dict['AcquisitionMode'] == 'AVE':
                if mt==0:
                    if self.__lib.ADQ_WaveformAveragingDisarm(self.__board,self.devnum)==0:
                        raise self.__DIGEXC('TrigDisarm','ERROR in disarming the AVE trigger')
                else:
                    if self.__lib.ADQ_WaveformAveragingArm(self.__board,self.devnum)==0:
                        raise self.__DIGEXC('TrigDisarm','ERROR in arming the AVE trigger')
            else:
                raise self.__DIGEXC('ACQERR','Set acquisition mode first')
                
        def trigwaiting(self):
            '''Returns 1 if the board is waiting for a trigger, else 0.'''
            return self.__lib.ADQ_GetWaitingForTrigger(self.__board,self.devnum)

        def trigger(self):
            '''function trigger():
            
            This function will software-trigger the board'''
            
            if self.__lib.ADQ_SWTrig(self.__board,self.devnum) == 0:
                print('Error in triggering the device')
                raise Exception('TrigEXC')

        def acquire_data(self):
            """
            This function is used to collect the data from the board's memory:
            """
           
            if self.pars_dict['AcquisitionMode'] == 'MR':
               return self.__mr_acquire()
            elif self.pars_dict['AcquisitionMode'] == 'AVE':
               return self.__ave_acquire()
            else:
                raise self.__DIGEXC('ACQERR','Set acquisition mode first')
        
        def __mr_acquire(self):
           Samples = self.pars_dict['Samples']
           Records = self.pars_dict['Records']
        
           tot=Samples*Records
           
           col=self.ct.c_void_p*tot
           row=self.ct.POINTER(self.ct.c_void_p)*2 #2 Channels
           buffer=row()
           
           Channel= self.pars_dict['ChannelUsed']
           
           buffer[0],buffer[1]=col(0),col(0)
           
           if Channel==3:
                if self.__lib.ADQ_GetData(self.__board,self.devnum,buffer,tot,2,0,Records,0x3,0,Samples,0)==0:
                    raise self.__DIGEXC('GETDATA',"Error in reading the data")
                
                z1,z2 = self.np.core.multiarray.fromiter(self.ct.cast(buffer[0],self.ct.POINTER(self.ct.c_short)) ,dtype= self.np.int16,count=tot), self.np.core.multiarray.fromiter(self.ct.cast(buffer[1],self.ct.POINTER(self.ct.c_short)) ,dtype=self.np.int16,count=tot)
                
                return self.np.reshape(z1,(Records,Samples)), self.np.reshape(z2,(Records,Samples))
                
           else:
                if self.__lib.ADQ_GetData(self.__board,self.devnum,buffer,tot,2,0,Records,Channel,0,Samples,0)==0:
                    raise self.__DIGEXC('GETDATA',"Error in reading the data")
                
                z1= self.np.fromiter(self.ct.cast(buffer[Channel-1],self.ct.POINTER(self.ct.c_short)) ,dtype=self.np.int16,count=tot)
                return self.np.reshape(z1,(Records,Samples))
            
            
            
            
        def __ave_acquire(self):
            """Function averages_acquire(Samples,[Channel=1]):
            
            This function will acquire the number of Samples specified by the memory of the board that stores the averaging result.
            
            - Channel can be 1,2 or 3 for both
            - Check: will check if the averaging is still in progress, disable it if auto-rearm is on (def True)
            """
            Check=True
            
            Channel = self.pars_dict['ChannelUsed']
            
            if Check is True:        
                if self.averaging_status()[0]==0:
                    print('Averaging still in progress')
                    return 0
            
            Samples = self.pars_dict['Samples']
            #Records = self.pars_dict['Records']
            
            buffer = self.ct.c_int*int(Samples*2)
            buffer=buffer()
            
            if self.__lib.ADQ_WaveformAveragingGetWaveform(self.__board,self.devnum,buffer)==0:
                raise self.__DIGEXC('GetData','Error in getting the data')
            
    
            target_data=self.ct.POINTER(self.ct.c_int)*2
            target_data=target_data()
            
            buf=self.ct.c_int*Samples
            
            target_data[0],target_data[1] = buf(0), buf(0)
            
    
            if self.__lib.ADQ_WaveformAveragingParseDataStream(self.__board,self.devnum,Samples,buffer,target_data)==0:
                raise self.__DIGEXC('GetData','Error in parsing the data')
            
            if Channel==3:
                return self.np.core.multiarray.fromiter(target_data[0],self.np.int32,Samples),self.np.core.multiarray.fromiter(target_data[1],self.np.int32,Samples)
            else:
                return self.np.core.multiarray.fromiter(target_data[Channel-1],self.np.int32,Samples)
            
            
#----- AWGCL00 --------------------------------------------------------------------------------------------


    class _AWG(object):
        import ctypes as ct
        import numpy as np
        def __init__(self,library,board,devnum):
            
            self.__DACSF = 1.6 #Gigasamples/s
            self.__max_awgfreq = 330 #MHz
            self.__max_awgamp = 0.6 #V
            self.__offmax = 0.5 #V
            self.__lib = library
            self.__board = board
            self.devnum =  devnum
            
            self.__min_seg_length = 1000
            self.__max_seg_length = 125e6
            self.__max_queue_length = 1024
            self.__max_reps = 2**31-1
            
            self.__trig_type= ['SW','EXT','PXI','INT']
            
            self.pars_dict = {'Bypass': False,
                              'Autoset':True}
            
            tmp = {'Offset': 0,
                                  'TriggerMode':None,
                                  'Queue': None,
                                  'Autorearm' : False,
                                  'Continuous': False
                             
                             }
            
            self.__set_bypass() #Safety
            
            self.channels_pars_dict = []
            for i in range(2):
                self.channels_pars_dict.append(tmp.copy())
                self.channels_pars_dict[i]['Queue'] = []
            del tmp
        
  
            
#----- AWGCL01 ----------------------------------------------------------------------- Checks and Exceptions            
        def __typecheck(self,Type,Typelist,Min,Max):
        
            if type(Type) is str:
                Type= Type.upper()
                try:
                    num=Typelist.index(Type)+Min
                except:
                    return None
            else:
                num=int(Type)
                if num<Min or num>Max:
                    return None
    
            return num      
        
        def __check_ch_num(self,number):
            number = int(number)
            if number<1 or number>3:
                raise self.__AWGEXC('CHNERR','Wrong channel number inserted: {1,2,3}')
            
            return number
                
        class __BASEEXC(Exception):
            pass
        
        class __AWGEXC(__BASEEXC):
            def __init__(self, expression, message):
                self.expression = expression
                self.message = message
            


#----- AWGCL03 ------------------------------------------------------------------ parameters functions ------------
        
        def autoset(self,arg=None):
            if arg is None:
                return self.pars_dict['Autoset']
            
            if arg is True or arg is 1:
                self.pars_dict['Autoset'] = True
            elif arg is False or arg is 0:
                self.pars_dict['Autoset'] = False
            else:
                raise self.__AWGEXC('Autoset','Wrong value inserted: {}'.format(arg)+'/{True,False}')

        def bypass(self,arg=None):
            if arg is None:
                return self.pars_dict['Bypass']
            else:
                if arg == 0 or arg == False:
                    self.pars_dict['Bypass'] = False
                elif arg==1 or arg == True:
                    self.pars_dict['Bypass']=True
                else:
                    raise self.__AWGEXC('BYPASS','Wrong arg inserted: {True,False}')
            
            self.__set_bypass()
                
        def offset(self,Voltage=None,Channel=3): #to check limits
            """This function will set the DC output voltage of the specified channel, 3 for both"""
            
            Channel=self.__check_ch_num(Channel)
            
            if Voltage is None: #NO HW query
                if Channel==3:
                    return self.channels_pars_dict[0]['Offset'],self.channels_pars_dict[1]['Offset']
                else:
                    return self.channels_pars_dict[Channel-1]['Offset']
            
            if Voltage < -self.__offmax or Voltage > self.__offmax:
                raise self.__AWGEXC('RANGEERR','Error: voltage out of range [-1,1] V')
            
            if Channel == 3:
                for i in range(len(self.channels_pars_dict)):
                    self.channels_pars_dict[i]['Offset']= Voltage
                
            else:
                self.channels_pars_dict[Channel-1]['Offset']= Voltage
    
            if self.autoset():
                self.__set_offset(Channel)

        def trigger_mode(self,Mode=None,Channel=3):
            ''' Function awg_trigger(self,Dacid,Type=''):
            
            This function is used to set the trigger mode of the AWG:
            
            - dacid can be 1 for TI channel, 2 for TQ channel or 3 for both
            - Type can be:
                - 'SW' or 1 for software (def)
                - 'EXT' or 2 for external trigger
                - 'PXI' or 3 for the chassis trigger
                - 'INT' or 4 for internal trigger '''
                
            
            
            Channel = self.__check_ch_num(Channel)
            
            if Mode is None:
                if Channel == 3:
                    return self.channels_pars_dict[0]['TriggerMode'],self.channels_pars_dict[1]['TriggerMode']
                else:
                    return self.channels_pars_dict[Channel-1]['TriggerMode']
            
            trt=self.__typecheck(Mode,self.__trig_type ,1,4)
            if trt is None:
                raise self.__AWGEXC('TRERR','Wrong trigger type inserted, check the print_trigger_types() function for help')
                
            
            
            if Channel == 3:
                for i in range(2):
                    self.channels_pars_dict[i]['TriggerMode'] = self.__trig_type[trt-1]
            else:
                self.channels_pars_dict[Channel-1]['TriggerMode'] = self.__trig_type[trt-1]
    
            if self.autoset():
                self.__set_trigger(Channel)
        
        def autorearm(self,arg=None,Channel=3):
            """The AWG will autorearm when this value is true, the user has to arm the first time."""
            
            Channel = self.__check_ch_num(Channel)
            
            if arg is None:
                return self.channels_pars_dict[0]['Autorearm'],self.channels_pars_dict[1]['Autorearm'],
            
            if arg == 0 or arg == False:
                    ar = False
            elif arg == 1 or arg == True:
                    ar = True
            else:
                raise self.__AWGEXC('VALERR','Wrong Autorearm value inserted: {False,True}')
            
            if Channel==3:
                for i in range(2):
                    self.channels_pars_dict[i]['Autorearm']=ar
            else:
                self.channels_pars_dict[Channel-1]['Autorearm']=ar
                
            if self.autoset():
                if Channel==3:
                    for i in range(2):
                        self.__set_autorearm(i+1)
                else:
                    self.__set_autorearm(Channel)

        def continuous(self,arg=None,Channel=3):
            """The AWG will reproduce the sequence continuously when this value is true."""
            
            Channel = self.__check_ch_num(Channel)
            
            if arg is None:
                return self.channels_pars_dict[0]['Continuous'],self.channels_pars_dict[1]['Continuous'],
            
            if arg == 0 or arg == False:
                    ar = False
            elif arg == 1 or arg == True:
                    ar = True
            else:
                raise self.__AWGEXC('VALERR','Wrong Continuous value inserted: {False,True}')
            
            if Channel==3:
                for i in range(2):
                    self.channels_pars_dict[i]['Continuous']=ar
            else:
                self.channels_pars_dict[Channel-1]['Continuous']=ar
                
            if self.autoset():
                if Channel==3:
                    for i in range(2):
                        self.__set_continuous(i+1)
                else:
                    self.__set_continuous(Channel)
        
        def clear_queue(self,Channel=3):
            """Clear the queue of the specified channel, 3 for both"""
            
            Channel = self.__check_ch_num(Channel)

            if Channel==3:
                for i in range(2):
                    self.channels_pars_dict[i]['Queue'] = []
            else:
                self.channels_pars_dict[Channel-1]['Queue'] = []
            
            
                        
                
        
        def insert_array_in_queue(self,array,ID=None,Channel=1,Repetitions=1,Amplitude='MAX'):
            """This function inserts a pulse (object or array) in the Queue of the specified Channel:
                
                args:
                    - pulse can be a PULSE object or an array/list of normalized numbers [-1:1]
                    - ID: Wave ID, if it is None (def), it will be automatically inserted
                    - Repetitions: how many times this pulse is repeated (def 1)
                    - Amplitude: can reduce the max amplitude: 'MAX' is equal to 2**13-1 = 2.2V (def)
            """
            
            Channel= self.__check_ch_num(Channel)
            
            
            for i in array:
                if i<-1 or i>1:
                    raise self.__AWGEXC('PULSEERR','Pulse is not correctly normalized: [-1,1]')
            
            
            
            if Channel==3:
                for i in range(2):
                    if ID is None:
                        ID = 'p{}'.format(len(self.channels_pars_dict[i]['Queue']))
                    
                    ID = ID+'-c{}'.format(i+1)
                    tmp = {'Pulse': array,'ID':ID,'Repetitions': Repetitions,'Amplitude':'MAX'}
                
                    self.channels_pars_dict[i]['Queue'].append(tmp)
            else:
                if ID is None:
                        ID = 'p{}'.format(len(self.channels_pars_dict[Channel-1]['Queue']))
                    
                ID = ID+'-c{}'.format(Channel)
                tmp = {'Pulse': array,'ID':ID,'Repetitions': Repetitions,'Amplitude':'MAX'}
                self.channels_pars_dict[Channel-1]['Queue'].append(tmp)
            
        
#------ AWGCL04     ----------------------------------------------------------------------------- Set functions
        def __set_bypass(self):
            if self.bypass():
                self.__lib.ADQ_WriteRegister(self.__board, 1, 10240, 0, 2)
            else:
                self.__lib.ADQ_WriteRegister(self.__board, 1, 10240, 0, 0)
                
        def __set_offset(self,Channel):
            
            if Channel==3:
                Voltage1,Voltage2 = self.offset(Channel=3)
                Voltage1 = self.ct.c_float(Voltage1*2)
                Voltage2 = self.ct.c_float(Voltage2*2)
                a = self.__lib.ADQ_SetDACOffsetVoltage(self.__board,self.devnum,1,Voltage1)
                a += self.__lib.ADQ_SetDACOffsetVoltage(self.__board,self.devnum,2,Voltage2)
                if a != 2:
                    raise Exception('AWGOFFSET','Error in setting the AWG offset')
            else:
                Voltage = self.offset(Channel=Channel)
                Voltage = self.ct.c_float(Voltage*2)
                if self.__lib.ADQ_SetDACOffsetVoltage(self.__board,self.devnum,Channel,Voltage)==0:
                    raise Exception('AWGOFFSET','Error in setting the AWG offset')
                    
        def __set_trigger(self,Channel):
            
            if Channel==3:
                trt1,trt2 = self.trigger_mode(Channel=3)
                if trt1 is None or trt2 is None:
                    raise Exception('AWGTRTYPE','Choose a trigger first')
                
                trt1 = self.__trig_type.index(trt1)
                trt2 = self.__trig_type.index(trt2)
                
                a = self.__lib.ADQ_AWGSetTriggerEnable(self.__board,self.devnum,1,1<<trt1)
                a += self.__lib.ADQ_AWGSetTriggerEnable(self.__board,self.devnum,2,1<<trt2)
                if a != 2:
                    raise Exception('AWGTRTYPE','Error in setting the AWG Trigger')
            else:
                trt = self.trigger_mode(Channel=Channel)
                if trt is None:
                    raise Exception('AWGTRTYPE','Choose a trigger first')
                trt = self.__trig_type.index(trt)
                if self.__lib.ADQ_AWGSetTriggerEnable(self.__board,self.devnum,Channel,1<<trt)==0:
                    raise Exception('AWGTRTYPE','Error in setting the AWG Trigger')

        def set_queue(self,Channel):
            """This function push the queue in the AWG memory"""
            
            Channel = self.__check_ch_num(Channel)
            if Channel==3:
                raise self.__AWGEXC('CHN','Channel = 3 is not allowed in this function')
            
            queue = self.channels_pars_dict[Channel-1]['Queue']
            for i in range(len(queue)):
                element = queue[i]
            
                y= element['Pulse']
                Amplitude = element['Amplitude']
                if Amplitude == 'MAX':
                    Amplitude = 2**13-1
                    
                Segid = element['ID']
                Repetitions = element['Repetitions']
                
                y= self.np.floor(Amplitude*y)
            
                yn= self.np.array(self.np.floor(self.np.where(y<0,y+2**14,y)),dtype=self.np.uint16)
                del y
        
                # Validate data
                if max(yn)>(2**14)-1:
                     raise self.__AWGEXC('SEGAMP', 'Values in pulse {} are bigger than 14-bits!!'.format( Segid))
              
                if (len(yn)%16) > 0:
                     k=len(yn)-1- (len(yn)%16)
                     print('WARNING: Section %d has %d samples, which is not multiple of 16. The data will be truncated to %d samples.'% k, len(yn),len(yn)-(len(yn)%16))
                     yn=yn[1:k]
            
        
                usedtype=self.ct.c_uint32
                
                
                if self.__lib.ADQ_AWGSegmentMalloc(self.__board,self.devnum,Channel, i+1, len(yn), 1)==0:
                        raise self.__AWGEXC('MEMERR','Error in memory allocation: '+repr(self.__lib.ADQControlUnit_GetLastFailedDeviceError(self.__board)))
                        
        
                # Write pattern to dram
                    
                         
                    
                cyn=(usedtype*len(yn))(*yn)
                
                # Write segment data to DRAM
                if self.__lib.ADQ_AWGWriteSegment(self.__board,self.devnum,Channel,i+1,1,Repetitions,len(yn),cyn)==0:
                         raise self.__AWGEXC('MEMERR',"ERROR in writing into AWG memory: "+repr(self.__lib.ADQControlUnit_GetLastFailedDeviceError(self.__board)))
                
                
            #Enable segments
                
            if self.__lib.ADQ_AWGEnableSegments(self.__board,self.devnum,Channel,len(queue))==0:
                raise self.__AWGEXC('ENSEGEXC','ERROR in enabling the segments')
                
                
        def __set_autorearm(self,Channel):
            if self.channels_pars_dict[Channel-1]['Autorearm'] is True:
                en=1
            else:
                en=0
            
            
            if Channel == 3:
                for i in range(2):
                        if self.__lib.ADQ_AWGAutoRearm(self.__board,self.devnum,i,en)==0:
                            raise self.__AWGEXC('AUTOR','Error in setting the autorearm')
            else:
                if self.__lib.ADQ_AWGAutoRearm(self.__board,self.devnum,Channel,en)==0:
                    raise self.__AWGEXC('AUTOR','Error in setting the autorearm')

        def __set_continuous(self,Channel):
            if self.channels_pars_dict[Channel-1]['Continuous'] is True:
                en=1
            else:
                en=0
            
            
            if Channel == 3:
                for i in range(2):
                        if self.__lib.ADQ_AWGContinuous(self.__board,self.devnum,i,en) == 0:
                            raise self.__AWGEXC('CONTERR','Error in setting the continuous mode')
            else:
                if self.__lib.ADQ_AWGContinuous(self.__board,self.devnum,Channel,en) == 0:
                    raise self.__AWGEXC('CONTERR','Error in setting the continuous mode')
            
#----- AWGCL05 ---------------------------------------------------------------------------  Playlist functions
        def trigarm(self,Mode='ON',Channel=3):
            """This function will arm (def) or disarm the specified AWG channel, 
                the AWG must be armed in order to accept triggers and play the sequences
                
            args:
                - Channel is the channel number: 1, 2 or 3 for both channels
                - Enable can be:
                    - 0 or 'OFF' to deactivate the function
                    - 1 or 'ON' to activate the function
            """
            
            Channel = self.__check_ch_num(Channel)
        
            entype=['OFF','ON',0,1]
            en=self.__typecheck(Mode,entype,0,1)
            if en is None:
                raise self.__AWGEXC('TYPEEXC','Wrong Mode inserted: {0,1} or {"on","off"}')
                
            if en==0:
                if Channel == 3:
                    a= self.__lib.ADQ_AWGDisarm(self.__board,self.devnum,1)
                    a+= self.__lib.ADQ_AWGDisarm(self.__board,self.devnum,2)
                    if a!= 2:
                        raise self.__AWGEXC('DISDAC','ERROR in disarming the DACs')
                        
                else:
                    if self.__lib.ADQ_AWGDisarm(self.__board,self.devnum,Channel)==0:
                        raise self.__AWGEXC('DISDAC','ERROR in disarming the DAC {}'.format(Channel))
            else:
                if Channel == 3:
                    a= self.__lib.ADQ_AWGArm(self.__board,self.devnum,1)
                    a+=self.__lib.ADQ_AWGArm(self.__board,self.devnum,2)
                    if a!=2:
                        raise self.__AWGEXC('ARMDAC','ERROR in arming the DACs')
                else:
                    if self.__lib.ADQ_AWGArm(self.__board,self.devnum,Channel)==0:
                        raise self.__AWGEXC('ARMDAC','ERROR in arming the DAC {}'.format(Channel))