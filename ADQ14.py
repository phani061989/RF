# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:33:36 2019

@author: Oscar
v1.2.2 - OSC:
    - inserted wait for WFA completion in the acquire data function
v1.2.1 - OSC:
    - implemented connection to multiple boards when more than one is in the 
    chassis
v1.2.0 - OSC:
    - removed 'SW' trigger mode (useless and it wasn't implemented correctly in this library) and Trigger function
    - removed 'ChannelUsed' and relative function (it is always asked in the AcquireData function)
    - modified some help
v1.1.1 - OSC:
    - Inserted the function trigger_output_setup
    
v1.1.0 - OSC:
    - changed __devnum in devnum
    - inserted dev id, but it needs a better implementation since it can't distinguish between ADQ14 and SDR14

v1.0.1 - fixed the averages conversion, it is now possible to use any number of averages (before it was only a multiple of 2
         the function Acquisition_setup returns now the effective number of samples

v1.0.0 - Library created as an interface with the ADQ14
"""

class ADQ14(object):
    import ctypes as ct
    import numpy as np
    import copy
    import time
    
    version='1.2.2'
    
    def __init__(self,board_num = 1):
        import os
        if os.name == 'nt':
            self.ADQAPI = self.ct.cdll.LoadLibrary("ADQAPI.dll")
        else :
            self.ADQAPI = self.ct.cdll.LoadLibrary("libadq.so")
        

        

        self._PXITrig0=0
        self._PXITrig1=0

        #some c language configuration
        # Manually set return type from some ADQAPI functions
        self.ADQAPI.CreateADQControlUnit.restype = self.ct.c_void_p
        self.ADQAPI.ADQ_GetRevision.restype = self.ct.c_void_p
        self.ADQAPI.ADQ_GetPtrStream.restype = self.ct.POINTER(self.ct.c_int16)
        self.ADQAPI.ADQControlUnit_FindDevices.argtypes = [self.ct.c_void_p]
        
        self.Connect(board_num)
        

        self.__ADCSF = 1.0 #GSamples/s
        
        #Dictionary definition
        self.__pars_dict= {'TriggerMode':'INT',
                             'TriggerOptions': 1e-3,
                             'AcquisitionDelay': 0,
                             'Averages': 1,
                             'Samples': 1080,
                             
                             'Buffer' : [8,1024*1024],
                             'TriggerOutput':{'Mode':'DISABLED','Opts':'RISE','PulseWidth':100,'Inverted':0},
                             
                             }
        
        
        
        self.__trig_type=['','EXT','LVL','INT','','PXI'] #from 1 to 6 on the board
        self.__trig_edge=['FALLING','RISING','BOTH'] #0-2
        
        self.__trig_output_mode = ['DISABLED','INT','REC']
        self.__trig_output_options = ['COPY','RISE','FALL','BOTH']                
        self.__set_triggermode() #def 'SW' trigger is set @ init

        

    # Properties definitions
    @property
    def SF(self):
        return self.__ADCSF
    
    @SF.setter
    def SF(self,arg):
        pass

    #---------------------------------
    @property
    def pars_dict(self):
        return self.copy.deepcopy(self.__pars_dict)

    @pars_dict.setter
    def pars_dict(self,value):
        pass

    #---------------------------------
    @property
    def trig_type(self):
        return self.__trig_type.copy()
    
    @trig_type.setter
    def trig_type(self,value):
        pass
    #---------------------------------
    @property
    def trig_edge(self):
        return self.__trig_edge.copy()
    
    @trig_edge.setter
    def trig_edge(self,value):
        pass
    #---------------------------------
    @property
    def trig_output_mode(self):
        return self.__trig_output_mode.copy()
    
    @trig_output_mode.setter
    def trig_output_mode(self,value):
        pass
    
    #---------------------------------
    @property
    def trig_output_options(self):
        return self.__trig_output_options.copy()
    
    @trig_output_options.setter
    def trig_output_options(self,value):
        pass
    
    #---------------------------------        
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
                
        self._adq = self.ct.c_void_p(self.ADQAPI.CreateADQControlUnit())

        nof_devices = self.ct.c_int(0)
        e = stwrapper()
        
        success = self.ADQAPI.ADQControlUnit_ListDevices(self._adq,self.ct.pointer(e),self.ct.pointer(nof_devices))
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
        
        success = success and self.ADQAPI.ADQControlUnit_OpenDeviceInterface(self._adq,board_num-1)
        success = success and self.ADQAPI.ADQControlUnit_SetupDevice(self._adq, board_num-1)
        
        self.devnum = 1
        self.id = 'ADQ14' #to implement better
        
        # Print ADQAPI revision
        print('ADQAPI loaded, revision {:d}.'.format(self.ADQAPI.ADQAPI_GetRevision()))   
        rev=self.Print_adq_device_revisions()
        
        if rev<30e3:
            print('WARNING: FPGA revision is not in the correct range, are you sure you are using the right board?\n')
        
        
        self.Clock_ref() #ext ref activated by def
        
        
        #self.dig.gainandoffset(1,1,301) #ch1 offset fix (calibration)
        #self.dig.gainandoffset(2,1,361) #ch2 offset fix (calibration)
        
        return 1
    
    def Disconnect(self):
        self.ADQAPI.ADQControlUnit_DeleteADQ(self._adq, self.devnum)
        self.ADQAPI.DeleteADQControlUnit(self._adq)
        
        self._adq= None
        
     
    def Print_adq_device_revisions(self):
        # Get revision info from ADQ
        rev = self.ADQAPI.ADQ_GetRevision(self._adq, self.devnum)
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
        self.ADQAPI.ADQ_Blink(self._adq,self.devnum)

    def Reset(self,Type='COMM'):
        '''function reset([Type='COMM']):
        
        this function will reset the communication as default.
        
        reset('TOT') or reset(1) will reset the board at default settings'''
        
        if type(Type) is str:
            if Type.upper()=='COMM':
                return self.ADQAPI.ADQ_ResetDevice(self._adq,self.devnum,8)
            elif Type.upper()=='TOT':
                return self.ADQAPI.ADQ_ResetDevice(self._adq,self.devnum,2)
            else:
                print('Wrong Type inserted')
                return 0
        else:
            if Type==1:
                return self.ADQAPI.ADQ_ResetDevice(self._adq,self.devnum,2)
            else:
                print('Wrong Type inserted')
                return 0

    def __typecheck(self,Type,Typelist,Min,Max):
    
        if type(Type) is str:
            try:
                num=Typelist.index(Type.upper() )+Min
            except:
                print('Wrong Type inserted')
                return None
        else:
            num=int(Type)
            if num<Min or num>Max:
                print('Wrong Type number inserted')
                return None

        return num   

    class __BASEEXC(Exception):
            pass
        
    class __ADQEXC(__BASEEXC):
        def __init__(self, expression, message):
            self.expression = expression
            self.message = message

    def Clock_ref(self,Type='EXTC'):
        ''' function clock_ref(self,Type='EXTC'):
        
        This function will set the clock reference, Type can be:
        
        - 'INT' or 1 - for the internal clock
        - 'EXT' or 2 - for an external clock
        - 'EXTC' or 3 (def) - for a 10 MHZ clock sync. signal
        '''
        
        ref_types=['INT','EXT','EXTC']
        
        num = self.__typecheck(Type,ref_types,0,2)
        if num is None:
            raise Exception(ValueError)
            
        if num==2:
            num=3
            
        if self.ADQAPI.ADQ_SetClockSource(self._adq,self.devnum,num)==0:
            print('ERROR in setting the clock reference')
            raise Exception('ClockRef')
            
#----------------------------------------------------------------------------------------- Digitizer functions
#------- f01  ---------------------------------------------------------------------- Gain and offset
    def Gainandoffset(self,Channel=1,Gain=None,Offset=None):
        """function GainOffset([Gain=None],[Offset=None])
        
        This function can be used to set the ADC Gain and Offset of the 
        specified Channel {1,2}. 
        If no argument is given, the value will be returned.
        Gain is [-30,30]
        """
        
        """because of a bug, gain and offsets for channel 1 must be executed on channel 1 and 2.
           gain and offsets for channel 2 must be executed on channel 3 and 4"""
           
        if Channel<1 or Channel>2:
            raise self.__ADQEXC('ChannelNum','Error: Wrong channel inserted')
            

        if Gain == None or Offset == None:
                value_type = self.ct.c_int32*1
                t1,t2 = value_type(0), value_type(0)
                g,o= [],[]
                if self.ADQAPI.ADQ_GetGainAndOffset(self._adq,self.devnum,Channel,t1,t2)==0:
                    raise self.__ADQEXC('GainOffsetRead','Error in getting Gain and Offset values')
                g,o = int(t1[0]),int(t2[0])
                del t1,t2
                
        
        loop_indexes = self.np.array([0,1])+2*(Channel-1)+1 #1 and 2 for Channel 1, 3 and 4 for Channel 2

        if Gain != None:
            if Gain<-30 or Gain>30:
                raise self.__DIGEXC("GainValue","Error, wrong gain: -30 < Gain < 30")
            
            Gain= int(Gain*1024)
            
            if Offset != None:
                for i in loop_indexes:
                        if self.ADQAPI.ADQ_SetGainAndOffset(self._adq,self.devnum,int(i),Gain,Offset)==0:
                            raise self.__ADQEXC('GainOffsetWrite','Error in setting Gain and Offset values')
            
            else:
                for i in loop_indexes:
                    if self.ADQAPI.ADQ_SetGainAndOffset(self._adq,self.devnum,int(i),Gain,o)==0: #Offset is unchanged
                            raise self.__ADQEXC('GainOffsetWrite','Error in setting Gain and Offset values')
        else:
            if Offset is None:
                return g/1024,o
            else:
                for i in loop_indexes:
                    if self.ADQAPI.ADQ_SetGainAndOffset(self._adq,self.devnum,Channel,g,Offset)==0: #gain is unchanged
                            raise self.__ADQEXC('GainOffsetWrite','Error in setting Gain and Offset values')
                            
#------ f02 ---------------------------------------------------------------------------  Trigger mode
    def Trigger_mode(self,Mode=None,arg=None):
        '''function trigger_mode(Type='SW'):
        
        This function is used to set the trigger mode, Type can be:
        
        
        - 'EXT' or 2 for external trigger 1
            - further options: edge
                - 0 for falling edge
                - 1 for rising edge (def)
                
        - 'LVL' or 3 for level trigger mode
            - NOT IMPLEMENTED IN THIS LIBRARY
                
        - 'INT' or 4 for internal trigger
            -further option: Period of the internal trigger in s (1e-3 def)
            
        - 'PXI' or 6 for the PXIe trigger (use the proper function to enable it)'''
        
        
            
        if Mode is None: #Query
            tmp= int(self.ADQAPI.ADQ_GetTriggerMode(self._adq,self.devnum))
            if tmp != self.__typecheck(self.pars_dict['TriggerMode'],self.trig_type,1,6):
                raise self.__ADQEXC('TriggerMode','TriggerMode mismatch!')
            
            if tmp == 'EXT' or tmp == 'INT' or tmp == 'LVL':
                return self.trig_type[tmp-1],self.pars_dict['TriggerOptions']
            else:
                return self.trig_type[tmp-1]
        
        if Mode == '':
            raise self.__ADQEXC('TriggerMode','Mode must be specified')
            
        Mode = self.__typecheck(Mode,self.trig_type,1,6)
        if Mode is None:
            raise self.__ADQEXC('TriggerMode','Wrong mode inserted, check .print_trigger_modes() for help: {}'.format(Mode))
        
        self.__pars_dict['TriggerMode']= self.trig_type[Mode-1]
        
        if Mode==2: #EXT
            if arg is None:
                arg = 'RISING'
            Edge = self.__typecheck(arg,self.trig_edge,0,2)
            if Edge is None:
                raise self.__ADQEXC('TriggerEdge','Wrong Edge inserted, check .print_trigger_edges() for help: {}'.format(Edge))
            self.__pars_dict['TriggerOptions'] = self.trig_edge[Edge]
        
        if Mode == 4: #INT
            if arg is None:
                arg = 1e-3
                
            if arg<0 or 1/arg > self.__ADCSF*1e9:
                raise self.__ADQEXC('InternalTriggerPeriod','Frequency must be smaller than 800 MHz')
            
            arg = self.np.floor(arg*self.SF*1e9)/(self.SF*1e9)
            
            self.__pars_dict['TriggerOptions'] = arg
        
        
        if Mode == 3: #LVL
            print("WARNING: Not implemented in this library\n")
            """
            if arg is None:
                arg = [2,1,5958]
            ch = int(arg[0])
            if ch<1 or ch>2:
                raise self.__ADQEXC('CHNERR','Wrong channel inserted: {}'.format(ch)+'/{1,2}' )
            
            edge = self.__typecheck(arg[1],self.trig_edge,0,2)
            if edge is None:
                raise self.__ADQEXC('TriggerEdge','Wrong Edge inserted, check .print_trigger_edges() for help: {}'.format(edge))
            
            lvl = int(arg[2])
            if lvl<=-2**15 or lvl >=2**15:
                raise self.__ADQEXC('LVLERR','Wrong level inserted: {}/[-2**15,2**15]'.format(lvl))
            
            self.__pars_dict['TriggerOptions'] = [ch,edge,lvl]
            """
            self.__pars_dict['TriggerOptions'] = None
        
        self.__set_triggermode()


    def trigger_output_setup(self,Mode=None,Opts=None,PulseWidth=100,Inverted=0):
        """This function setup the output trigger behavior.
        
        Args:
            - Mode: None - query
                    'Disabled' (def, trigger is input)
                    'INT' the internal clock is sent to the trigger out port
                    'REC': when acquisition is triggered, a pulse is sent
            - Opts: more options for the 'INT' mode, None will not change it
                    'COPY': a copy of the internal trigger is sent
                    'RISE': a pulse of PulseWidth is sent on the rising edge of
                            the internal clock (def)
                    'FALL': same as rise
                    'BOTH': 'RISE' and 'FALL' together
            - PulseWidth: the Width of the pulse in ns, ignored in 'COPY' mode
            - Invert: 0 (def) or 1, inverts the logic
        """
        
        if Mode is None:
            return self.pars_dict['TriggerOutput']
    
        if type(Mode)==str:
            Mode = Mode.upper()
            try:
                self.trig_output_mode.index(Mode)
            except ValueError:
                print('Wrong Mode inserted, check trig_output_mode for available modes\n')
                raise
        else:
            print('Mode must be a string, check trig_output_mode for available modes\n')
            raise TypeError

        self.__pars_dict['TriggerOutput']['Mode']=Mode
            
        if type(Opts)==str:
            Opts = Opts.upper()
            try:
                self.trig_output_options.index(Opts)
            except ValueError:
                print('Wrong option inserted, check trig_output_options for available options\n')
                raise
        else:
            print('Opts must be a string, check trig_output_options for available options\n')
            raise TypeError
            
        self.__pars_dict['TriggerOutput']['Opts'] = Opts
        
        if PulseWidth<10:
            print("Pulse Width can't be less than 10 ns, increased to 10 ns\n")
            PulseWidth=10
        
        self.__pars_dict['TriggerOutput']['PulseWidth']= PulseWidth
        
        if Inverted != 0 and Inverted != 1:
            print('Inverted must be {0,1}\n')
            raise ValueError
        
        self.__pars_dict['TriggerOutput']['Inverted']= Inverted
        
        self.__set_trigger_out()
    #----- [acqdel] ------------------------------------------------------------------------ Acquisition delay -------------------------
    '''
    def Acquisition_delay(self,arg=None,mode='t'):
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
            raise self.__ADQEXC('MODEERR','Wrong mode inserted: {"T","S"}')
        
        if arg is None:
            if mode == 0:
                return self.pars_dict['AcquisitionDelay']/self.SF
            else:
                return self.pars_dict['AcquisitionDelay']
            
        if mode: #samples
            ns = int(arg)
        else: #time
            if arg<0:
                raise self.__ADQEXC('NegTime','Time can\'t be negative')
        
            ns = self.np.floor(arg*self.__ADCSF)
            
        
        if ns>2**31:
            raise self.__ADQEXC('LargeDelayTime','Time delay too large, max: '+ str(2**31/self.SF))
        
        
        self.__pars_dict['AcquisitionDelay'] = int(ns)
        self.__set_trigdelay()
    '''
    #----- [acqmode] ------------------------------------------------------------------------ Aqcuisition mode -------------------------
    def Acquisition_setup(self,Averages=1,Samples=1008,Delay=0):
        """
        This function sets the acquisition parameters.
        
        args:
            - Averages (def 1), is the number of averages (on-board)
            - Samples (def 1008): is the number of points to acquire, it must be a multiple of 72, it will be automatically corrected
            - Delay (def 0): delay in ns from the trigger
        
        returns:
            - corrected number of samples
        """
        
        Averages = int(Averages)
        if Averages <1:
            raise self.__ACQEXC('AVEERR','Averages can\'t be less than 1: {}'.format(Averages))
        if Averages > 65536:
            raise self.__ACQEXC('AVEERR','Averages can\'t be more than 65536: {}'.format(Averages))

        #Check if Averages is a power of 2:
        """
        tmp = self.np.log2(Averages)
        if tmp-int(tmp) != 0:
            Averages = int(2**self.np.ceil(tmp))
            print('WARNING: Averages must be a power of 2, it has been increased to: {}'.format(Averages))
        """    
        #Evaluates the maximum number of samples in function of the records
        Samples = int(Samples)
        if  Samples % 72 != 0:
            Samples = self.np.ceil(Samples/72)
            Samples = int(Samples*72)
            print("WARNING: Samples must be a multiple of 72. Number of samples has been increased to: {}".format(Samples))
        #sam_limit = self.ct.c_uint()
        
        #self.ADQAPI.ADQ_GetMaxNofSamplesFromNofRecords (self._adq,self.devnum,Records,self.ct.byref( sam_limit))
        
        #sam_limit = int(sam_limit.value)
        #if Samples >= sam_limit:
            #raise self.__ADQEXC('SAMERR','Samples must be in the range: [16,{}]'.format(sam_limit))


        

        #Delay setup        
        Delay = int(Delay)
        if Delay<0:
            raise self.__ADQEXC('NegTime','Time can\'t be negative')
    
        if Delay>2**31:
            raise self.__ADQEXC('LargeDelayTime','Time delay too large, max: '+ str(2**31/self.SF))
        
        
        self.__pars_dict['AcquisitionDelay'] = int(Delay)
        self.__pars_dict['Averages'],self.__pars_dict['Samples'] = Averages,Samples
        
        self.__setup_acq()
        
        return Samples


    #--------- [buffer] --------------------------------------------------------------------- Buffer setup ------------------
    def Buffer(self,Buffers=8,Size=2**20):
        Buffers = int(Buffers)
        if Buffers<1:
            self.__ADQEXC('NUMERR','Buffers can\'t be less than 1\n')
        
        
        Size = int(Size)
        if Size<1:
            self.__ADQEXC('NUMERR','Size can\'t be less than 1\n')
            
        #Check if size is a power of 2:
        tmp = self.np.log2(Size)
        if tmp-int(tmp) != 0:
            Size = int(2**self.np.ceil(tmp))
            print('WARNING: Size must be a power of 2, Size has been increased to: {}'.format(Size))
        
        self.__pars_dict['Buffer'] = [Buffers,Size]
            
    #----- [SETF] -------------------------------------------------------------------------- Set functions --------------------
        
    def __set_triggermode(self):
                
        Mode = self.__typecheck(self.pars_dict['TriggerMode'],self.trig_type,1,6)
        if self.ADQAPI.ADQ_SetTriggerMode(self._adq,self.devnum,Mode)==0:
            raise self.__ADQEXC('TrigMode','ERROR in setting the trigger mode:{}'.format(self.pars_dict['TriggerMode']))
            
        
        if Mode==2: #EXT
            Edge = self.__typecheck(self.pars_dict['TriggerOptions'],self.trig_edge,0,2)
            if self.ADQAPI.ADQ_SetExternTrigEdge(self._adq,self.devnum,Edge) == 0:
                raise self.__ADQEXC('TrigEdge','ERROR in setting the external trigger edge:{}'.format(Edge))
        
        if Mode==3: #LVL
            print('Not implemented, set LVL trigger manually')
            """
            self.__set_trigger_channel(self.pars_dict['TriggerOptions'][0])
            Edge = self.__typecheck(self.pars_dict['TriggerOptions'][1],self.trig_edge,0,2)
            if self.ADQAPI.ADQ_SetLvlTrigEdge(self._adq,self.devnum,Edge) == 0:
                raise self.__ADQEXC('TrigEdge','ERROR in setting the level trigger edge:{}'.format(Edge))
            lvl = self.pars_dict['TriggerOptions'][2]
            if self.ADQAPI.ADQ_SetLvlTrigLevel(self._adq,self.devnum,lvl) == 0:
                raise self.__ADQEXC('TrigLevel','ERROR in setting the level trigger level:{}'.format(lvl))
        else:
            self.__set_trigger_channel(0) #Internal/external trigger
            """
        if Mode==4: #INT
            per = self.pars_dict['TriggerOptions']
            per = int(per*self.SF*1e9)
            if self.ADQAPI.ADQ_SetInternalTriggerPeriod(self._adq,self.devnum,per)==0:
                raise self.__ADQEXC('IntTrigPer','Error in setting the internal trigger period: {}'.format(per))
    

    def __set_trigger_out(self):
        pars = self.pars_dict['TriggerOutput']
        
        if pars['Mode'] == 'DISABLED':
            self.ADQAPI.ADQ_SetupTriggerOutput(self._adq,self.devnum,0,0,100,0) #Trigger is disabled
            return
        
        if pars['Mode'] == 'REC':
            self.ADQAPI.ADQ_SetupTriggerOutput(self._adq,self.devnum,0,3,pars['PulseWidth'],pars['Inverted']) #Trigger is disabled
            return
    
        if pars['Mode'] == 'INT':
            if pars['Opts'] == 'COPY':
                mode = 2
            elif pars['Opts'] == 'RISE':
                mode = 5
            elif pars['Opts'] == 'FALL':
                mode = 6
            elif pars['Opts'] == 'BOTH':
                mode = 7
            else:
                print('Unknown situation, check the source code:{}\n'.format(pars['Opts']))
                raise ValueError
                
            self.ADQAPI.ADQ_SetupTriggerOutput(self._adq,self.devnum,0,mode,pars['PulseWidth'],pars['Inverted']) #Trigger is disabled
            return
            
    def __setup_acq(self):
        # Setup WFA
        success = self.ADQAPI.ADQ_ATDSetupWFA(self._adq, self.devnum,
                                 self.pars_dict['Samples'],
                                 0,
                                 self.pars_dict['AcquisitionDelay'],
                                 self.pars_dict['Averages'],
                                 1)
        if (success == 0):
            self.__ADQEXC('ATDERR','ADQ_ATDSetupWFA failed.')
        
        print('Setting up streaming...')
        success=self.ADQAPI.ADQ_SetTransferBuffers(self._adq, self.devnum, 
                              self.pars_dict['Buffer'][0],
                              self.pars_dict['Buffer'][1])
        
        if (success == 0):
            self.__ADQEXC('BUFFERR','ADQ_SetTransferBuffers failed.')

    
    #------ [ACQFUN]  ----------------------------------------------------------------------------- Acquisition functions
    
    #----- [ACQUIRE] -------------------------------------------------------------------        ACQUIRE DATA ------------ 
    """
    def Acquire_data(self,channel_mask = 3,conversion_to_volt=True):
        '''This function start the acquisition, it will wait for the trigger.
        '''
        
        #check channel_mask first:
        channel_mask = int(channel_mask)
        if channel_mask<1 or channel_mask>3:
            self.__ADQEXC('CHMERR','Channel mask must be in the range [1,3], spiecified: {}\n'.format(channel_mask))
        
        
        target_buffers = (self.ct.POINTER(self.ct.c_int32*(self.pars_dict['Samples']))*4)()
        for bufp in target_buffers:
            bufp.contents = (self.ct.c_int32*(self.pars_dict['Samples']))()

        
        
        
        success = self.ADQAPI.ADQ_ATDStartWFA(self._adq, self.devnum,
                                 target_buffers, 0xf, 1)
        if success == 0:
            self.__ADQEXC('STARTERR','ADQ_ATDStartWFA failed!')
        
        #getting data
        
        #init only what is necessary:
        if channel_mask<3:
            data_32bit = self.np.array([], dtype=self.np.int32)
            if channel_mask == 1:
                chn=0
            else:
                chn=1
                
            data_32bit = self.np.frombuffer(target_buffers[chn].contents,
                            dtype=self.np.int32, count= self.pars_dict['Samples'])
            if conversion_to_volt:
                data_32bit = data_32bit*2/ 2**(16+self.np.log2(self.pars_dict['Averages'])) #2 Vpp
            else:
                data_32bit = data_32bit/ self.pars_dict['Averages'] #digits
        else:
            data_32bit = [self.np.array([], dtype=self.np.int32),
              self.np.array([], dtype=self.np.int32)]
        
            for ch in range(2):
                tmp = self.np.frombuffer(target_buffers[ch].contents,
                            dtype=self.np.int32, count= self.pars_dict['Samples'])
                try:
                    if conversion_to_volt:
                        tmp = 2*tmp/2**(16+int(self.np.log2(self.pars_dict['Averages']))) #2 Vpp
                    else:
http://localhost:8988/edit/python_repo/ADQ14.py#                        tmp = tmp/self.pars_dict['Averages'] #digits
                except:
                    print('Error in the data conversion, returned the first data array')
                    return tmp
                
                data_32bit[ch] = tmp.copy()
                            

        return data_32bit       
    """
    
    def Acquire_data(self,channel_mask = 3,conversion_to_volt=True,timeout=5):
        '''This function start the acquisition, it will wait for the trigger.
        '''
        
        #check channel_mask first:
        channel_mask = int(channel_mask)
        if channel_mask<1 or channel_mask>3:
            self.__ADQEXC('CHMERR','Channel mask must be in the range [1,3], spiecified: {}\n'.format(channel_mask))
        
        
        pointer = (self.ct.POINTER(self.ct.c_int32)*2)() #2 channels
        datach1,datach2 = self.np.ndarray(self.pars_dict['Samples'],dtype=self.np.int32),self.np.ndarray(self.pars_dict['Samples'],dtype=self.np.int32)
        pointer[0],pointer[1] = datach1.ctypes.data_as(self.ct.POINTER(self.ct.c_int32)),datach2.ctypes.data_as(self.ct.POINTER(self.ct.c_int32))
        
        
        
        success = self.ADQAPI.ADQ_ATDStartWFA(self._adq, self.devnum,
                                 pointer, 0xf, 1)
        if success == 0:
            self.__ADQEXC('STARTERR','ADQ_ATDStartWFA failed!')
        
        
        #Waiting for WFA
        start_time = self.time.time()
        while(not self.ADQAPI.ADQ_ATDWaitForWFACompletion(self._adq,self.devnum)):
            if (self.time.time()-start_time) > timeout:
                self.__BASEEXC('TIMEOUT','WFA failed!')
        
        #getting data
        
        #init only what is necessary:
        if channel_mask<3:
            
            if channel_mask == 1:
                selected_data = datach1
            else:
                selected_data = datach2
                
            
            if conversion_to_volt:
                selected_data = selected_data*2/ 2**(16+self.np.log2(self.pars_dict['Averages'])) #2 Vpp
            else:
                selected_data = selected_data/ self.pars_dict['Averages'] #digits
                
            return selected_data
        else:
                datach1 = datach1/self.pars_dict['Averages'] #digits
                datach2 = datach2/self.pars_dict['Averages'] #digits
                
                if conversion_to_volt:
                    datach1 = 2*datach1/(2**16-1) #2 Vpp
                    datach2 = 2*datach2/(2**16-1)#2 Vpp
                
            
                
                return datach1,datach2
    """        
    #----------- [SWTRIG] -------------------------------------------------------------- SW trigger -------            
    def Trigger(self):
        '''function trigger():
        
        This function will software-trigger the board'''
        
        if self.ADQAPI.ADQ_SWTrig(self._adq,self.devnum) == 0:
            print('Error in triggering the device')
            raise Exception('TrigEXC')


    """