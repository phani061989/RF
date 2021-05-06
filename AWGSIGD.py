#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:09:39 2017

@author: oscar
v2.6.1 - OSC:
    - samples limit was underestimated, now it should be correct

v2.6.0 - OSC:
    - made the names of some parameters more uniform
    - changed the function to clear queue to use a ChannelMask, for consistency
    - loading/removing waves, queuing in channels will act locally, final loading
    can be done manually or it will be automatically done when starting the AWG.
    
v2.5.4 - OSC:
    - created the function insert_sequence_IQ that get a sequence object 
    and a calibration file and sets them in the channels adjusting the phase
    automatically
    
v2.5.3 - OSC:
    - when loading waves in memory, the channels queue is cleaned. Now after 
    calling the function load_waves_in_AWG_memory, they will be re-queued 
    automatically
    
v2.5.2 - OSC:
    - inserted the stop_multiple function
    - inserted ext port setup functions
    - inserted queue mode 
    - inserted trigger port direction
    
v2.5.1 - OSC:
    - inserted the function parameters_dictionary for compatibility reasons, 
    it creates a dictionary of all the channels parameters
v2.5.0 - OSC:
    - changed dev id
v2.4.0 - OSC:
    - adapted the apply_corection function to the new CALPAR format, old cal files will not work
    
v2.3.1 - OSC:
    - removed the analog trigger from the options
    - fixed the print_channel_queue function
    - fixed the get_queue_length() function
    - fixed the plot_queue() function
    
v2.3.0 - OSC:
    - fixed a bug in the set_channel function
    - wrote help for the functions
    - moved print function in the main class
    
v2.2.1 - OSC:
    - fixed a bug inherent to wave creation from array in the AWG that was limiting the number of created waves (thanks to Aleksei)
    - improved error printout
    
v2.2.0 - OSC:
    - inserted autoset mode
    - changed the set_channel channel number to a mask mode, to avoid a problem with memory flushing
    - "protected" the channel dictionary
    
v2.1.3 - OSC:
    - the function insert in queue accept Repetitons = 0, that means infinite
    
v2.1.2 - OSC:
    - inserting a wave with the same ID, will overwrite the existing one and 
    not add another wave (BUGFIX)
    - inserted a function to insert a sequence object

v2.1.1 - OSC:
    - inserted a function to get a channel queue total length
    - inserted a device.id for compatiblity
v2.1.0 - OSC:
    - inserted a channel number id in the parameters, it should never be forced
    
v2.0.1 - OSC:
    - chassis and slot are now stored
    - inserted a function to get chassis and slot
    
v2.0.0 - OSC:
    - using dictionaries to get and set channels parameters
    
v1.2.1 - OSC:
    - apply_correction set the speed of the offset to immediate
    
    
v1.2.0 - OSC:
    - apply_corr will take directly the cal_dictionary from now on

v1.1.1 - OSC:
    - if the calibration is lsb, the phase is -90

v1.1.0 - OSC:
    - Introduced the slow change of offset (def is 1mV step and 1ms step time). The function to set/get the offset has been changed.

v1.0.1 - OSC:
    - moved the queue option in the channel, because the same wave can be queued with different options (less memory usage)
    - inserted a warning for too much memory

"""



class AWGChannel(object):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    
    def __init__(self, number):
        self.__FR_MAX = 450 #MHz
        self.__AMP_MIN = -1.5
        self.__AMP_MAX = 1.5
        if number<0 or number>3:
            print('Wrong channel number inserted')
            raise Exception(ValueError)
        
        self.__chn = number
        self.__modes_list = ['OFF','SIN','TRIANGULAR','PARTNER','SQUARE','DC','AWG']

        self.__trig_list = ['AUTO','SW','EXT','','', 'SW_CYCLE','EXT_CYCLE']

        self.__pars_dict = {'Channel number': self.__chn,
                         'Frequency':0,
                         'Amplitude':0,
                         'Offset' : 0,
                         'Offset_step' : 1e-3,
                         'Offset_speed' : 1e-3,
                         'Modulation': False,
                         'Phase': 0,
                         'Mode': 'OFF',
                         'Queue_mode': 'CYCLIC',
                         'Queue_list': [],
                         
                
                }
        
        
        self.__ext_port_list = ['EXT','PXI0','PXI1','PXI2','PXI3','PXI4','PXI5','PXI6','PXI7'] #0, 400x, x = idx-1
        self.__trigger_edge_list = ['AUTO','HIGH','LOW','RISE','FALL']
        self.__queue_mode_list = ['ONE_SHOT','CYCLIC']
        self.__ext_trig = {'PORT': 'EXT',
                           'EDGE': 'RISE',
                           'Sync' : True,
                
                }
        
    @property
    def pars_dict(self):
        return self.copy.deepcopy(self.__pars_dict)
    @pars_dict.setter
    def pars_dict(self,value):
        pass

    @property
    def ext_trigger(self):
        return self.copy.deepcopy(self.__ext_trig)
    @ext_trigger.setter
    def ext_trigger(self,value):
        pass

    @property
    def trigger_modes_list(self):
        return self.copy.deepcopy(self.__trig_list)
    @trigger_modes_list.setter
    def trigger_modes_list(self,value):
        pass

    @property
    def wave_modes_list(self):
        return self.copy.deepcopy(self.__modes_list)
    @wave_modes_list.setter
    def wave_modes_list(self,value):
        pass

    @property
    def ext_port_list(self):
        return self.copy.deepcopy(self.__ext_port_list)
    @ext_port_list.setter
    def ext_port_list(self,value):
        pass

    @property
    def trigger_edge_list(self):
        return self.copy.deepcopy(self.__trigger_edge_list)
    @trigger_edge_list.setter
    def trigger_edge_list(self,value):
        pass

    @property
    def queue_mode_list(self):
        return self.copy.deepcopy(self.__queue_mode_list)
    @queue_mode_list.setter
    def queue_mode_list(self,value):
        pass

##----------------------------------------------------------------------------------- ch01 checks ------------------------------    
    def __check_total_amplitude(self):
        tmp = self.pars_dict['Offset']+self.pars_dict['Amplitude']
        
        if tmp<self.__AMP_MIN or tmp>self.__AMP_MAX:
            print('WARNING: total channel amplitude out of range: {} V \n'.format(tmp))
    
## ---------------------------------------------------------------------------------- ch02 properties -----------------------------


    
    def frequency(self,Frequency=None):
        """Oscillator frequency in MHz"""
        if Frequency is None:
            return self.pars_dict['Frequency']
    
        if Frequency < 0 or Frequency > self.__FR_MAX:
            print('Frequency out of range: [{},{}]\n'.format(0,self.__FR_MAX))
            raise ValueError
            
        self.__pars_dict['Frequency'] = self.np.round(Frequency,12)
    
    def offset(self, Offset=None,Step_size=1e-3,Step_speed=1e-3):
        
        if Offset is None:
            return self.pars_dict['Offset'],self.pars_dict['Offset_step'],self.pars_dict['Offset_speed']
    
        if Offset < self.__AMP_MIN or Offset > self.__AMP_MAX:
            print('Offset out of range: [{},{}]\n'.format(self.__AMP_MIN,self.__AMP_MAX))
            raise ValueError

        if Step_size < 0:
            print("Step_size can't be negative")
            raise ValueError

        if Step_speed < 0:
            print("Step_sped can't be negative")
            raise ValueError

        self.__pars_dict['Offset'] = self.np.round(Offset,12)    
        self.__pars_dict['Offset_step'] = self.np.round(Step_size,12)
        self.__pars_dict['Offset_speed'] = self.np.round(Step_speed,12)
        self.__check_total_amplitude()
    
    def amplitude(self, Amplitude=None):
        if Amplitude is None:
            return self.pars_dict['Amplitude']
    
        if Amplitude < self.__AMP_MIN or Amplitude > self.__AMP_MAX:
            print('Amplitude out of range: [{},{}]\n'.format(self.__AMP_MIN,self.__AMP_MAX))
            raise ValueError

        self.__pars_dict['Amplitude'] = self.np.round(Amplitude,12)    
        self.__check_total_amplitude()
    
    def mode(self,Mode=None):
        if Mode is None:
            return self.pars_dict['Mode']
        
        if type(Mode) is str:
            Mode = Mode.upper()
            
            try:
                self.wave_modes_list.index(Mode)
            except ValueError:
                print('Wrong mode inserted, use print_modes_list() to check available modes\n')
                raise
        else:
            try:
                Mode = self.wave_modes_list[int(Mode)]
            except IndexError:
                print('Wrong mode inserted, use print_modes_list() to check available modes\n')
                raise
                
        self.__pars_dict['Mode'] = Mode
    
    def phase(self,Phase=None):
        if Phase is None:
            return self.pars_dict['Phase']
    
        
        self.__pars_dict['Phase'] = self.np.round(Phase % 360,12)
    
    def modulation(self,Modulation=None):
        if Modulation is None:
            return self.pars_dict['Modulation']
        
        if Modulation != False and Modulation != True and Modulation != 0 and Modulation != 1:
            print('Amplitude modulation can only be active (True) or not (False)\n')
            raise ValueError
        
        self.__pars_dict['Modulation'] = Modulation
        
    def queue_mode(self,Mode=None):
        if Mode is None:
            return self.pars_dict['Queue_mode']
        
        if type(Mode) is str:
            Mode = Mode.upper()
            
            try:
                self.queue_mode_list.index(Mode)
            except ValueError:
                print('Wrong mode inserted, use print_queue_modes_list() to check available modes\n')
                raise
        else:
            try:
                Mode = self.queue_mode_list[int(Mode)]
            except IndexError:
                print('Wrong mode inserted, use print_queue_modes_list() to check available modes\n')
                raise
                
        self.__pars_dict['Queue_mode'] = Mode


    def ext_trigger_config(self,port=None,edge=None,Sync=None):
        """ This function configure the external trigger
        
        args:
            No args will query
            - port: sets the external port for this channel
            - edge: sets the edge for the external signal
            - Sync: synchronized to 100 MHz clock or async
        
        It is possible to set also only one of the args by leaving the rest 
        to None
        """
        
        if port is None and edge is None and Sync is None:
            return self.ext_trigger
        
        if port is not None:
            if type(port) is str:
                port = port.upper()
            
                try:
                    self.ext_port_list.index(port)
                except ValueError:
                    print('Wrong port inserted, use print_ext_port_list() to check available ports\n')
                    raise
            else:
                try:
                    port = self.ext_port_list[int(port)]
                except IndexError:
                    print('Wrong port inserted, use print_ext_port_list() to check available ports\n')
                    raise
            
            self.__ext_trig['PORT'] = port
    
        if edge is not None:
            if type(edge) is str:
                edge = edge.upper()
            
                try:
                    self.trigger_edge_list.index(edge)
                except ValueError:
                    print('Wrong edge inserted, use print_trigger_edge_list() to check available edges\n')
                    raise
            else:
                try:
                    port = self.trigger_edge_list[int(edge)]
                except IndexError:
                    print('Wrong edge inserted, use print_trigger_edge_list() to check available edges\n')
                    raise
            
            self.__ext_trig['EDGE'] = edge
            
        if Sync is not None:
            if Sync != False and Sync != True and Sync != 0 and Sync != 1:
                print('Sync modulation can only be active (True) or not (False)\n')
                raise ValueError
            
                self.__ext_trig['Sync'] = Sync
        
##------------------------------------------------------------------------------------ch03 print properties ----------------------
    


    def print_channel_parameters(self):
        text = ''
        
        text+= 'Channel number: {}\n'.format(self.__chn)
        text+= 'Frequency: {} MHz\n'.format(self.frequency())
        text+= 'Amplitude: {} V\n'.format(self.amplitude())
        text+= 'Offset: {} V\n'.format(self.offset()[0])
        text+= 'Offset Step: {} V\n'.format(self.offset()[1])
        text+= 'Offset Step Wait: {} s\n'.format(self.offset()[2])
        text+= 'Phase: {} deg\n'.format(self.phase())
        text+= 'Mode: {}\n'.format( self.mode())
        text+= 'Amplitude Modulation: {}\n'.format(self.modulation())
        text+= 'Queue Mode: {}\n'.format(self.queue_mode())
        text+= 'Queue elements: {}\n'.format(len(self.pars_dict['Queue_list']))
        
        print(text)
    """    
    def print_queue(self):
        
        text=''
        
        for i,q in enumerate(self.pars_dict['Queue_list']):
            text+= 'Queue element number {}\n'.format(i)
            text+= 'Wave ID: {}\n'.format(q[0])
            text+= 'Trigger Mode: {}\n'.format(self.trigger_modes_list[q[1]])
            text+= 'Delay: {} ns\n'.format(q[2])
            text+= 'Repetitions: {}\n'.format(q[3])
            text+= 'Prescaler: {}\n\n'.format(q[4])
        
        if len(text)>0:
            print(text)
        else:
            print('Queue is empty')
    """            

##--------------------------------------------------------------------------------------- ch04 queue ------------------------------

        
    def insert_in_queue(self,ID,TriggerMode=0,Delay=0,Repetitions=1,Prescaler=0):
        """ 
        This function queue a wave by ID in the channel.
        
        function parameters:
            - TriggerMode: use print_modes_list()
            - Delay: delay after trigger, in ns
            - Repetitions: how mane time the segment is repeated, 0 for infinite
            - Prescaler: check the manual for more infos, it is used to resample the wave
        """
        
        if type(TriggerMode) is str:
            TriggerMode = TriggerMode.upper()
            if TriggerMode == '':
                print('Wrong mode inserted, use print_modes_list() to check available modes\n')
                raise ValueError
            
            try:
                self.trigger_modes_list.index(TriggerMode)
            except ValueError:
                print('Wrong mode inserted, use print_modes_list() to check available modes\n')
                raise 
        else:
            try:
                TriggerMode = self.trigger_modes_list[int(TriggerMode)]
            except IndexError:
                print('Wrong mode inserted, use print_triggermodes_list() to check available modes\n')
                raise
        
        if Delay <0:
            print('ERROR: Delay cannot be negative')
            raise ValueError
            
        if Repetitions <0:
            print('ERROR: Repetitions cannot be lees than one')
            raise ValueError
        
        if Prescaler <0:
            print('ERROR: Prescaler cannot be negative')
            raise ValueError
        
        
        tmp={'WaveID':ID,
             'TriggerMode':TriggerMode,
             'Delay':Delay,
             'Repetitions':Repetitions,
             'Prescaler':Prescaler,
                }
        
        self.__pars_dict['Queue_list'].append(tmp)
        
        
        
    
    def remove_from_queue(self,ID):
        for i,q in enumerate(self.pars_dict['Queue_list']):
            if q['WaveID']==ID:
                self.__pars_dict['Queue_list'].pop(i)
        
        
    def clear_queue(self):
        self.__pars_dict['Queue_list'] = []



        

    

    
##------------------------------------------------------------------------------------------ a00 AWG class --------------------------------    
        
class Awgsigd(object):
    version = '2.6.1'
    import keysightSD1 as sigd
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import copy
    
    def __init__(self,slot,chassis=0,autoset = True,init_AWG=True):
        """
        Initialize the Awgsigd class by specifying the slot number and the
        chassis number (def is 0). 
        
        - autoset: True by default, when changing a channel parameter, it will
        be set on the board as soon as possible. If it is False, the user needs
        to call the set_channel function to send all the parameters and waves
        to the board.
        
        - init_dig: True by def, it initialize the awg to the default settings
        when the class is initialized, otherwise only the dictionary will 
        contain the default settings, but the board is untouched.
        
        """ 
        
        self._awg = self.sigd.SD_AOU()
        self.__check_command( self._awg.openWithSlot('',chassis,slot) )
        
        self.__channels = []
        self.__prev_offset = []
        for i in range(4):
            self.__channels.append(AWGChannel(i))
            self.__prev_offset.append(0.) #there is no way to knwo the offset value of the channel
        
        self.__SF = 1.0 #GS/S
        self.__waves = []
        self.__waves_ID = []
        self.__max_points = 1048576
        self.__MINFREQ = 0 #MHz - NOTE: DC allowed, it is still not clear the freq resolution
        self._slot=slot
        self._chassis=chassis
        self.id = 'AWGKEY'
        
        self.trig_port_direction() #def is IN
        
        self.__autoset = True
        self.autoset = autoset
        
        if init_AWG:
            self.set_channel(15)
    
    @property
    def autoset(self):
        return self.__autoset
    
    @autoset.setter
    def autoset(self,value):
        if value==1 or value ==True:
            self.__autoset=True
        elif value==0 or value == False:
            self.__autoset = False
        else:
            print('Wrong value inserted: [True,False]\n')
            
    @property
    def SF(self):
        """Sampling frequency in GS/s """
        return self.__SF
    
    @SF.setter
    def SF(self,value):
        pass
    
##----------------------------------------------------------------------------------- cherr -----------------------------------
    class __AWGERR(Exception):
        pass

    class __AWGCMDERR(__AWGERR):
        def __init__(self,expression,message):
            self.message=message
            self.expression = expression
    
#--------------------------------------------------------------------------------------- a01 - checks -----------------------------
    def __check_samples_number_in_memory(self):
        tmp=0
        for n in self.__waves:
            tmp+=len(n)
        
        if tmp > self.__max_points:
            print('WARNING: AWG memory limit reached')
        
    def __check_channel_number(self,Channel):
        if Channel< 0 or Channel >3:
            print('Wrong Channel number inserted: [0,3]\n')
            raise Exception('CHNERR')
    
    def __check_command(self,cmd):
        if cmd <0:
            #print('ERROR CODE: {} - {}\n'.format(cmd,self.sd1.SD_Error.getErrorMessage(cmd)))
            raise self.__AWGCMDERR(str(cmd),self.sigd.SD_Error.getErrorMessage(cmd))
            
    def __check_wave_number(self,Number):
        if Number<0 or Number > len(self.__waves)-1:
            print('Wrong Number inserted: [0,{}]\n'.format(len(self.__waves)-1) )
            raise Exception('WAVENUMERR')

#------------------------------------------------------------------------------- p00 - print functions --------------------
    def print_wave_modes_list(self):
        """This function prints the wave modes that can be used with any 
        channel. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(0).wave_modes_list):
            text+= str(i)+': '+m+'\n'
        print(text)

    def print_trigger_modes_list(self):
        """This function prints the trigger modes that can be used with any 
        channel. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(0).trigger_modes_list):
            text+= str(i)+': '+m+'\n'
        print(text)      


    def print_trigger_edges_list(self):
        """This function prints the trigger signal edge modes that can be used 
        with any channel. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(0).trigger_edge_list):
            text+= str(i)+': '+m+'\n'
        print(text)      

    def print_queue_modes_list(self):
        """This function prints the queue mode that can be used 
        with any channel. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(0).queue_mode_list):
            text+= str(i)+': '+m+'\n'
        print(text)    
    
    def print_ext_ports_list(self):
        """This function prints the available external ports that can be used 
        with any channel. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(0).ext_port_list):
            text+= str(i)+': '+m+'\n'
        print(text)    
    
#-------------------------------------------------------------------------------------- a02 - properties ----------------------------

            
    def amplitude(self, Channel,Amplitude=None):
        """This function sets the Vp voltage for the specified channel: [0,1.5] V.
        If the value is None (def) it will be queried."""
        self.__check_channel_number(Channel)
        
        if Amplitude is None:
            return self.__channels[Channel].amplitude(Amplitude)
        else:
            self.__channels[Channel].amplitude(Amplitude)
        
        if self.autoset:
            if self.modulation(Channel):
                self.__check_command( self._awg.channelAmplitude(Channel,0 ) )
                self.__check_command( self._awg.modulationAmplitudeConfig( Channel,1,self.amplitude(Channel) )  )
                
            else:
                self.__check_command( self._awg.channelAmplitude(Channel,self.amplitude(Channel) ) )
                self.__check_command( self._awg.modulationAmplitudeConfig( Channel,0,0 )  )
            
        
    def offset(self, Channel,Offset=None,Step_size=1e-3,Speed=1e-3):
        """This function sets the channel offset, the offset is changed 
        slowly as default, with a step (V, def 1mV) and a delay (s, 1ms def). 
        If Speed is 0, it will be as fast as possible (AWG speed limit). If the
        Offset value is None, it will be queried, returning a list: 
        (Offset, Step, Wait)."""
        
        self.__check_channel_number(Channel)
        
        
        
        if Offset is None:
            return self.__channels[Channel].offset(Offset)
        else:
            self.__channels[Channel].offset(Offset,Step_size,Speed)
        
        if self.autoset:
            self.__change_offset_slow(Channel)
    
    def frequency(self, Channel,Frequency=None):
        """This function sets the oscillator frequency in MHz for the specified 
        channel: [0,450] MHz. If the value is None (def) it will be queried."""
        self.__check_channel_number(Channel)
        
        if Frequency is None:
            return self.__channels[Channel].frequency(Frequency)
        else:
            self.__channels[Channel].frequency(Frequency)
            
        if self.autoset:
            self.__check_command( self._awg.channelFrequency(Channel,self.np.round(self.frequency(Channel)*1e6,12) ) )
    
    def phase(self, Channel,Phase=None):
        """This function sets the Phase (deg) for the specified channel: [0,360] deg.
        If the value is None (def) it will be queried."""
        self.__check_channel_number(Channel)
        
        if Phase is None:
            return self.__channels[Channel].phase(Phase)
        else:
            self.__channels[Channel].phase(Phase)
            
        if self.autoset:
            self.__check_command( self._awg.channelPhase(Channel,self.phase(Channel) ) )
            self.phase_reset(1<<Channel)
    
    def mode(self, Channel,Mode=None):
        """This function sets the wave modefor the specified channel. Check the
        corresponding print function for the available modes.
        If the value is None (def) it will be queried."""
        self.__check_channel_number(Channel)
        
        tmp = self.__channels[Channel]
        if Mode is None:
            return tmp.mode(Mode)
        else:
            tmp.mode(Mode)
        
        if self.autoset:
            self.__check_command( self._awg.channelWaveShape(Channel,tmp.wave_modes_list.index( self.mode(Channel) )))
    
    def modulation(self, Channel, Modulation=None):
        """This function activate the amplitude modulation for the specified
        channel. Be aware that other types of modulation are available but
        non inserted in this library (and not tested). If the value is 
        None (def) it will be queried."""
        self.__check_channel_number(Channel)
        
        if Modulation is None:
            return self.__channels[Channel].modulation(Modulation)
        else:
            self.__channels[Channel].modulation(Modulation)
            
        if self.autoset:
            self.amplitude(Channel,self.amplitude(Channel)) #If one changes the modulation, the amplitude of the channel switches

    def queue_mode(self, Channel,Mode=None):
        """This function sets the queue mode for the specified channel. 
        If the value is None (def) it will be queried."""
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel]
        
        if Mode is None:
            return self.__channels[Channel].queue_mode(Mode)
        else:
            self.__channels[Channel].queue_mode(Mode)
            
        if self.autoset:
            self.__check_command(self._awg.AWGqueueConfig(Channel,tmp.queue_mode_list.index(self.queue_mode(Channel))))


    def ext_port_setup(self,Channel,port=None,edge=None,Sync=None):
        """This function can be used to configure the external trigger behavior
        for the specified channel.
        
        args:
            No args will query
            - port: sets the external port for this channel
            - edge: sets the edge for the external signal
            - Sync: synchronized to 100 MHz clock or async
        
        It is possible to set also only one of the args by leaving the rest 
        to None
        """

        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel]
        
        if port is None and edge is None and Sync is None:
            return tmp.ext_trigger_config()
        else:
            tmp.ext_trigger_config(port,edge,Sync)
            
        if self.autoset:
            pars = tmp.ext_trigger_config()
            pars = pars['PORT'],pars['EDGE'],pars['Sync']
            port_sn = tmp.ext_port_list.index(pars[0])
            if port_sn>0:
                port_sn= int(port_sn+3999)
            self.__check_command(self._awg.AWGtriggerExternalConfig(Channel,port_sn,tmp.trigger_edge_list.index(pars[1]),int(pars[2])))

    def trig_port_direction(self,direction=0):
        '''Sets the direction of the trigger port as input or output.
        
        Args:
            - direction: 'in' or 0 for input direction
                         'out' or 1 for output
        
        '''
        direction = str(direction).upper() 
        if direction == 'IN' or direction == '0':
            self.__check_command(self._awg.triggerIOconfig(self.sigd.SD_TriggerDirections.AOU_TRG_IN))
        elif direction == 'OUT' or direction == '1':
            self.__check_command(self._awg.triggerIOconfig(self.sigd.SD_TriggerDirections.AOU_TRG_OUT))
        else:
            raise self.__AWGCMDERR('TRDIR','Wrong direction inserted: {} / [0,1,"in","out"]\n'.format(direction))
    
##---------------------------------------------------------------------------------- a03 print properties --------------------------
    def get_chassis_and_slot(self):
        """ This function returns the board chassis and slot number. """
        return self._chassis,self._slot
        

    def get_channel_parameters(self,Channel):
        """This function returns the specified Channel dictionary that contains
        the channel configuration."""  
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel]
        
        tmp.print_channel_parameters()
        
    def parameters_dictionary(self):
        """This function returns a dictionary with all the channels parameters."""  
        
        full={}
        for i in range(4):
            tmp = self.__channels[i].pars_dict
            full.update({'ch{}'.format(i):tmp})
        
        return full
            
        
        

    def plot_queue(self,Channel):
        """This function plots the wave that have been queue in the specified 
        channel, this is an elaboration that takes in account what is in the pc
        side, taking in account delays and prescaler. Please check on a scope
        when having doubts about the reproduced wave."""
        Channel = int(Channel)
        self.__check_channel_number(Channel)
        
        
        queue = self.__channels[Channel].pars_dict['Queue_list']
        totalx,totaly = [],[]
        tmpx=0
        for q in queue:
            tmp = self.get_wave_by_ID(q['WaveID'])
            if q['Delay']!=0:
                totaly.append(self.np.zeros(int(q['Delay']/self.SF) ) ) #creates delay
                totalx.append(self.np.arange(tmpx,tmpx+len(totaly[-1])*self.SF,1/self.SF))
                tmpx = totalx[-1][-1]+1/self.SF
            
            totaly.append(tmp) #appends the wave itself after the specified delay
            totalx.append(self.np.arange(tmpx, tmpx+len(totaly[-1])/self.SF*(1+q['Prescaler']), 1/self.SF*(1+q['Prescaler']) ) )
            tmpx = totalx[-1][-1]+1/self.SF
        
        
        self.__check_samples_number_in_memory()
        totalx,totaly = self.np.hstack((totalx)),self.np.hstack((totaly))
        self.plt.plot(totalx,totaly)
        self.plt.xlabel('Time (ns)')
        self.plt.ylabel('Amplitude (norm.)')
        return totalx,totaly
        
    def print_waves_ID(self):
        """This function prints the Wave IDs inserted in the memory"""
        text = ''
        
        for count,IDS in enumerate(self.__waves_ID):
            text+= 'Wave {} ID: {}\n'.format(count,IDS)
        
        print(text)
    
    def print_channel_queue(self,Channel):
        """This function prints the specified channel queue, together with the
        waves IDs, there also the trigger options, delays, ecc."""
        self.__check_channel_number(Channel)
        
        text= str(self.channel(Channel).pars_dict['Queue_list'])
        text = text.replace(',','\n')
        text = text.replace('}','}\n')
        print(text)
#------------------------------------------------------------------------------------ a04 memory related funcs ---------------------

    def insert_array(self,Array,ID):
        """This function insert an array of normalized numbers in the AWG memory
        and assign an ID to it.
        
        NOTE: this function acts only on the class, use the function load_waves_in_AWG_memory()
        to load waves in the AWG memory. Also note that the waves are automatically loaded when
        AWG is started with the functio Start_multiple().
        """
        
        self.remove_wave_by_ID(ID,warning=False)
        self.__waves.append(Array)
        self.__check_samples_number_in_memory()
        self.__waves_ID.append( (str(ID)) )
        
        #if self.autoset:
        #    self.load_waves_in_AWG_memory()


    def insert_sequence_object(self,seq,queue_in_channel=None,clear_channel_queue=False):
        """This function can be used to queue a sequence object in a channel,
        the delay is inserted in the queue delay option
        
        - seq: sequence object from the PULSE library
        - queue_in_channel: the channel that is used, if it is None (def), the
        waves will only be loaded in the memory
        - clear_channel_queue: if it is True (not def), the channel queue will
        be cleaned before
        
        """
        
        if clear_channel_queue and queue_in_channel is not None:
                self.clear_channel_queue(queue_in_channel)
        
        ids,delays,pulses = seq.generate_sequence_for_AWG()
        for i in range(len(pulses)):
            #insert wave in AWG memory
            
            self.remove_wave_by_ID(ids[i],False)
            self.__waves.append(pulses[i])
            self.__check_samples_number_in_memory()
            self.__waves_ID.append( ids[i] )
        
            if queue_in_channel is not None:
                
                self.queue_in_channel(queue_in_channel,ids[i],0,delays[i],1,0)
        

    def insert_sequence_object_IQ(self,seq,cal_dict,amplitude= 0,clear_channels_queue=False):
        """This function can be used to queue a sequence object in the pair of
        channels chI,chQ=chI+1 (specified in the cal_dict created with the 
        IQCalibration library). The calibration will be applied to the channels,
        the channels will be set to AWG mode and the phase of the sequence
        on channel Q wil also be set automatically.
        
        - seq: sequence object from the PULSE library
        - cal_dict: calibration dictionary created with IQCalibration
        - amplitude (V, def 0V): the voltage to be applied to the channel pair
        - clear_channel_queue: if it is True (not def), the channels queue will
        be cleaned before
        
        """
        
        chI = cal_dict['AWG chI']
        
        
        if clear_channels_queue:
            self.clear_channel_queue((1<<chI)+(1<<(chI+1)))
            
        
        self.apply_correction(cal_dict,amplitude,0)
        
        seq.pl[0].phase(self.phase(chI))
        ids,delays,pulses0 = seq.generate_sequence_for_AWG()
        
        seq.pl[0].phase(self.phase(chI+1))
        ids,delays,pulses90 = seq.generate_sequence_for_AWG()
        
        for i in range(len(pulses0)):
            #insert wave in AWG memory
            ids_Q = ids[i]+'_Q'
            self.remove_wave_by_ID(ids[i],False)
            self.remove_wave_by_ID(ids_Q,False)
            
            self.__waves.append(pulses0[i])
            self.__waves.append(pulses90[i])
            
            self.__check_samples_number_in_memory()
            self.__waves_ID.append( ids[i] )
            self.__waves_ID.append( ids_Q )
            
            if i==0:
                trig=6
            else:
                trig=0
                
            self.queue_in_channel(chI,ids[i],trig,delays[i],1,0)
            self.queue_in_channel(chI+1,ids_Q,trig,delays[i],1,0)
        
        self.mode(chI,'AWG')
        self.mode(chI+1,'AWG')
        
        
        
        #self.set_channel((1<<chI)+(1<<(chI+1)),True) #Done when starting the AWG
        
        
        
    def get_queue_length(self,Channel):
        """This function returns the full length of the queue of the specified
        channel in ns.
        The functions takes in account the prescalers and the delays.
        
        Channel can be 'a' or 'all', in this case a list with all the channels 
        queue lengths will be returned.
        
        """
        if Channel == 'a' or Channel == 'A':
            tmp=[]
            for i in range(4):
                tmp.append(self.get_queue_length(i))
            return tmp
        
        else:
            self.__check_channel_number(Channel)
            
            ql = self.channel(Channel).pars_dict['Queue_list']
            delays = self.np.ndarray(len(ql))
            widths = self.np.ndarray(len(ql))
            prescaler= self.np.ndarray(len(ql))
            for i,el in enumerate(ql):
                delays[i]= el['Delay']
                widths[i]= len(self.get_wave_by_ID(el['WaveID']))
                prescaler[i] = el['Prescaler']+1
            return self.np.sum(delays)+self.np.sum(widths*prescaler)            
    
    
    def get_wave_by_number(self,Number=None):
        """This function return the array stored in the memory, given the 
        number"""
        if Number is None:
            return len(self.__waves)
        
        self.__check_wave_number(Number)
            
        return self.__waves[Number]
    
    def get_wave_by_ID(self,ID):
        """This function return the array stored in the memory, given the 
        ID"""
        ID = str(ID)
        
        try:
            tmp = self.__waves_ID.index(ID)
            return self.__waves[tmp]
        except ValueError:
            print('Wrong ID inserted\n')
            raise Exception('IDERR')
    
    def clear_channel_queue(self,Channel_mask):
        """This function clears the queue of the channels specified in the
        Channel_mask."""
        
        if Channel_mask<0 or Channel_mask>15:
            print('Wrong channel mask inserted: [0:15]\n')
            raise ValueError
        
        for i in range(4):
            if (Channel_mask>>i)%2:
                #self.__check_channel_number(i)
                self.__channels[i].clear_queue()
        
                if self.autoset:
                    self.__check_command(self._awg.AWGflush(i))
    
    def queue_in_channel(self,Channel,wave_ID,TriggerMode=0,Delay=0,Repetitions=1,Prescaler=0):
        """This function queue the wave, specified by the ID in the specified Channel.
        
        check the relative print function for the TriggerModes.
        Delay is in ns, Repetitions is the number of times the wave will be repeated.
        
        prescaler divides the max sampling frequency by (1+prescaler)
        
        NOTE: this function acts only on the class, use the function load_waves_in_channel()
        to manually queue waves after loading the AWG memory. This operation is automatically
        done when using start_multiple() command, on the channels selected by the ChannelMask.
        
        
        """
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel]
        
        if self.SF==0.5: #bugfix on old card
            Delay *=10
        
        
        tmp.insert_in_queue(wave_ID,TriggerMode,Delay,Repetitions,Prescaler)
        #if self.autoset:
        #    self.queue_waves_in_channel(Channel)
    
    def clear_waves(self):
        """This function wipes all waves in the memory. NOTE: doesn't clear the
        queues, so waves can be reloaded without queueing them again."""
        self.__waves = []
        self.__waves_ID = []
        #self.__check_command( self._awg.waveformFlush())

    def remove_wave_by_ID(self,ID,warning=True):
        """This function erases a wave given an ID. 
        If the ID is not found will return a Warning by def.
        
        NOTE: this function acts only on the class, use the function load_waves_in_AWG_memory()
        to load waves in the AWG memory. Also note that the waves are automatically loaded when
        AWG is started with the functio Start_multiple().
        """
        ID = str(ID)
        
        try:
            tmp = self.__waves_ID.index(ID)
            self.__waves_ID.pop(tmp)
            self.__waves.pop(tmp)
        except:
            if warning:
                print('ID not found')
                
        #if self.autoset:
        #    self.load_waves_in_AWG_memory()
            
#------------------------------------------------------------------------------------- a05 action functions -----------------------

    def queue_waves_in_channel(self,Channel):
        
        
        self.__check_command( self._awg.AWGflush(Channel) )
        queue = self.channel(Channel).pars_dict['Queue_list']
        if len(queue)>0:
            for q in queue:
                wave_number = self.__waves_ID.index(q['WaveID'])
                self.__check_command( self._awg.AWGqueueWaveform(Channel,
                                                                 wave_number,
                                                                 self.channel(Channel).trigger_modes_list.index(q['TriggerMode']),
                                                                 int(q['Delay']/10),
                                                                 q['Repetitions'],
                                                                 q['Prescaler']))#NOTE: delay/10 for AWG, 
        
        #self.time.sleep(0.1)

    def load_waves_in_AWG_memory(self):
        """With this function the waves inserted will be effectively loaded
        in the AWG memory."""
        
        self.__check_command( self._awg.waveformFlush())
        
        for n in range(len(self.__waves)):
            tmp = self.get_wave_by_number(n)
            wave = self.sigd.SD_Wave()
            err = wave.newFromArrayDouble(0,tmp)
            if err<0:
                raise self.__AWGCMDERR('MEMERRR','Error in loading waves into memory: {}\n',err)
            
            self.__check_command( self._awg.waveformLoad(wave,n))
            del wave #bugfix
        
        for i in range(4):
            self.queue_waves_in_channel(i)
        #self.time.sleep(0.1)
    
    def __change_offset_slow(self,channel):
    
        #self.mode(ch0,'AWG')
        #exp.awg.mode(ch1,'AWG')
        
        value,step,delay = self.offset(channel)
        if delay == 0:
            self.__check_command( self._awg.channelOffset(channel,self.np.round(value,10 ) ))
            self.__prev_offset[channel] = value
        else:
        
            start_voltage = self.__prev_offset[channel]
            if start_voltage > value:
                step = -abs(step)
                direction_up = False #False is down
            else:
                step = abs(step)
                direction_up = True #True is up
                
            tmp = self.np.round(start_voltage+step,10)
            while(1):
                
                
                self.__check_command( self._awg.channelOffset(channel,tmp ) )
                self.time.sleep(delay)
                tmp += step
                if direction_up is False:
                    if tmp<value:
                        self.__check_command( self._awg.channelOffset(channel,value ))
                        break
                else:
                    if tmp>value:
                        self.__check_command( self._awg.channelOffset(channel,value ))
                        break
            
            self.__prev_offset[channel]=value
            
        
    def channel(self,num=None):
        """This function returns the subclass channel object, the user should 
        usually not need this function."""  
        if num is None:
            print('Specify a channel number: [0:3]\n')
        return self.__channels[int(num)]
        
        
    def set_channel(self,ChannelMask=15,load_waves_in_memory = False):
        """When not using autoset (def is ON) or if the user thinks that the board
        configuration doesn't match the dictionary, this function can be used to
        send the channel parameters to the board.
        
        ChannelMask is an integer mask that tells the function which channel 
        need to be set up, examples:
        
        - 1 is channel 1
        - 2 is channel 2
        - 3 is channel 1 and 2 (in binary it is 0011)
        - 15 or 0xf is all of them (in binary it is 1111)
          
        load_waves_in_memory is False as default, when it is True the waves will
        be loaded in the awg memory. The user has to be aware that unfortunately
        there is no error if waves are queued but not in the AWG memory.
        
        """ 
        if ChannelMask<0 or ChannelMask>15:
            print('Wrong channel mask inserted: [0:15]\n')
            raise ValueError
            
        def setfunc(Channel):
            
            self.__check_channel_number(Channel)
            tmp = self.channel(Channel)
            
            self.__check_command( self._awg.channelWaveShape(Channel,tmp.wave_modes_list.index( tmp.mode() )))
            
            #Offset
            self.__change_offset_slow(Channel)
            
            
            self.__check_command( self._awg.channelFrequency(Channel,self.np.round(tmp.frequency()*1e6,12) ) )
            self.__check_command( self._awg.channelPhase(Channel,tmp.phase() ) )
            
            self.phase_reset()
            
            #ext trig
            pars = tmp.ext_trigger_config()
            pars = pars['PORT'],pars['EDGE'],pars['Sync']
            port_sn = tmp.ext_port_list.index(pars[0])
            if port_sn>0:
                port_sn= int(port_sn+3999) #flags for ext port
            self.__check_command(self._awg.AWGtriggerExternalConfig(Channel,port_sn,tmp.trigger_edge_list.index(pars[1]),int(pars[2])))

            
            
            #queue mode
            self.__check_command(self._awg.AWGqueueConfig(Channel,tmp.queue_mode_list.index(self.queue_mode(Channel))))
            
            if tmp.modulation() == 1:
                self.__check_command( self._awg.channelAmplitude(Channel,0 ) )
                self.__check_command( self._awg.modulationAmplitudeConfig( Channel,1,tmp.amplitude() )  )
                
            else:
                self.__check_command( self._awg.channelAmplitude(Channel,tmp.amplitude() ) )
                self.__check_command( self._awg.modulationAmplitudeConfig( Channel,0,0 )  )
                
            self.queue_waves_in_channel(Channel)
            
            
        if load_waves_in_memory:
                self.load_waves_in_AWG_memory()
        for i in range(4):
            if (ChannelMask>>i)%2:
                setfunc(i)
        
    def start_multiple(self,ChannelMask = 0xF):
        
        """Before starting any acquisition, the channel must be started. After
        the start, the board will wait for a trigger event or just record as 
        soon as possible if it is in auto mode.
        
        Channel_mask is an integer mask that tells the function which channel 
        need to be set up, examples:
        
        - 1 is channel 1
        - 2 is channel 2
        - 3 is channel 1 and 2 (in binary it is 0011)
        - 15 or 0xf is all of them (in binary it is 1111)
        
        NOTE: waves will be flushed, then loaded in the AWG and queued in the 
        channels selected in the ChannelMask.
        """
        self.set_channel(ChannelMask,True)
        self.__check_command(self._awg.AWGstartMultiple(ChannelMask))        
 

    def stop_multiple(self,ChannelMask = 0xF):
        """Command used to stop the AWG from playing waves.
        
        Channel_mask is an integer mask that tells the function which channel 
        need to be set up, examples:
        
        - 1 is channel 1
        - 2 is channel 2
        - 3 is channel 1 and 2 (in binary it is 0011)
        - 15 or 0xf is all of them (in binary it is 1111)
        """
        self.__check_command(self._awg.AWGstopMultiple(ChannelMask))   

    def sw_trigger(self,ChannelMask=0xf):
        """ This function is used to trigger a channel when its trigger is set
        to software trigger.
        
        Channel_mask is an integer mask that tells the function which channel 
        need to be set up, examples:
        
        - 1 is channel 1
        - 2 is channel 2
        - 3 is channel 1 and 2 (in binary it is 0011)
        - 15 or 0xf is all of them (in binary it is 1111)
        
        """
        self.__check_command( self._awg.AWGtriggerMultiple(ChannelMask ) )
       
    def register(self,Number,Value=None):
        """This function sets or gets (when Value is None, def) a 32 bit integer
        number to/from the specified board register [0:15] """
        Number = int(Number)
        if Number<0 or Number>15:
            print('Wrong register number inserted: {} / [0,15]\n'.format(int(Number)))
            raise Exception(ValueError)
            
        if Value is None:
            return self._awg.readRegisterByNumber(Number)[1]
        else:
            self._awg.writeRegisterByNumber(Number,int(Value))
        
    

    
    def phase_reset(self,ChannelMask=0xf):
        """This function reset the oscillators phase of the specified channels (all by def).
        It is good to reset the phase after setting the oscillators frequency.
        
        Channel_mask is an integer mask that tells the function which channel 
        need to be set up, examples:
        
        - 1 is channel 1
        - 2 is channel 2
        - 3 is channel 1 and 2 (in binary it is 0011)
        - 15 or 0xf is all of them (in binary it is 1111)
        
        """
        self.__check_command(self._awg.channelPhaseResetMultiple(ChannelMask))
        
    def apply_correction(self,cal_dictionary,amplitude=None, set_channel=True):
        """ This function can be used to load an IQ mixer calibration file in 
        the channels specified in the same file. The amplitude can be specifed
        here, if it is none, the channel I amplitude that is set will be used.
        It is a good thing to always use this function to set different 
        amplitudes.
        
        if set_channel is True (def), the settings will be loaded in the AWG.
        if autoset is used, set_channel value is ignored.
        
        """
        
        
        #a = load_calibration_file(filename)
        
        freq,chI,chQ = cal_dictionary['AWG frequency'],cal_dictionary['AWG chI'],cal_dictionary['AWG chQ']
        corr_par = [cal_dictionary['Offset chI'],cal_dictionary['Offset chQ'],cal_dictionary['Amplitude ratio'],cal_dictionary['Amplitude corrected channel'],cal_dictionary['Phase correction chQ']]
        
        self.frequency(chI,freq)
        self.frequency(chQ,freq)
        
        #off0,off1,amp,ph,amp_chan
        self.offset(chI,corr_par[0],0,0) #no need to be slow
        self.offset(chQ,corr_par[1],0,0) #no need to be slow
        
        if amplitude is None:
            amplitude = self.__channels[chI].amplitude()
        
        if corr_par[3] == 'chI':
            self.amplitude(chI,amplitude*corr_par[2])
            self.amplitude(chQ,amplitude)
        elif corr_par[3] == 'chQ':
            self.amplitude(chQ,amplitude*corr_par[2])
            self.amplitude(chI,amplitude)
        else:
            print('ERROR: calibration amplitude is not valid for this channel: '+str(corr_par[3]))
            raise Exception('CHERR')
        
        
        
        self.phase(chI,0)
        if cal_dictionary['Sideband']=='RSB':
            self.phase(chQ, self.np.round(90.+corr_par[4],9) ) 
        else:
            self.phase(chQ, self.np.round(270.+corr_par[4],9) ) 
        
        mask = (1<<chI)
        mask += (1<<chQ)
        
        
        if set_channel and not self.autoset:
            self.set_channel(mask)
         
        self.phase_reset(mask)
        
        
        
        #self.time.sleep(0.01)
    
##------------------------------------------------------------------------------- a06 saving functions -----------------------
    
 
    def save_waves(self,fname,Force=False):
        """save the waves in a file, using np.savez
            
            fname: path and filename where to save
            Force (def False): overwrite
        """
        import os
        
        
        
        # Split fname in folder and filename
        path = os.path.split(fname)

        # Create directory if not already existent
        if not os.path.exists(path[0]) and path[0]:
            os.makedirs(path[0])

        file_name = path[1]

        
        # Append Folder and be adaptive to windows, etc.
        file_name = os.path.normpath(os.path.join(path[0], file_name))

        # Check for Overwrite
        if not Force:
            if os.path.isfile(file_name):
                print('File already exists\n')
                raise Exception('FILEEXISTSERR')
        
        self.np.savez(file_name,waves=self.__waves,IDs=self.__waves_ID)
            
    def load_waves(self,filename):
        """load the waves from a npz file to the memory"""
        with open(filename, 'rb') as f:
            self.__waves=f['waves']
            self.__waves_ID=f['IDs']
        


