#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:20:31 2017

@author: oscar
v2.7.0 - OSC:
    - changed dev id
    
v2.6.0 - OSC:
    - changed some dictionary key name
    - one function was missing for the use of an external trigger
    
v2.5.0 - OSC:
    - renamed some function
    - inserted trigger direction function in the main class
    
v2.4.0 - OSC:
    - written help in the functions
    - fixed a bug with set_channel function
    - moved the print functions in the main class
    
v2.3.1 - OSC:
    - improved error printout
    
v2.3.0 - OSC:
    - "protected" the channel dictionary
    - implemented autoset
    - changed the set_channel Channel parameter to a mask
    
v2.2.0 - OSC:
    - added a number to the channels, it is a parameter that cannot be changed
    
v2.1.0 - OSC:
    - channels is now a function that return the channel, it is not anymore a
    list, safer

v2.0.1 - OSC:
    - chassis and slot are now stored
    - inserted a function to get chassis and slot
    
v2.0.0 - OSC:
    - using dictionaries to get and set channels parameters
    
v1.1.1 - OSC:
    - it seems that the delay in DAQconfig is in samples and not a multiple of 
      10ns, check on the manual

v1.1.0 - OSC:
    - it is not possible to change the channel amplitude continuously, there is a list of values that is function of the impedance,
    I added two lists and the user can only set those values.
"""


class DIGChannel(object):
    import numpy as np
    import copy
    
    def __init__(self, max_freq,number):
        self.__FR_MAX = max_freq
        self.__AMP_LIST_50 = [0.0625,0.125,0.25,1,2,4]
        self.__AMP_LIST_HIZ = [0.1,0.2,0.4,0.8,2,4,8]
        if number<1 or number>4:
            print('Wrong channel number inserted\n')
            raise ValueError
        
        self.__chn = int(number)
        
        self.__imp_list = ['HIZ','50']
        self.__coup_list = ['DC','AC']
        self.__edge_list = ['RISING','FALLING','BOTH']
        self.__trig_mod = ['AUTO','SW','EXT','ANALOG','', 'SW_CYCLE','EXT_CYCLE','ANALOGAUTO']
        self.__ext_trig = ['EXT','PXI0','PXI1','PXI2','PXI3','PXI4','PXI5','PXI6','PXI7']
        
        
        
        
        
        
        self.__pars_dict={'Channel number': self.__chn,
                        'Impedance':'50',
                        'Amplitude' : 1,
                        'Coupling' : 'AC',
                        'Analog trigger threshold': 0.1,
                        'Trigger edge' : 'RISING',
                        'External trigger type': 'EXT',
                        'Prescaler' : 0,
                        'Trigger_mode':'AUTO',
                        'Delay':0,
                        'Points':0,
                        'Cycles':1
                }
    
#--------------------------------------------------------------------------------- Properties defs -------------------------    
    @property
    def impedance_list(self):
        return self.copy.deepcopy(self.__imp_list)
    
    @impedance_list.setter
    def impedance_list(self,value):
        pass
    
    @property
    def coupling_list(self):
        return self.copy.deepcopy(self.__coup_list)
    
    @coupling_list.setter
    def coupling_list(self,value):
        pass

    @property
    def edge_list(self):
        return self.copy.deepcopy(self.__edge_list)
    
    @edge_list.setter
    def edge_list(self,value):
        pass
 
    @property
    def trigger_mode_list(self):
        return self.copy.deepcopy(self.__trig_mod)
    
    @trigger_mode_list.setter
    def trigger_mode_list(self,value):
        pass

    @property
    def amplitude_mode_list(self):
        if self.__pars_dict['Impedance']=='50':
            return self.copy.deepcopy(self.__AMP_LIST_50)
        else:
            return self.copy.deepcopy(self.__AMP_LIST_HIZ)
    
    @amplitude_mode_list.setter
    def amplitude_mode_list(self,value):
        pass

    @property
    def trigger_direction_list(self):
        return self.copy.deepcopy(self.__trig_dir)
    
    @trigger_direction_list.setter
    def trigger_direction_list(self,value):
        pass

    @property
    def ext_trigger_type_list(self):
        return self.copy.deepcopy(self.__ext_trig)
    
    @ext_trigger_type_list.setter
    def ext_trigger_type_list(self,value):
        pass


    @property
    def pars_dict(self):
        return self.copy.deepcopy(self.__pars_dict)
    
    @pars_dict.setter
    def pars_dict(self,value):
        pass

#-------------------------------------------------------------------------------- Check functions -------------------------------    
    def __check_amplitude(self,value):
        if self.pars_dict['Impedance'] == 0:
            lst = self.__AMP_LIST_HIZ
        else:
            lst = self.__AMP_LIST_50
            
        try:
            lst.index(value)
        except ValueError:
            print('wrong amplitude inserted: {}\n'.format(value))
            self.print_amplitude_list()
            raise Exception('AMPERR')
            
        
    

    
#-------------------------------------------------------------------------------- Set/get functions --------------------  
    def amplitude(self, Amplitude=None):
        
        if self.pars_dict['Impedance'] == 'HIZ':
            lst = self.__AMP_LIST_HIZ
        else:
            lst = self.__AMP_LIST_50
        
        if Amplitude is None:
            self.__check_amplitude(self.pars_dict['Amplitude'])
            return self.pars_dict['Amplitude']
        
        
        try:
            lst.index(Amplitude)
        except ValueError:
            print('Wrong Amplitude inserted: {} V\n'.format(Amplitude))
            self.print_amplitude_list()
            raise ValueError
            
        
        self.__pars_dict['Amplitude'] = self.np.round(Amplitude,12)    
        
    
    def coupling(self,Coupling=None):
        if Coupling is None:
            return self.pars_dict['Coupling']
        
        if type(Coupling) is str:
            Coupling = Coupling.upper()
            
            try:
                self.coupling_list.index(Coupling)
            except ValueError:
                print('Wrong Coupling inserted, use print_coupling_list() to check available modes\n')
                raise 
        else:
            try:
                Coupling=self.coupling_list[int(Coupling)]
            except IndexError:
                print('Wrong Coupling inserted, use print_coupling_list() to check available modes\n')
                raise
                
        self.__pars_dict['Coupling'] = Coupling
    
    def impedance(self,Impedance=None):
        if Impedance is None:
            return self.pars_dict['Impedance']
        
        if type(Impedance) is str:
            Impedance = Impedance.upper()
            
            try:
                Impedance = self.impedance_list.index(Impedance)
            except ValueError:
                print('Wrong Impedance inserted, use print_impedance_list() to check available modes\n')
                raise
        else:
            try:
                Impedance = self.impedance_list[int(Impedance)]
            except IndexError:
                print('Wrong Impedance inserted, use print_impedance_list() to check available modes\n')
                raise
                
        self.__pars_dict['Impedance'] = Impedance      
        
        
    def analog_trigger(self,Value=None,Edge=1):
        if Value is None:
            return self.pars_dict['Analog trigger threshold'],self.pars_dict['Trigger edge']
        
        if Value < self.__AMP_MIN or Value > self.__AMP_MAX:
            print('Analog trigger value is too high: [{},{}]\n'.format(self.__AMP_MIN,self.__AMP_MAX))
            raise ValueError
            
        if type(Edge) is str:
            Edge = Edge.upper()
            
            try:
                Edge = self.edge_list.index(Edge)
            except ValueError:
                print('Wrong Edge inserted, use print_edge_list() to check available modes\n')
                raise
        else:
            try:
                Edge = self.edge_list[int(Edge)]
            except IndexError:
                print('Wrong Edge inserted, use print_edge_list() to check available modes\n')
                raise
        
        self.__pars_dict['Analog trigger threshold'] = Value
        self.__pars_dict['Trigger edge'] = Edge

    def external_trigger(self,Type=None,Edge=1):
        if Type is None:
            return self.pars_dict['External trigger type'],self.pars_dict['Trigger edge']
        
        if type(Type) is str:
            Type = Type.upper()
            
            try:
                self.ext_trigger_type_list.index(Type)
            except ValueError:
                print('Wrong Type inserted, use print_external_trigger_type_list() to check available types\n')
                raise 
        else:
            try:
                Type = self.ext_trigger_type_list[int(Type)]
            except IndexError:
                print('Wrong Type inserted, use print_external_trigger_type_list() to check available types\n')
                raise
            
        if type(Edge) is str:
            Edge = Edge.upper()
            
            try:
                Edge = self.edge_list.index(Edge)
            except ValueError:
                print('Wrong Edge inserted, use print_edge_list() to check available modes\n')
                raise
        else:
            try:
                Edge = self.edge_list[int(Edge)]
            except IndexError:
                print('Wrong Edge inserted, use print_edge_list() to check available modes\n')
                raise
        
        self.__pars_dict['External trigger type'] = Type
        self.__pars_dict['Trigger edge'] = Edge


    def prescaler(self,Prescaler = None):
        if Prescaler is None:
            return self.pars_dict['Prescaler']
        
        if Prescaler <0:
            print('Prescaler cannot be negative\n')
            raise ValueError
        
        self.__pars_dict['Prescaler'] = Prescaler
        
    def trigger_mode(self,Mode=None):
        if Mode is None:
            return self.pars_dict['Trigger_mode']
        
        if type(Mode) is str:
            Mode = Mode.upper()
            
            try:
                self.trigger_mode_list.index(Mode)
            except ValueError:
                print('Wrong Mode inserted, use print_trigger_mode_list() to check available modes\n')
                raise 
        else:
            try:
                Mode = self.trigger_mode_list[int(Mode)]
            except IndexError:
                print('Wrong Mode inserted, use print_trigger_mode_list() to check available modes\n')
                raise
                
        self.__pars_dict['Trigger_mode'] = Mode     


    def delay(self,Delay = None):
        if Delay is None:
            return self.pars_dict['Delay']
        
        if Delay <0:
            print('Delay cannot be negative\n')
            raise ValueError
        
        self.__pars_dict['Delay'] = Delay
        
    def points(self,Points = None,Cycles = 1):
        if Points is None:
            return self.pars_dict['Points'],self.pars_dict['Cycles']
        
        if Points <0:
            print('Points cannot be negative\n')
            raise ValueError
        
        if Cycles <1:
            print('Cycles cannot be less than 1\n')
            raise ValueError
        
        self.__pars_dict['Points'] = int(Points)
        self.__pars_dict['Cycles'] = int(Cycles)

    
        

        
 
    



##------------------------------------------------------------------------------------------------------

class Digsigd(object):
    import keysightSD1 as sigd
    version = '2.7.0'
    
    def __init__(self,slot,chassis=0,autoset=True,init_dig=True):
        """
        Initialize the Digsigd class by specifying the slot number and the
        chassis number (def is 0). 
        
        - autoset: True by default, when changing a channel parameter, it will
        be set on the board as soon as possible. If it is False, the user needs
        to call the set_channel function to send all the parameters and waves
        to the board.
        
        - init_dig: True by def, it initialize the awg to the default settings
        when the class is initialized, otherwise only the dictionary will 
        contain the default settings, but the board is untouched.
        
        """ 
        self.__FR_MAX = 200
        
        self.__SF = 0.5 #GS/s
        
        tmp = self.sigd.SD_AIN()
        self.__check_command(tmp.openWithSlot('',chassis,slot) )
        self._dig = tmp
        
        
        
        self.__channels = []
        
        for i in range(4):
            self.__channels.append(DIGChannel(self.__FR_MAX,i+1))
        
        self.__autoset = True
        self.__trigger_dir = 1 # 1 in IN
        self.__trig_dir_list = ['OUT','IN']
        
        
        if init_dig:
            self.set_channel(15)
        
        self._chassis = chassis
        self._slot = slot
        self.id = 'DIGKEY'
        

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


        
#------------------------------------------------------------------------------- Custom exceptions ------------------        
    class __BASEEXC(Exception):
            pass
        
    class __DIGEXC(__BASEEXC):
        def __init__(self, expression, message):
                self.expression = expression
                self.message = message
#------------------------------------------------------------------------------- Check functions ------------------        
    def __check_channel_number(self,Channel):
        if Channel< 1 or Channel >4:
            print('Wrong Channel number inserted: [1,4]\n')
            raise ValueError
    
    def __check_command(self,cmd):
        if cmd <0:
            #print('ERROR in the command: {}\n'.format(cmd))
            raise self.__DIGEXC(str(cmd),self.sigd.SD_Error.getErrorMessage(cmd))

#-------------------------------------------------------------------------------- Print functions -------------------------------    
    
    def print_impedance_list(self):
        """This function prints the impedance modes that can be used with any 
        channel. The user can use the number or the string""" 
        text = ''
                
        for i,m in enumerate(self.channel(1).impedance_list):
            text+= str(i)+': '+m+'\n'
        print(text)
    
    def print_coupling_list(self):
        """This function prints the coupling types that can be used with any 
        channel. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(1).coupling_list):
            text+= str(i)+': '+m+'\n'
        print(text)
    
    def print_edge_list(self):
        """This function prints the possible modes for the trigger edte. 
        The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(1).edge_list):
            text+= str(i)+': '+m+'\n'
        print(text)

    def print_external_trigger_type_list(self):
        """This function prints the available external trigger modes for any 
        channel. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(1).ext_trigger_type_list):
            text+= str(i)+': '+m+'\n'
        print(text)
    
    def print_trigger_mode_list(self):
        """This function prints the available trigger modes for any 
        channel. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.channel(1).trigger_mode_list):
            text+= str(i)+': '+m+'\n'
        print(text)

    def print_trigger_direction_list(self):
        """This function prints the available trigger directions for the 
        trigger port. The user can use the number or the string""" 
        text = ''
        
        for i,m in enumerate(self.__trig_dir_list):
            text+= str(i)+': '+m+'\n'
        print(text)
    
    
    def print_amplitude_list(self,channel):
        """This function prints the full scale amplitude (Vpp) that can be used 
        with any channel. It is a function of the impedance used.""" 
        
        lst_amp = self.channel(1).amplitude_mode_list
        self.__check_channel_number(channel)
        
        if self.channel(channel).pars_dict['Impedance']=='50':
            text = '50 Ohm:\n'
        else:
            text = 'High impedance:\n'
        
        for i,m in enumerate(lst_amp):
            text+= str(m)+' - '
        print(text[:-2]+'\n\n')
        
        

#-------------------------------------------------------------------------------- Set/Get functions    
    def SF(self):
        """Sampling frequency in GS/s""" 
        return self.__SF
    
    


    
    def amplitude(self, Channel,Amplitude=None):
        """This function sets the amplitude for the specified channel, if it is 
        None (def) it will be queried. Check print functions for the available
        values.""" 
        self.__check_channel_number(Channel)
        
        tmp = self.__channels[Channel-1]
        
        if Amplitude is None:
            return tmp.amplitude(Amplitude)
        else:
            tmp.amplitude(Amplitude)
            
        if self.autoset:
            self.__check_command( self._dig.channelInputConfig(Channel,tmp.amplitude(),tmp.impedance_list.index(tmp.impedance()),tmp.coupling_list.index(tmp.coupling() ) ) )
            
    def coupling(self, Channel,Coupling=None):
        """This function sets the coupling for the specified channel, if it is 
        None (def) it will be queried. Check print functions for the available
        values.""" 
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        if Coupling is None:
            return tmp.coupling(Coupling)
        else:
            tmp.coupling(Coupling)
            
        if self.autoset:
            self.__check_command( self._dig.channelInputConfig(Channel,tmp.amplitude(),tmp.impedance_list.index(tmp.impedance()),tmp.coupling_list.index(tmp.coupling() ) ) )
    
    def impedance(self, Channel,Impedance=None):
        """This function sets the impedance for the specified channel, if it is 
        None (def) it will be queried. Check print functions for the available
        values.""" 
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        if Impedance is None:
            return tmp.impedance(Impedance)
        else:
            tmp.impedance(Impedance)
            
        if self.autoset:
            self.__check_command( self._dig.channelInputConfig(Channel,tmp.amplitude(),tmp.impedance_list.index(tmp.impedance()),tmp.coupling_list.index(tmp.coupling() ) ) )
    
    def delay(self, Channel,Delay=None):
        """This function sets the delay for the specified channel, if it is 
        None (def) it will be queried. Check print functions for the available
        values.""" 
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        if Delay is None:
            return tmp.delay(Delay)
        else:
            tmp.delay(Delay)
        
        if self.autoset:
            self.__check_command( self._dig.DAQconfig(Channel,tmp.points()[0],tmp.points()[1], int(tmp.delay()*self.SF()),tmp.trigger_mode_list.index( tmp.trigger_mode())  )) #delay is in samples, check on the manual
    
    def points(self, Channel,Points=None,Cycles=1):
        """This function sets the number of points and Cycles (def 1) for the 
        specified channel, if it is None (def) it will be queried. The query 
        returns (points,cycles).""" 
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        if Points is None:
            pts = tmp.points()
            return pts[0],pts[1]
        else:
            tmp.points(Points,Cycles)
       
        if self.autoset:
            self.__check_command( self._dig.DAQconfig(Channel,tmp.points()[0],tmp.points()[1], int(tmp.delay()*self.SF()),tmp.trigger_mode_list.index( tmp.trigger_mode())  )) #delay is in samples, check on the manual

    
            
    def prescaler(self, Channel,Prescaler=None):
        """This function sets the prescaler for the specified channel, used to
        slow down the sampling rate. The new sampling rate is equal to the 
        digitizer sampling frequency divided by (1+Prescaler). If it is 
        None (def) it will be queried. Check print functions for the available
        values.""" 
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        if Prescaler is None:
            return tmp.prescaler(Prescaler)
        else:
            tmp.prescaler(Prescaler)
            
        if self.autoset:
            self.__check_command( self._dig.channelPrescalerConfig(Channel,tmp.prescaler() ) )

    def trigger(self, Channel,Trigger=None):
        """This function sets the trigger mode for the specified channel, if it 
        is None (def) it will be queried. Check print functions for the available
        values.""" 
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        if Trigger is None:
            return tmp.trigger_mode(Trigger)
        else:
            tmp.trigger_mode(Trigger)            
            
        if self.autoset:
            self.__check_command( self._dig.DAQconfig(Channel,tmp.points()[0],tmp.points()[1], int(tmp.delay()*self.SF()),tmp.trigger_mode_list.index( tmp.trigger_mode())  )) #delay is in samples, check on the manual
    
    def ext_trigger_port_direction(self,Direction=None):
        """This function sets the Direction of the trigger port, check the corresponding
        print function for available modes. If it is None it will be queried."""
        if Direction is None:
            return self.__trig_dir_list[self.__trigger_dir]
        
        if type(Direction) is str:
            Direction = Direction.upper()
            
            try:
                Direction=self.__trig_dir_list.index(Direction)
            except ValueError:
                print('Wrong Direction inserted, use print_trigger_direction_list() to check available modes\n')
                raise 
        else:
            try:
                Direction = int(Direction)
                self.__trig_dir_list[Direction]
            except IndexError:
                print('Wrong Direction inserted, use print_trigger_direction_list() to check available modes\n')
                raise
        
        #can't have exteranl trigger and trigger as an output at the same time
        
        self.__trigger_dir = Direction
        
        if self.autoset:
            self.__check_command(self._dig.triggerIOconfig(self.__trigger_dir))   
    
    def ext_trigger(self,Channel,Type=None,Edge=0):
        """This function sets the external trigger type and Edge used by the 
        specified channel, if it is None (def) it will be queried. Check 
        print functions for the available values.""" 
        
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        if Type is None:
            return tmp.external_trigger()
        else:
            tmp.external_trigger(Type,Edge)       
        
        if self.autoset:    
            #if tmp.trigger_mode() == 2 or tmp.trigger_mode() == 6: #EXT or EXT_CYCLE
            self.__check_command( self._dig.DAQtriggerExternalConfig(Channel,tmp.ext_trigger_type_list.index(tmp.external_trigger()[0]),tmp.edge_list.index(tmp.external_trigger()[1])+1) )
    
    def analog_trigger(self, Channel,Threshold=None,Edge=0):
        """This function sets the Threshold (V) and Edge (def rising) for the 
        specified channel, if it is None (def) it will be queried. Check 
        print functions for the available values.""" 
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        if Threshold is None:
            return tmp.analog_trigger()
        else:
            tmp.analog_trigger(Threshold,Edge)       
        
        if self.autoset:    
            if tmp.trigger_mode() == 3 or tmp.trigger_mode() == 7: #ANALOG and ANALOGAUTO
                self.__check_command( self._dig.channelTriggerConfig(Channel,tmp.edge_list.index(tmp.analog_trigger()[1]),tmp.analog_trigger()[0] ) )
            
    
    
    
    def get_chassis_and_slot(self):
        """ This function returns the board chassis and slot number. """
        return self._chassis,self._slot
    
    def get_channel_parameters(self,Channel):
        """This function returns the specified Channel dictionary that contains
        the channel configuration."""  
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        
        text = str(tmp.pars_dict)
        
        
        print(text.replace(',','\n'))
        
        
    def set_channel(self,Channel_mask=15):
        """When not using autoset (def is ON) or if the user thinks that the board
        configuration doesn't match the dictionary, this function can be used to
        send the channel parameters to the board.
        
        Channel_mask is an integer mask that tells the function which channel 
        need to be set up, examples:
        
        - 1 is channel 1
        - 2 is channel 2
        - 3 is channel 1 and 2 (in binary it is 0011)
        - 15 or 0xf is all of them (in binary it is 1111)
            
        """ 
        
        if Channel_mask<0 or Channel_mask>15:
            print('Wrong channel mask inserted: [0:15]\n')
            raise ValueError
        
        def setfunc(Channel):
            
            
            self.__check_channel_number(Channel)
            tmp = self.__channels[Channel-1]
            
            
            self.__check_command( self._dig.channelInputConfig(Channel,tmp.amplitude(),tmp.impedance_list.index(tmp.impedance()),tmp.coupling_list.index(tmp.coupling() ) ) )
            
            self.__check_command( self._dig.channelPrescalerConfig(Channel,tmp.prescaler() ) )
            
            if tmp.trigger_mode() == 3 or tmp.trigger_mode() == 7:
                self.__check_command( self._dig.channelTriggerConfig(Channel,tmp.edge_list.index(tmp.analog_trigger()[1]),tmp.analog_trigger()[0] ) )
            
            self.__check_command( self._dig.DAQconfig(Channel,tmp.points()[0],tmp.points()[1], int(tmp.delay()*self.SF()),tmp.trigger_mode_list.index( tmp.trigger_mode())  )) #delay is in samples, check on the manual
        
        
        for i in range(4):
            if (Channel_mask>>i)%2:
                setfunc(i+1)
                
                
        

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
        self.__check_command( self._dig.DAQtriggerMultiple(ChannelMask ) )
    
    def get_wave(self,Channel):
        """ This function is used to get the data from the specified Channel at
        the end of the acquisition.
        
        It returns a list of arrays (matrix) of dimension: Cycles x Points
        """
        
        
        self.__check_channel_number(Channel)
        tmp = self.__channels[Channel-1]
        a=[]
        for c in range(tmp.points()[1]):
            wavetmp = self._dig.DAQread(Channel, tmp.points()[0],10) 
            
            a.append( wavetmp)
        return a
                
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
        """
        self.__check_command(self._dig.DAQstartMultiple(ChannelMask))
            
    def register(self,Number,Value=None):
        """This function sets or gets (when Value is None, def) a 32 bit integer
        number to/from the specified board register [0:15] """
        Number = int(Number)
        if Number<0 or Number>15:
            print('Wrong register number inserted: {} / [0,15]\n'.format(int(Number)))
            raise ValueError
            
        if Value is None:
            return self._dig.readRegisterByNumber(Number)[1]
        else:
            self._dig.writeRegisterByNumber(Number,int(Value))
    
    def channel(self,num):
        """This function returns the subclass channel object, the user should 
        usually not need this function."""  
        self.__check_channel_number(num)
        return self.__channels[int(num-1)]