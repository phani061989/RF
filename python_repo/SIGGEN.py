# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:48:09 2014


@author: Oscar, Iman, Christian

v1.5.2 - OSC:
    - inserted the function connection_timeout, it can be used only with ANAPICO signal generators with firmware >0.4.106.

v1.5.1 - OSC:
    - erased the exception catch if no connections are available, the exception couldn't be captured anymore in the previous way. 
    Moreover the print would be annoying if re-trying to connect. This modification has been made especially for the Fanta4 device.

v1.4.1:
    - it is possible now to disable identify message and power warning message (but default is on as before)
    
v1.4.0 (?):
    ???
v 1.3.1:
- Bugfix: used np.round for check due to precission erros

version 1.3.0:
    - added a control check to the function to set the power, if the power that has been set is different from the specified one, an error will occurr
    - added a warning to the power function, it will warn the used if the power is > 0 dBm
    - modified the pulse mode function to make it compatible with ANAPICO, def pulse mode for ANAPICO is external gated
    - scrambled the parameters order of the pulse_mode function to make his use faster with this new settings

Version 1.2.1 (Christian)
- Code cosmetics for PEP Style

Version 1.2.0 (Christian):
- Adapted to Instrument class (now every function of vxi11.Instrument is
  inherited)
- Bug fixes

Version 1.1.1:
- Bug fix for connection loss because of failure in Anapico Firmware
  (CMF Schneider)

Changes in v 1.1.0:
The function ALC without arguments will return the state
The function pulse_trigged has been renamed
The function pulse_triggered without arguments will return the state
The function output has a new option, Check, if it is True the function will
return the output state together with ALC and Pulse state
"""
VERSION = '1.5.2'
print('SIGGEN v' + VERSION)
###############################################################################
import vxi11
import atexit  # BUG fix for not closing Firmware of Anapico devices
import numpy as np


class Siggen(vxi11.Instrument):
    def __init__(self, ip,identify=True):
        self.version = VERSION
        self.ip = ip
        # Initialize device
        
        super(Siggen, self).__init__(ip)
        if identify:
                print(self.identify())
        
        # Bugfix of errornous Anapico firmware
        atexit.register(self.close)

    def cmd(self, str1, arg=None):
        if arg is None:
            self.write(str(str1) + '\n')
        else:
            self.write(str(str1) + ' ' + str(arg) + '\n')

    def v(self):
        """ Downward compatibility """
        return self

    def query(self, str1):
        return self.ask(''.join([str(str1), '?\n']))

    def cmdque(self, string, argument=None):
        """ Sends a query if no argument is given else wise it sends a command
        with the given argument.

        string = (Str) : Command for device
        argument = (Str) : Argument
        """
        if argument is None:
            return self.ask(str(string) + '?\n')
        else:
            return self.write(str(string) + ' ' + str(argument) + '\n')

    def identify(self):
        """ Returns Manufacturer, DeviceType, Firmware Nr., Firmware Nr. """
        return self.query('*IDN')

    def connection_timeout(self,argument=None):
        '''This function sets the VXI connection timeout, it only works
        for ANAPICO signal generator.
        
        argument is in sec, greater or equal to zero, or 'INF'. If it is 
        None (def), it will be queried.
        
        NOTE: 'INF' is the default value in the machine.
        
        '''
        
        if argument is None:
            return self.query('SYST:COMM:VXI:RTMO')
        
        if type(argument) is str:
            argument = argument.upper()
            if argument != 'INF':
                print('Wrong argument inserted')
                raise Exception ('ARGERR')
            
        else:
            argument = int(argument)
            if argument <0:
                print('Wrong argument inserted')
                raise Exception ('ARGERR')
            
        self.cmd('SYST:COMM:VXI:RTMO',argument)
            
            
        

    def abort(self):
        """ This command causes the List or Step sweep in progress to abort.
        Even if INIT:CONT[:ALL] is set to ON, the sweep will not immediately
        re-initiate."""
        str1 = ':ABOR'
        self.cmd(str1)

    def initiate(self):
        """ Sets trigger to the armed state. """
        str1 = ':INIT'
        self.cmd(str1)

    def initiate_cont(self, arg=None):
        """ Continuously rearms the trigger system after completion of a
        triggered sweep.
        arg = ON|OFF|1|0
        """
        str1 = ':INIT:CONT'
        return self.cmdque(str1, arg)

    def output(self, arg=None, Check_all=False, channel=1):
        """ Turns RF output power on/off."""
        if channel == 1:
            str1 = ':OUTP'
        else:
            str1 = ':OUTP'+ str(channel)
        
        try:
            
            
            if arg is None:
                if Check_all is True:

                    output_state = int(self.query(str1))
                    alc_state = int(self.ALC())
                    pulse_state = int(self.pulse_triggered()[0])
                    return output_state, alc_state, pulse_state
                else:
                    return int(self.query(str1))
            else:
                self.cmd(str1, arg)
        except:
            print("Device {}\n does not respond on channel {}"
                   .format(self.identify(), channel))

    def output_blanking(self, arg=None):
        """ ON causes the RF output to be turned off (blanked) during frequency
        changes. OFF leaves RF output turned on (unblanked)."""
        return self.cmdque(':OUTP:BLAN', arg)

    def frequency(self, arg=None, channel=1):
        """ This command sets the signal generator output frequency for the CW
        frequency mode in GHz."""
        try:
            str1 = 'SOUR{}:FREQ'.format(channel)
            if arg is None:
                return float(self.query(str1))/1e9
            else:
                return self.cmd(str1, arg*1e9)
        except:
            print("Device {}\n does not respond on channel {}"
                   .format(self.identify(), channel))

    def frequency_mode(self, arg=None):
        """ This command sets the frequency mode of the signal generator to CW,
        (list) sweep or chirp."""
        return self.cmdque(':FREQ:MODE', arg)

    def frequency_start(self, arg=None):
        """ This command sets the first frequency point in a chirp or step
        sweep.
        """
        return self.cmdque(':FREQ:STAR', arg)

    def frequency_stop(self, arg=None):
        """ This command sets the last frequency point in a chirp or step
        sweep.
        arg = value
        """
        return self.cmdque(':FREQ:STOP', arg)

    def frequency_step(self, arg=None):
        """ This query returns the step sizein Hz for a linear step sweep."""
        return self.cmdque(':FREQ:STEP', arg)

    def frequency_stepLog(self, arg=None):
        """ This query returns the step size (step factor) for a logarithmic
        step sweep.
        """
        return self.cmdque(':FREQ:STEP:LOG', arg)

    def frequency_reference(self, Mode='INT'):
        """ This function will set the frequency reference:
        - Mode can be 'INT' (default) for the internal oscillator frequency
        or 'EXT' for a 10 MHz external reference oscillator
        """

        if Mode.upper() == 'EXT':
            self.cmd(':FREQ:REF:STAT', 'ON')
            self.cmd(':ROSC:SOUR:AUTO', 'OFF')
            self.cmd(':ROSC:SOUR', 'EXT')

            if self.identify()[:7] == 'AnaPico':
                self.cmd(':ROSC:EXT:FREQ', '10MHz')

        else:
            self.cmd(':FREQ:REF:STAT', 'OFF')
            self.cmd(':ROSC:SOUR:AUTO', 'ON')
            self.cmd(':ROSC:SOUR', 'INT')

    def chirp_count(self, arg=None):
        """ This command specifies the number of repetitions for the chirp.
        Set to INF for infinite repetitions.
        arg = INFinite | <val>
        """
        return self.cmdque(':CHIR:COUN', arg)

    def chirp_time(self, arg=None):
        """ Sets the time span for the chirp.
        arg = <val>
        """
        return self.cmdque(':CHIR:TIME', arg)

    def chirp_direction(self, arg=None):
        """ This command sets the direction of the chirp. DU means direction
        down first, then direction up. UD means direction UP first.
        arg = UD|DU|DOWN|UP
        """
        return self.cmdque('CHIR:DIR', arg)

    def pulse_triggered(self, State=None,Pulse_length=1000, Delay=0,mode=0):
        """ command pulse_triggered(Pulse_length,[Delay=0],[State='ON'])

        this command is used to set the generator in pulse mode, the pulse will
        have the specified length and delay (def 0), in nanosecond.
        The Pulse trigger is a TTL signal on the PULSE port of the generator.

        function arguments:
        - Pulse_length (ns): the pulse length
        - Delay (ns, def 0ns): the pulse delay after the trigger
        - mode (def 0): we usually use fixed pulse shape (mode 0, default) or gated (mode 1, the pulse is on while the TTL signal is up)
            NOTE: ANAPICO can only use gated mode,  in gated mode Pulse_length and Delay are ignored
        - State (def 'ON'): can be 'ON' or 1 to activate or 'OFF' or 0 to
        deactivate, if it is 'OFF' other arguments will be ignored
        
        
        If State is None the function will return: State,Length (ns), Delay (ns)
        """
        
        if self.identify().split(',')[0] == 'AnaPico AG':
            mode = 1
        
        if State is None:
            return np.int(self.query(':PULM:STAT')),np.float(self.query(':PULM:INT:PWID')), np.float(self.query(':PULM:INT:DEL'))

        if type(State) == str:
            if State.upper() == 'OFF':
                State = 0
            else:
                State = 1

        if State == 0:
            self.cmd(':PULM:STAT', 'OFF')
        else:
            if mode is 1:
                self.cmd(':PULM:SOUR', 'EXT')
                self.cmd(':PULM:POL', 'NORM')
            elif mode is 0:
                self.cmd(':PULM:SOUR', 'INT')
                self.cmd(':PULM:SOUR:INT', 'TRIG')
                self.cmd(':PULM:INT:PWID', str(Pulse_length)+'NS')
                self.cmd(':PULM:INT:DEL', str(Delay)+'NS')
            else:
                print('ERROR: mode not recognized: {}'.format(mode))
                raise Exception('MODEERR')
            self.cmd(':PULM:STAT', 'ON')

    def power(self, arg=None, channel=1,warning=True):
        """ This command sets the RF output power. """
        try:
            if arg is not None:
                if arg > 0 and warning:
                    print('Warning: HI-power')
                    
                self.cmdque('SOUR{}:POW'.format(channel), arg)
                
                test = np.round(float(self.query('SOUR{}:POW'.format(channel))), 3) 
                if test != np.round(float(arg), 3):
                    self.output(0)
                    print(('Error in setting the power, actual power: {} dBm.' +
                          'Signal generator output is OFF now!').format(test))
                    raise Exception('POWER')
            else:
                return self.cmdque('SOUR{}:POW'.format(channel), arg)
        except:
            print("Device {}\n does not respond on channel {}"
                   .format(self.identify(), channel))

    def ALC(self, State=None):
        """ This command is used to set the ALC mode ON (def) or OFF
        Arguments:
        - State (def 'ON'): can be 'ON' or 1 (def) to activate, can be 'OFF' or
        0 to deactivate
        """
        if State is None:
            return self.query(':SOUR:POW:ALC:STAT')

        if type(State) == str:
            if State.upper() == 'OFF':
                State = 0
            else:
                State = 1

        if State == 0:
            self.cmd(':SOUR:POW:ALC:STAT', 'OFF')
            self.cmd(':SOUR:POW:ALC:SEAR', 'OFF')
        else:
            self.cmd(':SOUR:POW:ALC:STAT', 'ON')
            self.cmd(':SOUR:POW:ALC:SEAR', 'ON')

    
        