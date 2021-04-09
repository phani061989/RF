# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 2016


@author: Michael

"""
VERSION = '1.0.0'
print('BKPSIGGEN v' + VERSION)
###############################################################################
import visa


class Bkpsiggen(object):
    def __init__(self, resID):
        self.version = VERSION
        self.resID = resID
        self._channelOpts = ['C1', 'C2']
        # Initialize device
        try:
            self.rm = visa.ResourceManager()
            self.dev = self.rm.open_resource(resID)
            # bug in device - device reset command *RST only changes buzzer state
            # self.cmd('*RST')            
            print(self.identify()+ '\nDevice reset.')
        except Exception as e:
            print("Connection not possible. Please restart Device and ensure" +
                  " connection.\n")
            print("Details: " + str(e))

    def write(self, str1):
        return self.dev.write(str1)
    
    def query(self, str1):
        return self.dev.query(str1)
    
    def cmd(self, str1, arg=None):
        if arg is None:
            self.write(str(str1) + '\n')
        else:
            self.write(str(str1) + ', ' + str(arg) + '\n')

    def que(self, str1):
        return self.query(str(str1) + '?\n')

    def identify(self):
        """ Returns Manufacturer, DeviceType, Serial Nr., Software version,
        Hardware version."""
        return self.que('*IDN')

    def setOutputState(self, channel=None, arg=None):
        """ Turns output power on/off."""
        if channel in self._channelOpts:        
            outputOpts = ['ON', 'OFF']
            if arg in outputOpts:
                self.cmd(channel + ':OUTP ' + arg)
            else:
                print('Warning: Unknown output state!')
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')

    def getOutputState(self, channel=None):
        """ Returns current output state."""
        if channel in self._channelOpts:        
            return self.que(channel + ':OUTP')
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
    
    def setImpedanceMode(self, channel=None, arg=None):
        """ Sets a channels expected output impedance.
        Options are '50' or 'HZ'."""
        if channel in self._channelOpts:        
            zOpts = ['50', 'HZ']
            if arg in zOpts:
                self.cmd(channel + ':OUTP LOAD', arg)
            else:
                print('Warning: Unknown impedance mode!')
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
    
    def copyChannelConfig(self, sourceChannel=None, destChannel=None):
        """ Copies configurations of source channel to destination channel. """
        if (sourceChannel in self._channelOpts and destChannel in self._channelOpts):        
            self.cmd('PACP ' + destChannel + ', ' + sourceChannel)
        else:
            print('Warning: Unknown Source or Destination Channel! Options are C1 for Channel 1 and C2 for Channel 2.')

    def setFrequency(self, channel=None, arg=None):
        """ This command sets the signal frequency.
        Argument for frequency in units of Hz."""
        if channel in self._channelOpts:            
            str1 = ':BSWV FRQ'
            self.cmd(channel + str1, str(arg))
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')

    def setAmplitude(self, channel=None, arg=None):
        """ This command sets the output voltage amplitude.
        Argument for amplitude in units of V."""
        if channel in self._channelOpts:
            str1 = ':BSWV AMP'        
            self.cmd(channel + str1, str(arg))
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
        
    def setPeriod(self, channel=None, arg=None):
        """ This command sets the period of the signal.
        Argument for period in units of s."""
        if channel in self._channelOpts:        
            str1 = ':BSWV PERI'        
            self.cmd(channel + str1, str(arg))
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
    
    def setOffset(self, channel=None, arg=None):
        """ This command sets the output offset voltage.
        Argument for offset in units of V."""
        if channel in self._channelOpts:
            str1 = ':BSWV OFST'        
            self.cmd(channel + str1, str(arg))
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
    
    def setPhase(self, channel=None, arg=None):
        """ This command sets the signal phase.
        Argument for phase in units of °."""
        if channel in self._channelOpts:
            str1 = ':BSWV PHSE'        
            self.cmd(channel + str1, str(arg))
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
    
    def getBasicSignal(self, channel=None):
        """ Returns the current parameters of the signal. In case the
        Modulation/Sweep/Burst mode is activated, the returned parameters
        describe the carrier signal."""
        if channel in self._channelOpts:
            return self.que(channel + ':BSWV')
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
    
    def setBasicSignal(self, channel=None, waveform=None, freq=None,
                    period=None, amp=None, offset=None, highLev=None,
                    lowLev=None, phase=None, width=None, duty=None,
                    risetime=None, falltime=None, delay=None, symmetry=None,
                    stDev=None, mean=None):
        """ Command to change the signal parameters for a standard signal
        (no modulation, sweep or burst). It is also possible to only change
        selected parameters, but it may be necessary to specify other 
        parameters in addition.
        
        Parameter description:        
        channel:  Options: 'C1'/'C2' for channel 1/channel 2.
        waveform: Signal waveform.
                  Options: 'SINE', 'SQUARE', 'RAMP', 'PULSE', 'NOISE', 'DC'
                  Depending on the chosen waveform, some parameters are not
                  available.
        freq:     Frequency in unit Hz. Complementary to 'period'.
                  Not available for waveforms DC, NOISE.
        period:   Period in unit s. Complementary to 'freq'.
                  Not available for waveforms DC, NOISE.
        amp:      Amplitude in unit Vpp. Complementary to specification of
                  'highLev'/'lowLev'. Not available for waveforms DC, NOISE.
        offset:   Offset in unit V. Complementary to specification of
                  'highLev'/'lowLev'. Only parameter for waveform DC.
                  Not available for waveform NOISE.
        highLev:  High level of signal in unit V. Complementary to
                  specification of 'amp'/'offset'. Not available for waveforms
                  DC, NOISE.
        lowLev:   Low level of signal in unit V. Complementary to
                  specification of 'amp'/'offset'. Not available for waveforms
                  DC, NOISE.
        phase:    Phase in degree. Not available for waveforms PULSE, DC, NOISE.
        width:    Pulsewidth in unit s. Measured from 50% level of rising to 
                  50% level of falling edge. Complementary to specification of
                  'width'.Only available for waveform PULSE.
        duty:     Duty cycle in percent. Ratio between pulsewidth and period.
                  Complementary to specification of 'width'. Only available for
                  waveforms SQUARE, PULSE.
        risetime: Risetime in unit s. Minimum is 6ns. Only available for
                  waveform PULSE.
        falltime: Falltime in unit s. Minimum is 6ns. Only available for
                  waveform PULSE.
        delay:    Pulse delay in unit s. Range is 0 < delay < period.
                  Only available for waveform PULSE.
        symmetry: Symmetry in percent. Used to skew the ramp/triangle waveform
                  into a sawtooth waveform. Only available for waveform RAMP.
        stDev:    Standard deviation of Gaussian noise in unit V.
                  Only available for waveform NOISE.
        mean:     Mean of Gaussian noise in unit V. Only available for
                  waveform NOISE.
        """        
        
        if channel in self._channelOpts:
            str1 = ''            
            
            wfOpts = ['SINE', 'SQUARE', 'RAMP', 'PULSE', 'NOISE', 'DC']
            if waveform in wfOpts:
                str1 = str1 + ', WVTP, ' + waveform
            elif waveform != None:
                str1 = str1 + ', WVTP, SINE'
                print('Warning: Unknown Waveform type! Assuming SINE as default.')
            
            if waveform != 'DC' or waveform != 'NOISE':
                # general options for most waveforms
                if freq != None:
                    str1 = str1 + ', FRQ, ' + str(freq)
                elif period != None:
                    str1 = str1 + ', PERI, ' + str(period)
                
                if amp != None or offset != None:
                    if amp != None:
                        str1 = str1 + ', AMP, ' + str(amp)
                    
                    if offset != None:
                        str1 = str1 + ', OFST, ' + str(offset)
                elif highLev != None or lowLev != None:
                    if highLev != None:
                        str1 = str1 + ', HLEV, ' + str(highLev)
                    
                    if lowLev != None:
                        str1 = str1 + ', LLEV, ' + str(lowLev)
            
            # phase option not defined for PULSE, DC and NOISE waveform
            if phase != None and (waveform != 'PULSE' and waveform != 'DC' and
                                  waveform != 'NOISE'):
                str1 = str1 + ', PHSE, ' + str(phase)        
            
            if waveform == 'DC':
                # DC option completely defined by offset
                if offset != None:
                    str1 = str1 + ', OFST, ' + str(offset)
            
            # special options for PULSE waveform
            if waveform == 'PULSE':
                if width != None:
                    str1 = str1 + ', WIDTH, ' + str(width)
                elif duty != None:
                    str1 = str1 + ', DUTY, ' + str(duty)
                
                if risetime != None:
                    str1 = str1 + ', RISE, ' + str(risetime)
                if falltime != None:
                    str1 = str1 + ', FALL, ' + str(falltime)
                
                if delay != None:
                    str1 = str1 + ', DLY, ' + str(delay)
            
            # special option for RAMP waveform
            if symmetry != None and waveform == 'RAMP':
                str1 = str1 + ', SYM, ' + str(symmetry)
            
            # special option for SQUARE waveform
            if duty != None and waveform == 'SQUARE':
                str1 = str1 + ', DUTY, ' + str(duty)
            
            # special options for NOISE waveform
            if waveform == 'NOISE':
                if stDev != None:
                    str1 = str1 + ', STDEV, ' + str(stDev)
                if mean != None:
                    str1 = str1 + ', MEAN, ' + str(mean)
            
            if len(str1) > 0:            
                self.cmd(channel + ':BSWV' + str1[1:])
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
        
    
    def getSweepSignal(self, channel='C1'):
        """ Returns the current parameters of the sweep and carrier 
        configuration. In case the state of the sweep mode is set to OFF,
        no parameters but a STATE OFF response are returned."""
        if channel in self._channelOpts:
            return self.que(channel + ':SWWV')
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
    
    def setSweepSignal(self, channel=None, sweepTime=None, startFreq=None, 
                    stopFreq=None, sweepMode=None, direction=None,
                    trigSource=None, trigEdge=None, trigOutput=None,
                    carrWF=None, carrFreq=None, carrAmp=None, carrOffset=None,
                    carrPhase=None, carrSymmetry=None, carrDuty=None):
        """ Command to change the parameters for a sweep signal. It is also
        possible to only change selected parameters of the sweep or carrier,
        but it may be necessary to specify other parameters in addition.
        
        Parameter description:        
        channel:      Options: 'C1'/'C2' for channel 1/channel 2.
        ----------------------
        Sweep parameters:
        ----------------------
        sweepTime:    Sweep time/duration in unit s.
        startFreq:    Start frequency in unit Hz.
        stopFreq:     Stop frequency in unit Hz.
        sweepMode:    Sweep mode. Options: 'LINE'/'LOG' for linear/logarithmic
                      sweep.
        direction:    Sweep direction. Options: 'UP'/'DOWN' for
                      increasing/decreasing frequency sweep.
        trigSource:   Trigger source. Options: 'INT'/'EXT'/'MAN'
        trigEdge:     Trigger edge/slope. Options: 'RISE'/'FALL' for setting
                      the trigger point on the rising/falling edge.
                      Only available if trigSource is set to 'EXT'.
        trigOutput:   Trigger output mode. Options: 'ON'/'OFF' for 
                      activated/deactivated output of the trigger signal at the
                      SYNC Out port.
                      Only available if trigSource is set to 'INT' or 'MAN'.
        -------------------
        Carrier parameters:
        -------------------
        carrWF:       Carrier waveform.
                      Options: 'SINE', 'SQUARE', 'RAMP'
                      Depending on the chosen waveform, some parameters are not
                      available.
                      IMPORTANT: Necessary argument if you want to set any of
                      the carrier parameters that are exclusive for a certain
                      waveform.
        carrFreq:     Carrier frequency in unit Hz.
        carrAmp:      Carrier amplitude in unit Vpp.
        carrOffset:   Carrier offset in unit V.
        carrPhase:    Carrier phase in degree.
        carrSymmetry: Carrier symmetry in percent. Used to skew the
                      ramp/triangle waveform into a sawtooth waveform. Only
                      available for carrier waveform RAMP.        
        carrDuty:     Carrier duty cycle in percent. Ratio between pulsewidth
                      and period. Only available for waveform SQUARE.
        """
        
        if channel in self._channelOpts:
            # activate sweep menu
            self.cmd(channel + ':SWWV STATE', 'ON')
            
            str1 = ''
            
            # sweep parameters            
            if sweepTime != None:
                str1 = str1 + ', TIME, ' + str(sweepTime)
            
            if stopFreq != None:
                str1 = str1 + ', STOP, ' + str(stopFreq)
            
            if startFreq != None:
                str1 = str1 + ', START, ' + str(startFreq)
            
            sweepModeOpts = ['LINE', 'LOG']
            if sweepMode in sweepModeOpts:
                str1 = str1 + ', SWMD, ' + sweepMode
            elif sweepMode != None:
                str1 = str1 + ', SWMD, LINE'
                print('Warning: Unknown Sweep mode! Assuming LINE as default.')
        
            directionOpts = ['UP', 'DOWN']
            if direction in directionOpts:
                str1 = str1 + ', DIR, ' + direction
            elif direction != None:
                str1 = str1 + ', DIR, UP'
                print('Warning: Unknown Sweep direction! Assuming UP as default.')
            
            # trigger parameters        
            trigOpts = ['INT', 'EXT', 'MAN']
            if trigSource in trigOpts:
                str1 = str1 + ', TRSR, ' + trigSource
            elif trigSource != None:
                str1 = str1 + ', TRSR, INT'
                print('Warning: Unknown Trigger source! Assuming INT as default.')
                    
            if trigSource == 'EXT':
                if trigEdge != None:
                    trigEdgeOpts = ['RISE', 'FALL']
                    if trigEdge in trigEdgeOpts:
                        str1 = str1 + ', EDGE, ' + trigEdge
                    elif trigEdge != None:
                        str1 = str1 + ', EDGE, RISE'
                        print('Warning: Unknown Trigger slope! Assuming RISE as default.')
            else:            
                if trigOutput != None:
                    trigOutputOpts = ['ON', 'OFF']
                    if trigOutput in trigOutputOpts:
                        str1 = str1 + ', TRMD, ' + trigOutput
                    elif trigOutput != None:
                        str1 = str1 + ', TRMD, OFF'
                        print('Warning: Unknown Trigger output mode! Assuming OFF as default.')        
            
            # set sweep parameters        
            if len(str1 > 0):
                self.cmd(channel + ':SWWV' + str1[1:])
            
            
            # carrier parameters
            str1 = ''            
            
            wfOpts = ['SINE', 'SQUARE', 'RAMP']
            if carrWF in wfOpts:
                str1 = str1 + ', WVTP, ' + carrWF
            elif carrWF != None:
                str1 = str1 + ', WVTP, SINE'
                print('Warning: Unknown Carrier waveform! Assuming SINE as default.')
            
            # general options for available waveforms
            if carrFreq != None:
                str1 = str1 + ', FRQ, ' + str(carrFreq)
            
            if carrAmp != None:
                str1 = str1 + ', AMP, ' + str(carrAmp)
                
            if carrOffset != None:
                str1 = str1 + ', OFST, ' + str(carrOffset)
            
            if carrPhase != None:
                str1 = str1 + ', PHSE, ' + str(carrPhase)
            
            # special option for RAMP waveform
            if carrSymmetry != None and carrWF == 'RAMP':
                str1 = str1 + ', SYM, ' + str(carrSymmetry)        
                
            # special option for SQUARE waveform
            if carrDuty != None and carrWF == 'SQUARE':
                str1 = str1 + ', DUTY, ' + str(carrDuty)
            
            
            # set carrier parameters
            if len(str1) > 0:
                self.cmd(channel + ':SWWV CARR', str1[2:])
        
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')
        
    
    def getModulationSignal(self, channel=None):
        """ Returns the current parameters of the modulation and carrier 
        configuration. In case the state of the modulation mode is set to OFF,
        no parameters but a STATE OFF response are returned."""        
        if channel in self._channelOpts:
            return self.que(channel + ':MDWV')
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')

        
    def setModulationSignal(self, channel=None, modMode=None, modSource=None,
                    modWF=None, modFreq=None, depth=None,
                    deviation=None, keyFreq=None, hopFreq=None,
                    carrWF=None, carrFreq=None, carrAmp=None, carrOffset=None,
                    carrPhase=None, carrSymmetry=None, carrDuty=None,
                    carrRiseTime=None, carrFallTime=None, carrDelay=None):
        """ Command to change the parameters for a modulation signal. It is
        also possible to only change selected modulation or carrier parameters,
        but it may be necessary to specify other parameters in addition.
        
        Parameter description:
        channel:      Options: 'C1'/'C2' for channel 1/channel 2.
        ----------------------
        Modulation parameters:
        ----------------------
        modMode:      Modulation mode. 
                      Options: 'AM'/'FM'/'PM'/'ASK'/'FSK'/'DSBAM'/'PWM'
                      Depending on the chosen waveform, some parameters are not
                      available.
                      IMPORTANT: Necessary if you want to set any of the other
                      modulation parameters.
        modSource:    Source of the modulating signal.
                      Options: 'INT'/'EXT'
                      Depending on the chosen waveform, some parameters are not
                      available.
        modWF:        Waveform of the modulating signal.
                      Options: 'SINE'/'SQUARE'/'TRIANGLE'/'UPRAMP'/'DNRAMP'/
                      'NOISE'
                      Only available if modSource is set to 'INT' and not
                      available for modModes 'ASK' and 'FSK'.
        modFreq:      Frequency of modulating signal.
                      Not available for modModes 'ASK' and 'FSK'.
                      Only available if modSource is set to 'INT'.
        depth:        Modulation depth in percent, ranging from 0 to 120%.
                      Only available if modMode is set to 'AM' and if modSource
                      is set to 'INT'.
        deviation:    Parameter meaning depends on modulation mode.
                      - For modMode 'FM':
                          FM frequency deviation in unit Hz.
                          Range from 0 to carrier frequency.
                      - For modMode 'PM':
                          PM phase deviation in degree.
                          Range from 0 to 360.
                      - For modMode 'PWM':
                          Duty cycle deviation in percent.
                          Range from 0 to (carrier pulse width - 15ns).
                      Not available for other modulation modes.
                      Only available if modSource is set to 'INT'.
        keyFreq:      ASK/FSK Key frequency. Only available for modModes 'ASK'
                      and 'FSK' and if modSource is set to 'INT'.
        hopFreq:      Only available for modMode 'FSK' (regardless of
                      choice for modulation source).
        -------------------
        Carrier parameters:
        -------------------
        carrWF:       Carrier waveform.
                      Options: 'SINE', 'SQUARE', 'RAMP', 'PULSE'
                      Depending on the chosen waveform, some parameters are not
                      available.
                      IMPORTANT: Necessary argument if you want to set any of
                      the carrier parameters that are exclusive for a certain
                      waveform.
        carrFreq:     Carrier frequency in unit Hz.
        carrAmp:      Carrier amplitude in unit Vpp.
        carrOffset:   Carrier offset in unit V.
        carrPhase:    Carrier phase in degree.
                      Not available for carrWF 'PULSE'.
        carrSymmetry: Carrier symmetry in percent. Used to skew the
                      ramp/triangle waveform into a sawtooth waveform. Only
                      available for carrWF 'RAMP'.        
        carrDuty:     Carrier duty cycle in percent. Ratio between pulsewidth
                      and period. Only available for carrWFs 'SQUARE' and
                      'PULSE'.
        carrRiseTime: Risetime in unit s. Minimum is 6ns. Only available for
                      carrWF 'PULSE'.
        carrFallTime: Falltime in unit s. Minimum is 6ns. Only available for
                      carrWF 'PULSE'.
        carrDelay:    Pulse delay in unit s. Range is 0 < delay < period.
                      Only available for carrWF 'PULSE'.
        """
        if channel in self._channelOpts:
            # activate modulation menu
            self.cmd(channel + ':MDWV STATE', 'ON')
            
            str1=''            
            modModeOpts = ['AM', 'FM', 'PM', 'FSK', 'ASK', 'DSBAM', 'PWM']
            if modMode in modModeOpts:
                str1 = channel + ':MDWV ' + modMode
            elif modMode != None:
                str1 = channel + ':MDWV AM'
                modMode = 'AM'
                print('Warning: Unknown Modulation mode! Assuming AM as default.')
                        
            modSourceOpts = ['INT', 'EXT']
            if modSource in modSourceOpts:
                str1 = str1 + ', SRC, ' + modSource
            elif modSource != None:
                str1 = str1 + ', SRC, INT'
                print('Warning: Unknown Modulation source! Assuming INT as default.')
            
            if modSource == 'INT':
                if not modMode in ['FSK', 'ASK']:
                    modWFOpts = ['SINE', 'SQUARE', 'TRIANGLE',
                                 'UPRAMP', 'DNRAMP', 'NOISE']            
                    if modWF in modWFOpts:
                        str1 = str1 + ', MDSP, ' + modWF
                    elif modWF != None:
                        str1 = str1 + ', MDSP, SINE'
                        print('Warning: Unknown Modulation waveform! Assuming SINE as default.')
                
                    if modFreq != None:
                        str1 = str1 + ', FRQ, ' + str(modFreq)
                        
                if modMode == 'AM':
                    # depth DEPTH
                    if depth != None:
                        str1 = str1 + ', DEPTH, ' + str(depth)
                    
                elif modMode == 'FSK':
                    # key frequncy KFRQ
                    if keyFreq != None:
                        str1 = str1 + ', KFRQ, ' + str(keyFreq)
                    # hop frequency HFRQ
                    if hopFreq != None:
                        str1 = str1 + ', HFRQ, ' + str(hopFreq)
                
                elif modMode == 'ASK':
                    # key frequency KFRQ
                    if keyFreq != None:
                        str1 = str1 + ', KFRQ, ' + str(keyFreq)
            
            else:
                if modMode == 'FSK':
                    # hop frequency HFRQ
                    if hopFreq != None:
                        str1 = str1 + ', HFRQ, ' + str(hopFreq)
                        
            if modMode in ['FM', 'PM', 'PWM']:
                    # deviation DEVI
                    if deviation != None:
                        str1 = str1 + ', DEVI, ' + str(deviation)                    
                        
            # set modulation parameters
            if len(str1) > 0 and modMode != None:
                self.cmd(str1)               
            elif modMode == None:
                print('Warning: Specify modulation mode to change modulation parameters!')
            
                           
            # carrier parameters
            str1 = ''
            wfOpts = ['SINE', 'SQUARE', 'RAMP', 'PULSE']
            if carrWF in wfOpts:
                str1 = str1 + ', WVTP, ' + carrWF
            elif carrWF != None:
                if modMode == 'PWM':
                    str1 = str1 + ', WVTP, PULSE'
                    print('Warning: Unknown Carrier waveform! Assuming PULSE as default for PWM mode.')
                else:
                    str1 = str1 + ', WVTP, SINE'
                    print('Warning: Unknown Carrier waveform! Assuming SINE as default.')
            
            # general options for available waveforms
            if carrFreq != None:
                str1 = str1 + ', FRQ, ' + str(carrFreq)
            
            if carrAmp != None:
                str1 = str1 + ', AMP, ' + str(carrAmp)
                
            if carrOffset != None:
                str1 = str1 + ', OFST, ' + str(carrOffset)
            
            if carrPhase != None and carrWF != 'PULSE':
                str1 = str1 + ', PHSE, ' + str(carrPhase)
            elif carrWF == 'PULSE':
                print('Warning: Phase can not be set for carrier waveform PULSE!')
            
            # special option for RAMP waveform
            if carrSymmetry != None and carrWF == 'RAMP':
                str1 = str1 + ', SYM, ' + str(carrSymmetry)        
                
            # special option for SQUARE and PULSE waveform
            if carrDuty != None and carrWF in ['SQUARE', 'PULSE']:
                str1 = str1 + ', DUTY, ' + str(carrDuty)
            
            # special options for PULSE waveform        
            if carrWF == 'PULSE':
                if carrRiseTime != None:
                    str1 = str1 + ', RISE, ' + str(carrRiseTime)
                
                if carrFallTime != None:
                    str1 = str1 + ', FALL, ' + str(carrFallTime)
                
                if carrDelay != None:
                    str1 = str1 + ', DLY, ' + str(carrDelay)
            
            # set carrier parameters
            if len(str1) > 0:
                self.cmd(channel + ':MDWV CARR', str1[2:])
                
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')

        
    def getBurstSignal(self, channel=None):
        """ Returns the current parameters of the burst and carrier 
        configuration. In case the state of the burst mode is set to OFF,
        no parameters but a STATE OFF response are returned."""        
        if channel in self._channelOpts:
            return self.que(channel + ':BTWV')
        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')

        
    def setBurstSignal(self, channel=None, burstMode=None, trigSource=None,
                    trigMode=None, period=None, startPhase=None, polarity=None,
                    nCycle=None, delay=None, edge=None,
                    carrWF=None, carrFreq=None, carrAmp=None, carrOffset=None,
                    carrPhase=None, carrSymmetry=None, carrDuty=None,
                    carrRiseTime=None, carrFallTime=None, carrDelay=None,
                    carrStDev=None, carrMean=None):
        """ Command to change the parameters for a burst signal. It is
        also possible to only change selected burst or carrier parameters,
        but it may be necessary to specify other parameters in addition.
        
        Parameter description:
        channel:      Options: 'C1'/'C2' for channel 1/channel 2.
        ----------------------
        Burst parameters:
        ----------------------
        burstMode:    Burst mode. 
                      Options: 'NCYC'/'GATE'
                      Depending on the chosen burst mode, some parameters are
                      not available.
                      IMPORTANT: Necessary if you want to set any of the other
                      burst parameters.
        trigSource:   Source of the signal that triggers the burst.
                      Options: 'INT'/'EXT'/'MAN'
                      Depending on the chosen waveform, some parameters are not
                      available.
        trigMode:     Trigger output mode. Only available if trigSource is set
                      to 'INT' or 'MAN'.
                      'Ext Trig/Gate/FSK/Burst' output port on the back of the
                      device becomes a sync output.
                      Options: 'RISE'/'FALL'/'OFF'
                      For burstMode 'NCYC', the options 'RISE'/'FALL' specify
                      whether the burst triggers on the rising/falling slope of
                      the square wave trigger signal.
                      For burstMode 'GATE', the options 'RISE'/'FALL' specify
                      if the signal is 5V/GND if the burst is active and GND/5V
                      in the waiting period.
                      For trigMode 'OFF' there is no output at the port. 
        period:       Period of the square wave signal triggering the burst.
                      Only available if trigSource is 'INT'.
        startPhase:   Initial phase of the carrier signal at starting point of
                      burst. Not available for carrWFs 'PULSE' and 'NOISE'.
        polarity:     Decides whether the gated burst is active if the gate
                      signal is at 5V/GND level.
                      Options: 'POS'/'NEG'
                      For option 'POS'/'NEG', the burst is active if the gate
                      signal is 5V/GND.
        nCycle:       Number of cycles to output after burst was triggered.
                      Only natural numbers are accepted as argument.
                      Only available if burstMode is set to 'NCYC'. If
                      trigMode is 'EXT' you can also set this value to 'INF'.
                      and 'FSK' and if modSource is set to 'INT'.
        delay:        Burst delay with respect to trigger point.
        edge:         Edge/Slope of the trigger signal on which the burst is
                      fired.
                      Options: 'RISE'/'FALL'
                      Only available if trigMode is 'EXT'.
        -------------------
        Carrier parameters:
        -------------------
        carrWF:       Carrier waveform.
                      Options: 'SINE', 'SQUARE', 'RAMP', 'PULSE', 'NOISE'
                      Depending on the chosen waveform, some parameters are not
                      available.
                      IMPORTANT: Necessary argument if you want to set any of
                      the carrier parameters that are exclusive for a certain
                      waveform.
        carrFreq:     Carrier frequency in unit Hz.
        carrAmp:      Carrier amplitude in unit Vpp.
        carrOffset:   Carrier offset in unit V.
        carrPhase:    Carrier phase in degree. 
                      Not available for carrWFs 'PULSE' and 'NOISE'.
        carrSymmetry: Carrier symmetry in percent. Used to skew the
                      ramp/triangle waveform into a sawtooth waveform. Only
                      available for carrWF 'RAMP'.        
        carrDuty:     Carrier duty cycle in percent. Ratio between pulsewidth
                      and period. Only available for carrWFs 'SQUARE' and
                      'PULSE'.
        carrRiseTime: Risetime in unit s. Minimum is 6ns. Only available for
                      carrWF 'PULSE'.
        carrFallTime: Falltime in unit s. Minimum is 6ns. Only available for
                      carrWF 'PULSE'.
        carrDelay:    Pulse delay in unit s. Range is 0 < delay < period.
                      Only available for carrWF 'PULSE'.
        carrStDev:    Standard deviation of gaussian noise carrier signal in
                      unit V. Only available for carrWF 'NOISE'.
        carrMean:     Mean of gaussian noise carrier signal in unit V. Only
                      available for carrWF 'NOISE'.
        """
        
        if channel in self._channelOpts:
            # activate burst menu
            self.cmd(channel + ':BTWV STATE', 'ON')
            
            str1=''

            burstModeOpts = ['GATE', 'NCYC']            
            if burstMode not in burstModeOpts and burstMode != None:
                print('Warning: Unknown Burst mode! Assuming NCYC as default.')                
                burstMode = 'NCYC'
            
            # device bug: some parameters can not be accessed in GATE mode
            # solution: set parameters in NCYC mode first and then change
            #           to GATE mode                
            str1 = channel + ':BTWV GATE_NCYC, NCYC'            
             
            trigSourceOpts = ['INT', 'EXT', 'MAN']                
            if trigSource in trigSourceOpts:            
                str1 = str1 + ', TRSR, ' + trigSource
            else:
                str1 = str1 + ', TRSR, INT'
                print('Warning: Unknown Trigger source! Assuming INT as default.')
                trigSource = 'INT'            
            
            if carrWF != 'NOISE':
                if trigSource == 'EXT':
                    if burstMode == 'NCYC':
                        if nCycle != None:
                            if nCycle == 'INF':
                                str1 = str1 + ', TIME, INF'
                            else:
                                nCycle = abs(int(nCycle))
                                str1 = str1 + ', TIME, ' + str(nCycle)
                                
                        edgeOpts = ['RISE', 'FALL']                    
                        if edge in edgeOpts:
                            str1 = str1 + ', EDGE, ' + edge
                        elif edge != None:
                            str1 = str1 + ', EDGE, RISE'
                            print('Warning: Unknown external Trigger slope! Assuming RISE as default.')
                    
                    else:
                        polarityOpts = ['POS', 'NEG']            
                        if polarity in polarityOpts:
                            str1 = str1 + ', PLRT, ' + polarity
                        elif polarity != None:
                            str1 = str1 + ', PLRT, POS'
                            print('Warning: Unknown Polarity! Assuming POS as default.')
                                
                else:
                    if burstMode == 'NCYC':                    
                        trigModeOpts = ['RISE', 'FALL', 'OFF']
                        if trigMode in trigModeOpts:
                            str1 = str1 + ', TRMD, ' + trigMode
                        elif trigMode != None:
                            str1 = str1 + ', TRMD, OFF'
                            print('Warning: Unknown Trigger output mode! Assuming OFF as default.')
                    
                        if nCycle != None:
                            if nCycle == 'INF':
                                nCycle = int(1)
                                print('Warning: Number of Cycles INF only valid with Trigger source EXT! Assuming 1 as default.')
                            else:
                                nCycle = abs(int(nCycle))
                            str1 = str1 + ', TIME, ' + str(nCycle)
                    else:
                        trigModeOpts = ['RISE', 'FALL', 'OFF']
                        if trigMode in trigModeOpts:
                            str1 = str1 + ', TRMD, ' + trigMode
                        elif trigMode != None:
                            str1 = str1 + ', TRMD, OFF'
                            print('Warning: Unknown Trigger output mode! Assuming OFF as default.')                        
                        
                        polarityOpts = ['POS', 'NEG']            
                        if polarity in polarityOpts:
                            str1 = str1 + ', PLRT, ' + polarity
                        elif polarity != None:
                            str1 = str1 + ', PLRT, POS'
                            print('Warning: Unknown Polarity! Assuming POS as default.')
                                
                if startPhase != None and carrWF != 'PULSE':
                    str1 = str1 + ', STPS, ' + str(startPhase)
        
            else:            
                if trigSource != 'EXT':
                    trigModeOpts = ['RISE', 'FALL', 'OFF']
                    if trigMode in trigModeOpts:
                        str1 = str1 + ', TRMD, ' + trigMode
                    elif trigMode != None:
                        str1 = str1 + ', TRMD, OFF'
                        print('Warning: Unknown Trigger output mode! Assuming OFF as default.')
                
                                
                
                polarityOpts = ['POS', 'NEG']
                if polarity in polarityOpts:
                    str1 = str1 + ', PLRT, ' + polarity
                elif polarity != None:
                    str1 = str1 + ', PLRT, POS'
                    print('Warning: Unknown Polarity! Assuming POS as default.')
            
            if period != None and trigSource == 'INT':
                    str1 = str1 + ', PRD, ' + str(period)
                
            if delay != None:
                str1 = str1 + ', DLAY, ' + str(delay)            
            
            # set burst parameters
            if len(str1) > 0 and burstMode != None:
                self.cmd(str1)
                
                # bug fix for GATE burst mode
                if burstMode == 'GATE':
                    str1 = channel + ':BTWV GATE_NCYC, GATE'
                    self.cmd(str1)
                elif carrWF == 'NOISE':
                    print('Warning: For Carrier waveform NOISE, the only valid burst mode is GATE! Burst mode was automatically set to GATE.')
                    str1 = channel + ':BTWV GATE_NCYC, GATE'
                    self.cmd(str1)
            elif burstMode == None:
                print('Warning: Specify burst mode to change burst parameters!')
                
            
            str1 = ''
                           
            # carrier parameters
            wfOpts = ['SINE', 'SQUARE', 'RAMP', 'PULSE', 'NOISE']
            if carrWF in wfOpts:
                str1 = str1 + ', WVTP, ' + carrWF
            elif carrWF != None:
                str1 = str1 + ', WVTP, SINE'
                print('Warning: Unknown Carrier waveform! Assuming SINE as default.')
                carrWF = 'SINE'            
            
            if carrWF != 'NOISE':
                # general options for available waveforms
                if carrFreq != None:
                    str1 = str1 + ', FRQ, ' + str(carrFreq)
                
                if carrAmp != None:
                    str1 = str1 + ', AMP, ' + str(carrAmp)
                    
                if carrOffset != None:
                    str1 = str1 + ', OFST, ' + str(carrOffset)
                
                # special option for SINE, SQUARE and RAMP waveform
                if carrPhase != None and carrWF in ['SINE', 'SQUARE', 'RAMP']:
                    str1 = str1 + ', PHSE, ' + str(carrPhase)
                
                # special option for RAMP waveform
                if carrSymmetry != None and carrWF == 'RAMP':
                    str1 = str1 + ', SYM, ' + str(carrSymmetry)        
                    
                # special option for SQUARE and PULSE waveform
                if carrDuty != None and carrWF in ['SQUARE', 'PULSE']:
                    str1 = str1 + ', DUTY, ' + str(carrDuty)
                
                # special options for PULSE waveform        
                if carrWF == 'PULSE':
                    if carrRiseTime != None:
                        str1 = str1 + ', RISE, ' + str(carrRiseTime)
                    elif carrFallTime != None:
                        str1 = str1 + ', FALL, ' + str(carrFallTime)
                    
                    if carrDelay != None:
                        str1 = str1 + ', DLY, ' + str(carrDelay)
                
            else:
                # special options for NOISE waveform
                if carrStDev != None:
                    str1 = str1 + ', STDEV, ' + str(carrStDev)
                if carrMean != None:
                    str1 = str1 + ', MEAN, ' + str(carrMean)
            
            # set carrier parameters
            if len(str1) > 0:
                self.cmd(channel + ':BTWV CARR', str1[2:])

        else:
            print('Warning: Unknown Channel! Options are C1 for Channel 1 and C2 for Channel 2.')