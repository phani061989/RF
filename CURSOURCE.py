# -*- coding: utf-8 -*- #######################################################
"""
Created on Thu Jul  7 10:29:08 2016

@author: Olivia Lanes, CMF Schneider

v1.2.4 - OSC:
    - added an exception to the lower_terminal_state function, it is not possible
    to change the terminal state if the output is on
v.1.2.3 - OSC:
    - removed the set_limits and output_off mode in the init, it was a problem
    when different users are using different channels. Now everybody needs to 
    set them after connecting to the device.
    - inserted the function low_terminal_state used to set it to ground or 
    float mode

v1.2.2 - CHR:
    - Fixed bug in compliance

v1.2.1 - Oscar:
    - inserted the function output_protection
    - in set_limits is now possible to specify the output_protection state

v1.2.0 - Oscar:
    - corrected some minor bug in the code
    - inserted the function output_off_mode, also the instrument will initialize the output_off_mode to 'zero' as def
    - inserted the function set_limits to re-write the max, min values for voltage and current, set automatically the compliance to max value
    - inserted a safe_mode in the function output_level that gives an error if the value is higher than the max

v1.1.0 - Oscar:
fixed some mistake in the code as self.v.write calls
modified the functions plot_IV, plot_VI, this function will return a data module now.

"""
VERSION = "1.2.3"
print('CURSOURCE v{}'.format(VERSION))
###############################################################################
# class definition
import vxi11
import numpy as np
import DataModule as dm




class CurSource(vxi11.Instrument):
  """ Library for the Keysight B29XX Measuring Unit
  Requires IP Address of the device.

  Optional Parameters:
  min_cur = Minimum Current (A) - def -1 mA
  max_cur = Maximum Current (A) - def +1 mA
  min_vol = Minimum Voltage (V) - def 0 V
  max_vol = Maximum Voltage (V) - def 10 mV
  
  output_off_mode (def 0) - when the devise is off, the output is at 0 A or 0 V , set to 1 for HI-impedance
  
  reset (def False): reset the device to the default settings
  
  """ 
  def __init__(self, ip, reset = False):
    self.version = VERSION
    self.ip = ip
    self.min_cur = [0,0]
    self.max_cur = [0,0]
    self.min_vol = [0,0]
    self.max_vol = [0,0]

    # Initialize device
    super(CurSource, self).__init__(ip)
    print(self.identify())
    print('\nNOTE: Remember to set_limits, low_terminal_state and output_off mode\n')
    
    if reset:
        self.reset()

    #self.set_limits(min_cur,max_cur,min_vol,max_vol,1,output_protection)
    #self.set_limits(min_cur,max_cur,min_vol,max_vol,2,output_protection)
    """
    # This is the auto-range used for resistance,voltage or current measurements "on spot", they are set as pre-caution
    self.write('CURR:RANG:AUTO:LLIM {}'.format(min_cur))
    self.write('CURR:RANG:AUTO:ULIM {}'.format(max_cur))
    self.write('VOLT:RANG:AUTO:LLIM {}'.format(min_vol))
    self.write('VOLT:RANG:AUTO:ULIM {}'.format(max_vol))
    """
    
    #Set output_off to 'zero'
    #self.output_off_mode(output_off_mode,1)
    #self.output_off_mode(output_off_mode,2)
    
  # STANDARD IO FUNCTIONS ######################################################
  def cmd(self, str1, arg = ''):
    return self.write(''.join([str(str1),' ',str(arg),'\n']))

  def query(self, str1, arg = ''):
    return self.ask(''.join([str(str1),'? ',arg,'\n']))
    
  # Identification
  def identify(self):
    return self.query('*IDN')

  def read_errors(self):
    """ Returns list of errors """
    return self.query(':SYST:ERR:ALL') 

  # Utilities functions
  def __check_channel(self, ch, channels=[1,2]):
      ch = int(ch)
      if ch not in channels:
          print('Wrong channel inserted')
          raise Exception('CHERR')
      return ch
  
  def set_limits(self,min_cur=-1e-3, max_cur=1e-3, min_vol=0, max_vol=10e-3,
                 channel=1, output_protection=True):
      """This function sets the limits of the output for the specified channel 
      (def channel 1)
      
      min_cur (def -1e-3 A)
      max_cur (def 1e-3 A)
      min_col (def 0 V)
      max_col (def 10e-3 V)
      
      channel (def 1)
      output_protection (def True): if the inst goes in compliance, it will 
      switch off the channel
      """
      self.__check_channel(channel)    
      
      self.min_cur[channel-1] = min_cur
      self.max_cur[channel-1] = max_cur
      self.min_vol[channel-1] = min_vol
      self.max_vol[channel-1] = max_vol
      
      #Output protection
      self.output_protection(output_protection,channel)
      
      mode = self.output_mode(None,channel)
      
      if mode == 'CURR':
          self.compliance(max_vol, channel)
      else:
          self.compliance(max_cur, channel)
          
    
    
  ##############################################################################
  # Device specific utilities ##################################################  
  def abort(self): # arg = ON|OFF|1|0
    """ This command causes the List or Step sweep in progress to abort. 
    Even if INIT:CONT[:ALL] is set to ON, the sweep will not immediately 
    re-initiate.
    """
    str1 = ':ABORt'
    self.cmd(str1,'')   

  # :INITiate Subsystem
  def initiate(self): # 
    """ Sets trigger to the armed state."""
    str1 = ':INITiate'
    self.cmd(str1,'')
  
  def initiate_cont(self,arg): # arg = ON|OFF|1|0
    """ Continuously rearms the trigger system after completion of a triggered 
    sweep.
    """
    str1 = ':INITiate:CONTinuous'
    self.cmd(str1,arg)

  def output(self, arg = None, channel = 1):
    """ Turn output 1|2 power on/off.
    arg = ON|OFF|1|0
    channel = 1|2
    """
     #output: (int) 1= on, 0 = off
    str1 = ':OUTP'+str(channel)+':STAT'
    if arg is None:
      return int(self.query(str1))

    else:
      return self.cmd(str1,arg)      
      
  def output_mode(self, arg = None, channel = 1):
    """ Reads or sets the output function 
    arg = None | "CURR" | "VOLT" 
    channel = 1|2
    
    safe_mode (def True): it will automatically set the compliance following the max/min settings registered in the Cursource module
    """
    
    channel = self.__check_channel(channel)
    
    str1 = ':SOUR'+str(channel)+':FUNC:MODE'
    if arg == None:
      # Just print current mode
      return self.query(str1).upper()
    
    if arg.upper() == 'CURR':
      # Set mode to 'CURR' and set voltage compliance
      if self.output_mode(channel=channel) != 'CURR':
          self.cmd(str1, arg)
      self.compliance(self.max_vol[channel-1],channel)
      
    elif arg.upper() == 'VOLT':
      # Set mode to 'VOLT' and set current compliance
      if self.output_mode(channel=channel) != 'VOLT':
          self.cmd(str1, arg)
      self.compliance(self.max_cur[channel-1],channel)
    
    else:
      print("No valid argument")
      raise Exception('MODERR')

  def output_off_mode(self,mode=None,channel=1):
      """Query/Set the output_off_mode:
          
          mode: 0 or 'zero' to have the machine set the output value to 0V or 0A
                1 or 'HIZ' to set HI-impedance
      """
      channel= self.__check_channel(channel)
      
      cmd = ':OUTP{}:OFF:MODE'.format(channel)
      
      if mode is None:
          return self.query(cmd)
      
      if type(mode) is str:
          if mode.upper() == 'ZERO':
              mode = 0
          elif mode.upper() == 'HIZ':
              mode = 1
          else:
              print('Wrong mode inserted')
              raise Exception('MODERR')
      
      if mode<0 or mode>1:
          print('Wrong mode inserted')
          raise Exception('MODERR')
      else:
          if mode is 0:
              self.cmd(cmd,'ZERO')
          else:
              self.cmd(cmd,'HIZ')
      

  def output_level(self, arg = None, channel = 1,safe_mode = True):
    """ Query/Set the level of output 1|2
    arg = None | Float (A or V, depending on mode)
    channel = 1 | 2
    
    safe_mode (def True): doesn't allow to set a current or voltage higher than the one set in the limit for the specified channel
    """
    
    channel = self.__check_channel(channel)
    mode = self.output_mode(None, channel) # CURR or VOLT
    
    str1 = ':SOUR' + str(channel) + ':'
    str1+= mode
    str1 += ':LEV:IMM:AMPL'
    
    if arg is None:
      return self.query(str1)
    
    if safe_mode is True:
      if mode == 'CURR':
          if arg< self.min_cur[channel-1] or arg > self.max_cur[channel-1]:
              print('ERROR: current value too high')
              raise Exception('HIGHCURR')
      else:
          if arg< self.min_vol[channel-1] or arg > self.max_vol[channel-1]:
              print('ERROR: voltage value too high')
              raise Exception('HIGHVOLT')
              
    self.cmd(str1, arg)
  
  def output_protection(self, arg=None, channel=1):
      """This function sets/query the output protection for the specified 
      channel, if it is True or 1, when the instruments goes over the 
      compliance, the output will be switched off and the current/voltage value
      will be set to zero"""
      
      channel = self.__check_channel(channel)
      
      str1 = ':OUTP{}:PROT'.format(channel)
      if arg is None:
          return self.query(str1)
      
      if arg is True or arg ==1:
              self.cmd(str1,'ON')
      elif arg is False or arg == 0:
          self.cmd(str1,'OFF')
      else:
          print('Wrong argument inserted')
          raise Exception('ARGERR')

  def low_terminal_state(self,mode=None,channel=1):
      '''This function sets the low_terminal state, the options are:
          
          - 0 of 'f' for float mode 
          - 1 or 'g' for ground mode
          
          if it is None (def), it will be queried.
          
          Channel can be 1 (def) or 2
      '''
      
      channel = int(channel)
      if channel<1 or channel>2:
          print('Wrong channel inserted')
          raise Exception('CHNUMERR')
          
      if self.output(channel=channel) == 1:
          print('It is not possible to change the terminal state if the output is on')
          raise Exception('TERSTATEERR')
      
      cmdstr = 'OUTP{}:LOW'.format(str(channel))
      
      if mode is None:
           return self.query(cmdstr)
      
      if type(mode) is str:
          if mode.lower() == 'g':
              mode = 1
          elif mode.lower() == 'f':
              mode = 0
          else:
              print('Wrong mode inserted')
              raise Exception('LOWTMODERR')
      
      mode = int(mode)
      if mode<0 or mode>1:
            print('Wrong mode inserted')
            raise Exception('LOWTMODERR')
        
      if mode == 0:
            self.cmd(cmdstr,'FLO')
      else:
            self.cmd(cmdstr,'GRO')
      
        

  def compliance(self, arg=None, channel=1):
    """Query/Set the compliance for 'CURR' | 'VOLT' for channel 1 | 2
    
    Parameters
    -----------
    arg : float
        Compliance value in A or V, depending on mode
    channel : int
        Channel (1 or 2)

    Returns
    --------
    str
        Current value for query mode
    """
    channel = self.__check_channel(channel)

    # reads the mode of the specified channel
    mode = self.output_mode(channel=channel)

    # the compliance can be only set on the opposite, ex: voltage for 
    # mode=current 
    if mode == 'CURR':
        mode = 'VOLT'
    else:
        mode = 'CURR'
    
    str1 = ':SENS{channel}:{mode}:PROT'.format(channel=channel, mode=mode)
    if arg == None:
        # Query and display
        return self.query(str1)
    else:
        # Set compliance
        return self.cmd(str1, arg)   

  def reset(self):
    """ Resets machine to default settings """
    self.write("*RST")


  def arm_trigger(self, points, time, channel = 1):
    """ Arm trigger for sweeps and acquisition. 
    This must be done before initializing a measurement.
    Measures and change output at same time""" 
    str_base = ":TRIG" + str(channel) + ":"
    # For setting voltage/current
    str1 = str_base + 'TRAN:'
    self.cmd(str1 + 'COUN', str(points))
    self.cmd(str1 + 'TIM', str(time))
    # For Acquisition
    str2 = str_base + 'ACQ:'
    self.cmd(str2 + 'COUN', str(points))
    self.cmd(str2 + 'TIM', str(time))

  def meas_volt(self, channel = 1): 
    self.write('INIT (@{})'.format(channel))
    str2 = 'FETC:ARR:VOLT? (@{})'.format(channel)
    data = self.ask(str2)
    data = data.split(',')
    data = np.float_(data)
    return data   

  def meas_curr(self, channel = 1):
    """ Initiates current measurement and retrieves data  """
    self.write('INIT (@{})'.format(channel))
    str2 = 'FETC:ARR:CURR? (@{})'.format(channel)
    data = self.ask(str2)
    data = data.split(',')
    data = np.float_(data)
    return data    
  
  def meas_res(self, channel):
    """ Initiates resistance measurement and returns data """ 
    # Set device mode to resistance measurement
    self.write(':SENS:FUNC "RES" (@{})'.format(channel))
    self.write(':SENS:RES:MODE AUTO (@{})'.format(channel))
    self.write(':SENS:RES:OCOM ON (@{})'.format(channel))
    self.write(':SENS:RES:RANG: AUTO ON  (@{})'.format(channel))
    self.write('INIT (@{})'.format(channel))
    
    data = self.query('FETC:ARR:RES (@{})'.format(channel))
    data = data.split(',')
    data = np.float_(data)
    return data 

  def stairsweep(self, minval, maxval, points, Compliance= None,mode = "CURR", channel = 1):
    """ Sweep using discrete steps 
    Automatically turns on Output.
    mode = "CURR" | "VOLT" 
    channel = 1 | 2
    """
    # Set mode
    self.output_mode(mode, channel)
    
    if Compliance is not None:
        if mode.upper() == 'CURR':
            self.compliance(mode = 'VOLT', arg = Compliance, channel = 1)
        elif mode.upper() == 'VOLT':
            self.compliance(mode = 'CURR', arg = Compliance, channel = 1)
        else:
            print('Wrong mode inserted')
            raise Exception('MODEERR')
    
    str_base = "SOUR" + str(channel) + ":" + str(mode) + ":"
    # Set mode to sweep
    str1 = str_base + "MODE"
    self.cmd(str1, "SWE")
    # Set start
    str2 = str_base + "STAR"
    self.cmd(str2, str(minval))
    # Set stop
    str3 = str_base + "STOP"
    self.cmd(str3, str(maxval))
    # Set points
    str4 = str_base + "POIN"
    self.cmd(str4, str(points))

  def listsweep(self, point_list, mode = "CURR", channel = 1):
      """ Sweep through the points of a list """ 
      self.output_mode(mode, channel)
      str2 = "SOUR" + str(channel) + ":LIST:" + str(mode)
      qmsg2= str(list(point_list)).strip('[]')
      self.cmd(str2,qmsg2)
   
  def plot_VI(self, start, stop, points, interval, Compliance=None, channel = 1,linfit=True):
    """Performs a stairsweep measurement V vs I, returns a data module, as default a linear fit on the data will be performed"""
    
    self.stairsweep(start, stop, points, Compliance,"CURR",channel)
    self.arm_trigger(points, interval, channel)
    
    x = np.linspace(start, stop, points)
    y = self.meas_volt(channel)
    
    data= dm.data_2d(x,y)
    if linfit is True:
        data.fit([0.1,0.1],dm.poly_fit,plot=False)
    
    return data
      
  def plot_IV(self, start, stop, points, interval, Compliance=None,channel = 1,linfit=True):
    """Performs a stairsweep measurement I vs V, returns a data module"""
    self.stairsweep(start, stop, points, Compliance,"VOLT", channel)
    self.arm_trigger(points, interval, channel)
    
    x = np.linspace(start, stop, points)
    y = self.meas_curr(channel)
    
    data= dm.data_2d(x,y)
    if linfit is True:
        data.fit([0.1,0.1],dm.poly_fit,plot=False)
    
    return data
    

  #Returns the current channel 1 current value without changing into resistance mode
  def spot_meas_curr1(self):
     output=str(self.query(':SENS:FUNC'))
     
     if output == '"VOLT","CURR"':
       self.cmd(":FORM:ELEM:SENS CURR",'')
       data=self.query("MEAS","(@1)")
      
       return data
     else:
      pass
 
  #Returns the current channel 2 current value without changing into resistance mode
  def spot_meas_curr2(self):
     output=str(self.query(':SENS:FUNC'))
     
     if output == '"VOLT","CURR"':
       self.cmd(":FORM:ELEM:SENS CURR",'')
       data=self.query("MEAS","(@2)")
      
       return data
     else:
      pass

     
  #Returns the current channel 1 voltage value 
  def spot_meas_volt1(self):
     self.cmd(":FORM:ELEM:SENS VOLT",'')
     data=self.query("MEAS","(@1)")
     return data

  #Returns the current channel 2 voltage value 
  def spot_meas_volt2(self):
     self.cmd(":FORM:ELEM:SENS VOLT",'')
     data=self.query("MEAS","(@2)")
     return data
     
  def spot_meas_res1(self):
    """ Returns the current channel 1 resistance value without changing into 
    current mode
    """
    output = str(self.query(':SENS:FUNC'))
     
    if "RES" in output.split(','):
      self.cmd(":FORM:ELEM:SENS RES",'')
      data=self.query("MEAS","(@1)")
      return data       
    else:
      return None
    
  def spot_meas_res2(self):
    output=str(self.query(':SENS:FUNC'))
     
    if "RES" in output.split(','):
      self.cmd(":FORM:ELEM:SENS RES",'')
      data=self.query("MEAS","(@2)")
      return data       
    else:
      return None

