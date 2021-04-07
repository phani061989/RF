
import numpy as np
import time
from .UtilityFunctions import load_HVI,digits_to_Vp



class Readout(object):
    def __init__(self,Cryostat_ID=None,pars_dict=None):
        """
        This class is used to perform a system readout:
        Args:
            - Cryostat_ID: None (def) or string, it specify the Cryostat so 
            that it can be used to read the base temp and insert it in the 
            measurement
            - pars_dict: a python dictionary with all the parameters for the 
            readout
        
        the dictionary used to perform a readout is the following:
            {'Readout generator': None,
             'LO generator': None,
             'Digitizer': None,
             'HVI_path': '',
             'Averages': 1,
             'Repetitions period': 1000,
             'Readout frequency':10,
             'Downmixer power':8,
             'Readout_pulse': {'Length':1000,'Delay':0}}
        
        the user must connect the instruments and use the dedicated functions
        to set the previous values.
        
        """
        if Cryostat_ID is None:
            print('No Cryostat_ID specified, no temperature will be measured')
            Cryostat_ID=''
        if type(Cryostat_ID)!=str:
            print('Cryostat_ID must be a string')
            raise Exception(TypeError)
        
            
        if pars_dict is None:
            self.pars_dict = {'Cryostat_ID':Cryostat_ID}
        else:
            if type(pars_dict)!=dict:
                print('exp_dict must be a python dictionary')
                raise Exception(TypeError)
            
            self.pars_dict = pars_dict
            self.pars_dict.update({'Cryostat_ID':Cryostat_ID})
            
        self.pars_dict.update({'Readout generator': None,
                             'LO generator': None,
                             'Digitizer': None,
                             'Dig channel': 1,
                             'HVI': None,
                             'Averages': 1,
                             'Repetitions period': 1000,
                             'DM frequency':10,
                             'Downmixer power':8,
                             #'Acquisition delay fix':320,
                             })
        self._rogen = None
        self._logen = None
        self._dig = None
        self.cryo_id = Cryostat_ID
##----------------------------------------------------------------------------- Set functions ----
    
    
##-------------------------------------------------------------------------------------------    
    
    def set_HVI(self,path,dig,awg):
        """Function used to set the HVI, requires the path and initialized
        digitizer and awg classes"""
        
        self._HVI = load_HVI(path,dig,awg)
        self.pars_dict.update({'HVI': path})
        self._dig = dig
        self.pars_dict.update({'Digitizer':{'Slot':dig._slot,'Chassis': dig._chassis}})
    
    def dig_ch(self,chn=None):
        """Function used to get(def)/set which channel of the digitizer will be 
        used for readout"""
        if chn is None:
            return self.pars_dict['Dig channel']
        else:
            
            self.pars_dict.update({'Dig channel': int(chn)})
    
    def rogen(self,generator=None):
        """Function used to get(def)/set which generator will be used for the
        readout pulse, it must be a Instruments.SG class object"""
        if generator is None:
            if self._rogen is None:
                print('No generator assigned')
            else:
                print(self._rogen.id)
                self._rogen.get_parameters()
        else:
            self._rogen = generator
            self.pars_dict.update({'Readout generator': generator.id})
    
    def logen(self,generator=None):
        """Function used to get(def)/set which generator will be used for 
        downmixing, it must be a Instruments.SG class object"""
        if generator is None:
            if self._logen is None:
                print('No generator assigned')
            else:
                print(self._logen.id)
                self._logen.get_parameters()
        else:
            self._logen = generator
            self.pars_dict.update({'LO generator': generator.id})
    
    def averages(self,averages=None):
        """Function used to get(def)/set the averages, minimum is 1"""
        if averages is None:
            return self.pars_dict['Averages']
        else:
            averages = int(averages)
            if averages<0:
                raise self._ROEXC('ValueError','averages must be an integer larger than 1\n')
            self.pars_dict.update({'Averages': averages})
    """
    def acq_delay(self,delay=None):
        '''Function used to get(def)/set the acquisition delay fix, 320 def.
        It must be a multiple of 10ns'''
        if delay is None:
            return self.pars_dict['Acquisition delay fix']
        else:
            delay = int(delay)
            if delay%10 != 0:
                raise self._ROEXC('ValueError','delay must be a multiple of 10.')
            self.pars_dict.update({'Acquisition delay fix': delay})
    """
    def repetitions_period(self,rep_per=None):
        """Function used to get(def)/set the wait time between measurements,
        in microseconds"""
        if rep_per is None:
            return self.pars_dict['Repetitions period']
        else:
            self.pars_dict.update({'Repetitions period': rep_per})
    
    def DM_frequency(self,dm_freq=None):
        """Function used to get(def)/set the down mixed frequency, in MHz. def
        is 10 MHz"""
        if dm_freq is None:
            return self.pars_dict['DM frequency']
        else:
            self.pars_dict.update({'DM frequency': dm_freq})

    def DM_power(self,dm_power=None):
        """Function used to get(def)/set the LO power in dBm, def is 8 dBm"""
        if dm_power is None:
            return self.pars_dict['Downmixer power']
        else:
            self.pars_dict.update({'Downmixer power': dm_power})

    def __set_LO(self):
        freq = np.round(self._rogen.frequency()+self.DM_frequency()*1e-3,10)
        self._logen.frequency(freq)
        self._logen.power(self.DM_power())
        self._logen.output(1)
        
    def __set_dig(self,acq_length=None,acq_delay=None,bugfix=False):
        
        #digsigd seems to have a bug at the 1st acquisition, so I measure one more and discard the 1st
        AVE = self.averages()
        if bugfix and AVE>1:
            AVE+=1
        
        if acq_length is None:
            self._dig.points(self.dig_ch(), int(self._rogen.instr.pulse_width()*self._dig.SF()),AVE)
        else:
            self._dig.points(self.dig_ch(), int(acq_length*self._dig.SF()),AVE)
            
        #if acq_delay is not None:
            #self._dig.delay(self.dig_ch(),acq_delay)

        
        
        
        self._dig.register(4,int(self.repetitions_period()*100))
        if bugfix and self.averages()>1:
            self._dig.register(2,AVE) 
        else:
            self._dig.register(2,AVE)
            
        if acq_delay is None:
            self._dig.register(0,int(self._rogen.instr.pulse_delay()/10))
            
        else:
            self._dig.register(0,int(acq_delay/10))
   
    def generators_parameters_list(self):
        tmp={'DIG':self._dig.channel(self.dig_ch()).pars_dict}
        
        reg_list={'DIG-R0': self._dig.register(0),
                  'DIG-R1': self._dig.register(1),
                  'DIG-R2': self._dig.register(2),
                  'DIG-R3': self._dig.register(3),
                  'DIG-R4': self._dig.register(4)
                }
        
        tmp.update(reg_list)
        tmp.update({self._rogen.id:self._rogen.parameters_dictionary()})
        tmp.update({self._logen.id:self._logen.parameters_dictionary()})
        return tmp
        
#------------------------------------------------------------------------------- Custom exceptions
    class __BASEEXC(Exception):
        pass
    
    class _ROEXC(__BASEEXC):
        def __init__(self,Expression,Message):
            self.Expression = Expression
            self.Message = Message

#------------------------------------------------------------------------------- readout waves        
    def readout_waves(self,channel_list=None,acq_length=None,acq_delay=None,bugfix=True):
        """Function used to perform a readout and get all the waves in the pc 
        memory. It is possible to specify multiples channels, acq_length and 
        acq_delay in ns for future implementation or debug"""    
        if self._HVI is None:
            print('Load HVI first')
            return
        
        self.__set_LO()
        if acq_delay is not None:
            Channel_parameters = self._dig.channel(self.dig_ch()).pars_dict
            old_delay = Channel_parameters['Delay']
        self.__set_dig(acq_length,acq_delay,bugfix)
        
        self._HVI.start()
        
        
        self._dig.register(3,1)
        try:
            start_time = time.time()
            while(self._dig.register(3)==0):
                time.sleep(0.001)
                if time.time()-start_time > 20:
                    raise self._ROEXC('TIMEOUT','Exceeded 20 sec wait time in the acquisition loop')
        except KeyboardInterrupt:
            print('Interrupted')
            raise KeyboardInterrupt
        
        
        if channel_list is None:
            channel_list = [self.dig_ch(),]
        
        data = []
        for c in channel_list:
            tmp = self._dig.get_wave(c)
            if bugfix and self.averages()>1:
                tmp = tmp[1:] #discard the 1st
            data.append(tmp)
            
        
        self._dig.register(3,1) #stops the HVI
        
        if acq_delay is not None:
            Channel_parameters = self._dig.channel(self.dig_ch()).pars_dict
            Channel_parameters['Delay'] = old_delay
        
        if len(channel_list)==1:
            return data[0]
        else:
            return data
                
    def readout(self,volt_conversion=True,bugfix=True):
        """Function used to average the acquisitions and get the amplitude,
        if volt_conversion is True (def), the amplitude will be in Volts"""
            
        data = self.readout_waves(bugfix=bugfix)
        
        tot = np.average(data,0)
        fa = np.fft.fft(tot)/len(tot)
        
        
        
        index = np.int(np.round(self._rogen.instr.pulse_width()*self.DM_frequency()/1e3,12))
        
        tmp = np.abs(fa[index])*2
        
        
        if volt_conversion is True:
            tmp = digits_to_Vp(tmp,self._dig.amplitude(self.dig_ch()))
    
        return tmp