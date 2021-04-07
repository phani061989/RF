from UtilitiesLib import progressive_plot_2d
#from IPython import display
import lmfit
from .IQCAL_base import IQCal_base
from ziUtilities import apply_correction as apc

class IQCal_ZIAWG(IQCal_base):
    from .CALPAR import CalibrationParameters
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    
    
    
    #----------------------------------------------------------------------------------------------------------------------------Initialization of the Class-----------------------
    
    def __init__(self, device_id, sgLO, SpecAna,OS='W', AWG_channel_cal_amplitude = 1., cal_file = None):
        """
        This object contains the calibration parameters of an IQAM.
        
        It is possible to initialize the parameters when initializing the 
        object or do it later with the proper functions.
        
        Parameters
        ----------
        AWG : object
            Initialized AWGSIG object
        sgLO: Initialized Instruments.SG()
            
        SpecAna : string, optional
            The spectrum analyzer used for the experiment (default: 'SH'). The 
            two options are:
            'SHN':               Signal Hound N
            'RS':               Rohde Schwarz
        AWG_channel_cal_amplitude : float, optional
            The amplitude for the AWG channels given in (V) (default: 1.).
        cal_file : string, optional
            The filepath to a saved calibration file (default: None).
        
        Notes
        -----
        If a calibration file is used, you will need to set generators. 
        Only the parameters of the calibration.calibration_dictionary 
        are initialized.
        
        """
        
        
        
        
        
        
        IQCal_base.__init__(self,sgLO, SpecAna,OS, cal_file)
        self.connect_AWG(device_id)
        
        self.AWG_calibration_amplitude( AWG_channel_cal_amplitude)                             # Amplitude of the AWG
        
    
    def __check_channel_chI(self,num):
        if num<1 or num>7:
            print('Wrong channel inserted, chI can be {1,3,5,7}\n')
            raise ValueError
        
        
        if num%2 == 0:
            print('Wrong channel inserted, chI can be {1,3,5,7}\n')
            raise ValueError
        
        
    
    def connect_AWG(self,device_id):
        from zhinst.utils import create_api_session
        self._dev_ID = device_id
        (zi,device_id,  _) = create_api_session(device_id, 6, required_devtype='HDAWG',required_err_msg='')
        
        # Create an instance of the AWG Module
        awg = zi.awgModule()
        awg.set('awgModule/device', self._dev_ID)
        
        self._daq = zi
        self._awg = awg
    
    #----------------------------------------------------------------------------------------------------------------------------Initialization of the instruments-----------------
    
    
    def AWG_calibration_amplitude(self,amplitude=None):
        '''This function changes the AWG channel amplitude used during the calibration'''
        
        if amplitude is None:
            return self._amplitude
        
        max_amp = self._daq.getDouble('/%s/sigouts/0/range'%self._dev_ID)
        
        if amplitude <0 or amplitude > max_amp:
            print('Wrong AWG amplitude inserted: [0, {}] V\n'.format(max_amp))
            raise Exception('AWGAMPERR')
        self._amplitude = amplitude
        self.calibration.cal_amplitude(amplitude)
        
    def set_AWG(self, AWG_freq, chI = 1):
        """
        Set the values of the AWG using the methods provided from the 
        CalibrationParameters class.
    
        Parameters
        ----------
        AWG_freq : float
            The frequency for the AWG (MHz) used in this calibration.
        amp_corr_ch: int, optional
            the channel that will be amplitude corrected (def 0)
        chI : int, optional
            The channel for the I signal (default: 0).
            IMPORTANT: chQ will always be chI+1, the oscillator is associated based to the channel couple.
            
            
        
        Returns
        -------
        nothing : None
    
        """
        
        self.__check_channel_chI(chI)
        
        #stop AWG if it is on
        self._daq.setInt('/{}/awgs/{}/enable'.format(self._dev_ID,int(chI/2)), 0) #stop
        self._daq.setInt('/{}/awgs/{}/enable'.format(self._dev_ID,int(chI/2)), 0) #stop
        
        #set mod off
        self._daq.setInt('/{}/awgs/{}/outputs/0/modulation/mode'.format(self._dev_ID,int(chI/2)),0 )
        self._daq.setInt('/{}/awgs/{}/outputs/1/modulation/mode'.format(self._dev_ID,int(chI/2)),0)
        
        self._daq.setInt('/{}/sines/{}/enables/1'.format(self._dev_ID,chI-1), 0) #chI to sin1
        self._daq.setInt('/{}/sines/{}/enables/1'.format(self._dev_ID,chI), 0) #chQ to sin2

    
        
        self.calibration.AWG_parameters(AWG_freq, chI)




    #--------------------------------------------------------------------------
    
    def apply_correction(self):
        """
        Apply the values determined by the calibration to the AWG.
    
        Parameters
        ----------
        none : None
    
        Returns
        -------
        nothing: None
    
        """

        
        freq, dummy,chI, chQ = self.calibration.AWG_parameters()
        if freq is None or self.calibration.frequency() is None:
            print('Init not completed\n')
            return
        
        apc(self._daq,self._dev_ID,self.calibration.calibration_dictionary,self._amplitude)



 
    #--------------------------------------------------------------------------
    
    def initialize_calibration(self, AWG_pulse, frequency,SB = 'R', LO_pwr=13):
        """
        Initializes all the parameters for the calibration of the experiment. 
        
        Parameters
        ----------
        AWG_pulse : list: [awg_freq,ch_to_corr,awg_chI,awg_chQ]
            This list contains all the AWG parameters for the calibration of 
            the I/Q mixer.
        frequency: frequency to calibrate
        SB : string, optional
            Defines which sideband is used for the calibration (default: 'R').
            The used convention is:
            'R' or 'r':         right (def)
            'L' or 'l':         left
        LO_pwr:              default to 13 dbm
        
        
        Returns
        -------
        nothing: None
        
        Notes
        -----
        If values for the LO are given, the first one is always used for the 
        power and the second one for the channel. It is possible to give zero, 
        one or two values for those parameters. Depending on the number of the 
        given values, the default values are used.
        
        """
        
        self.set_AWG(*AWG_pulse)
        self.set_frequency(frequency,SB)
        self.apply_correction()
        
        
        self.set_LO(LO_pwr)



    #----------------------------------------------------------------------------------------------------------------------------Measurement of the Sidebands----------------------
    


    
    #----------------------------------------------------------------------------------------------------------------------------Calibration of the setup--------------------------
        



