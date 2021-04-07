from UtilitiesLib import progressive_plot_2d
#from IPython import display
import lmfit
from .IQCAL_base import IQCal_base


class IQCal_KEYAWG(IQCal_base):
    from .CALPAR import CalibrationParameters
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    
    
    #----------------------------------------------------------------------------------------------------------------------------Initialization of the Class-----------------------
    
    def __init__(self, AWG, sgLO, SpecAna = 'SH',OS='W', AWG_channel_cal_amplitude = 1., cal_file = None):
        """
        This object contains the calibration parameters of an IQAM.
        
        It is possible to initialize the parameters when initializing the 
        object or do it later with the proper functions.
        
        Parameters
        ----------
        AWG : object
            Initialized AWGSIG object
        sgLO : initialized Instuments.SG()
            
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
        
        self._AWG = AWG
        
        
        self.AWG_calibration_amplitude( AWG_channel_cal_amplitude)                             # Amplitude of the AWG
        
        
        
    #----------------------------------------------------------------------------------------------------------------------------Initialization of the instruments-----------------
    
    
    def AWG_calibration_amplitude(self,amplitude=None):
        '''This function changes the AWG channel amplitude used during the calibration'''
        
        if amplitude is None:
            return self._amplitude
        
        max_amp = self._AWG.channel(0)._AWGChannel__AMP_MAX
        
        if amplitude <0 or amplitude > max_amp:
            print('Wrong AWG amplitude inserted: [0, {}] V\n'.format(max_amp))
            raise Exception('AWGAMPERR')
        self._amplitude = amplitude
        self.calibration.cal_amplitude(amplitude)
        
    def set_AWG(self, AWG_freq, chI = 0):
        """
        Set the values of the AWG using the methods provided from the 
        CalibrationParameters class.
    
        Parameters
        ----------
        AWG_freq : float
            The frequency for the AWG (MHz) used in this calibration.
        chI : int, optional
            The channel for the I signal (default: 0).
        
        NOTE: chQ : will always be chI+1
    
        Returns
        -------
        nothing : None
    
        """
        
        #amp_corr_ch check is done in the function
        self._AWG.mode(chI,1) #SIN
        self._AWG.mode(chI+1,1) #SIN
        self._AWG.modulation(chI,0)
        self._AWG.modulation(chI+1,0)
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
        
        self._AWG.apply_correction(self.calibration.calibration_dictionary,self._amplitude)



 
    #--------------------------------------------------------------------------
    
    def initialize_calibration(self, AWG_pulse, frequency,SB = 'R', LO_pwr=13):
        """
        Initializes all the parameters for the calibration of the experiment. 
        
        Parameters
        ----------
        AWG_pulse : list: [awg_freq,awg_chI]
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




    

    
