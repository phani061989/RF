


class CalibrationParameters(object):
    import numpy as np
    import copy
        
    
    
    def __init__(self, offI = 0., offQ = 0., amp_ratio = 1., phase_corr = 0.,dictionary=None):
        """
        This object contains the calibration parameters of an IQ mixer.
        
        It is possible to initialize the parameters when initializing the 
        object or do it later with the proper functions.
        
        Parameters
        ----------
        offI : float, optional
            The offset at the I entry of the IQ mixer (default: 0.). The value
            is given in (V).
        offQ : float, optional
            The offset at the Q entry of the IQ mixer (default: 0.). The value
            is given in (V).
        amp_ratio: float, optional
            The ratio between the amplitudes of the signals in I and Q 
            (default: 1.). 
        amp_channel : int, optional
            The channel of the input signal (default: 1).
        phase_corr : float, optional
            The phase correction between the two input signals (default: 0.). 
            The value is given in degrees and always in [0, 360]°. It is 
            applied to the signal in channel Q.
        
        """
        
        if dictionary is None:
            self.__calibration_dictionary={'Offset chI': 0,
                                     'Offset chQ': 0,
                                     'Amplitude ratio': 1,
                                     'Amplitude corrected channel': 'chI',
                                     'Phase correction chQ': 0,
                                     'Sideband': 'RSB',
                                     'Frequency': None,
                                     'AWG chI': None,
                                     'AWG chQ': None,
                                     'AWG frequency': None,
                                     'LSB': None,
                                     'Carrier': None,
                                     'RSB': None,
                                     'Calibration amplitude': None,
                                     'Calibration results': None,
                                     'LO power': 13}
            self.set_cal_parameters(offI, offQ, amp_ratio,  phase_corr)
        
        else:
            if type(dictionary) != dict:
                print('dictionary must be a python dictionary')
                raise TypeError
            
            self.__calibration_dictionary = dictionary

#------------------------------------------------------------------------------- Parameters definitions
    
    #-------------------------------------------------------------------- offI
    @property
    def offI(self):
        return self.calibration_dictionary['Offset chI']
    
    @offI.setter
    def offI(self,value):
        self.__dict_access('Offset chI',self.np.round(value,12))
    #-------------------------------------------------------------------- offQ
    @property
    def offQ(self):
        return self.calibration_dictionary['Offset chQ']
    
    @offQ.setter
    def offQ(self,value):
        self.__dict_access('Offset chQ',self.np.round(value,12))
        
    #-------------------------------------------------------------------- ratio
    @property
    def ratio(self):
        return self.calibration_dictionary['Amplitude ratio']
    
    @ratio.setter
    def ratio(self,value):
        self.__dict_access('Amplitude ratio',self.np.round(value,12))
    #-------------------------------------------------------------------- amp_corr_ch
    @property
    def amp_corr_ch(self):
        return self.calibration_dictionary['Amplitude corrected channel']
    
    @amp_corr_ch.setter
    def amp_corr_ch(self,value):
        if value.lower() == 'chi':
            self.__calibration_dictionary['Amplitude corrected channel'] ='chI'
        elif value.lower() == 'chq':
            self.__calibration_dictionary['Amplitude corrected channel'] ='chQ'
        else:
            print('Wrong channel inserted: {} / {chI,chQ}'.format(value))
            raise ValueError
        
        
    
    #-------------------------------------------------------------------- phase
    @property
    def phase(self):
        return self.calibration_dictionary['Phase correction chQ']
    
    @phase.setter
    def phase(self,value):
        self.__dict_access('Phase correction chQ',self.np.round(value % 360, 12))
        
    #-------------------------------------------------------------------- LOPWR
    @property
    def LOPWR(self):
        return self.calibration_dictionary['LO power']
    
    @LOPWR.setter
    def LOPWR(self,value):
        self.__dict_access('LO power',self.np.round(value,12))


    #-------------------------------------------------------------------- results
    @property
    def results(self):
        return self.calibration_dictionary['Calibration results']
    
    @results.setter
    def results(self,value):
        if len(value)!=3:
            print('Something is wrong, results should be a list of three elements')
            raise ValueError

        
        self.__calibration_dictionary['Calibration results'] = value

    
    #-------------------------------------------------------------------- dictionary
    @property
    def calibration_dictionary(self):
        return self.copy.deepcopy(self.__calibration_dictionary)
    
    @calibration_dictionary.setter
    def calibration_dictionary(self,value):
        pass



#--------------------------------------------------------------------------------- Set functions -------------
    def __dict_access(self,key,value):
        #print('here')#debug
        
        
        if self.np.isscalar(value):
            self.__calibration_dictionary[key] = value
        else:
            print('The value inserted is not a number: {}\n'.format(value))
            return
        
        
        
    
    
    def set_cal_parameters(self, offI = 0., offQ = 0., amp_ratio = 1., phase_corr = 0.):
        """
        Set the calibration parameters to specific values.
    
        Parameters
        ----------
        offI : float, optional
            The offset at the I entry of the IQ mixer (default: 0.). The value
            is given in (V).
        offQ : float, optional
            The offset at the Q entry of the IQ mixer (default: 0.). The value
            is given in (V).
        amp_ratio: float, optional
            The ratio between the amplitudes of the signals in I and Q 
            (default: 1.). 
        phase_corr : float, optional
            The phase correction between the two input signals (default: 0.). 
            The value is given in degrees and always divided by 360°. It is 
            applied to the signal in channel Q.
    
        Returns
        -------
        nothing: None
            This function sets the internal values of the class to the values
            given to the parameters.
    
        See Also
        --------
        __init__()
    
    
        Notes
        -----
        If no values whatsoever are given, then the function simply initializes
        all the parameters to the default values specified in Parameters.
        All the input values are rounded according to the function np.round(). 
        This is done to avoid infinite internal representations of certain 
        numbers.
    
        """
       
       
        
        self.offI = offI
        self.offQ = offQ
        self.ratio = amp_ratio
        self.phase = phase_corr
         
         

    
            
    
    #----------------------------------------------------------------------------------------------------------------------------Output functions, delete at end-------------------
    def LSB(self):
        return self.calibration_dictionary['LSB']
    
    def carrier(self):
        return self.calibration_dictionary['Carrier']
    
    def RSB(self):
        return self.calibration_dictionary['RSB']
    
    


    #--------------------------------------------------------------------------

    def cal_par_list(self):
        """
        Return the parameters of the calibration in a tuple.
    
        Parameters
        ----------
        none
    
        Returns
        -------
        tuple: tuple
            The return value is a tuple containing all the values of the 
            parameters set by the function set_cal_parameters().
            
            (offI, offQ, amp_ratio, amp_channel, phase_corr)
    
        See Also
        --------
        set_cal_parameters()
    
        """
        
        return (self.calibration_dictionary['Offset chI'], 
                self.calibration_dictionary['Offset chQ'], 
               self.calibration_dictionary['Amplitude ratio'],
               self.calibration_dictionary['Amplitude corrected channel'], 
               self.calibration_dictionary['Phase correction chQ'])
    
        
    #--------------------------------------------------------------------------
    
    def frequency(self, frequency = None, SB = 'R'):
        """
        Query or set the up-mixed frequency. It is possible to specify the 
        sideband that is calibrated.
    
        Parameters
        ----------
        frequency : float, optional
            The calibrated frequency (default: None). The value is given in 
            (GHz).
        SB : string, optional
            Defines which sideband is used for the calibration (default: 'R').
            The used convention is:
            'R' or 'r':         right
            'L' or 'l':         left
    
        Returns
        -------
        Tuple: tuple
            This function returns the frequency and the sideband in a tuple:
            
            (frequency, SB)
    
    
        Notes
        -----
        The frequency has to be an element of a certain range: [3, 12].
    
        """
        
        # If no values are given, return the current values.
        if frequency is None:
            return self.calibration_dictionary['Frequency'], self.calibration_dictionary['Sideband']
        
        
        # Ensure that the frequency is positive.
        frequency = abs(frequency)
        
        
        
        # Define the sideband for the calibration.
        if SB.upper() == 'R':
            self.__calibration_dictionary['Sideband'] = 'RSB'
        elif SB.upper() == 'L':
            self.__calibration_dictionary['Sideband'] = 'LSB'
        else:
            print('ERROR: wrong sideband specified')
            raise ValueError
        
        # Set the new value for the frequency.
        self.__dict_access('Frequency', self.np.round(frequency, 12) )
        #return self.calibration_dictionary['Frequency'], self.calibration_dictionary['Sideband']
        #calculate Sidebands
        self.Sidebands()
        

    
    #--------------------------------------------------------------------------    
    
    def AWG_parameters(self, AWG_freq = None, chI = 0):
        """
        Set or return the AWG parameters.
    
        Parameters
        ----------
        AWG_freq : float, optional
            The frequency of the AWG (default: None). The value is given in 
            (MHz)
        chI : int, optional
            The channel of the AWG connected to the I port (default: 0).
            NOTE: chQ will always be chI+1
    
        Returns
        -------
        Tuple: tuple
            This function returns the frequency and the channels of the AWG in 
            a tuple:
            
            (AWG_freq, chI, chQ)
    
        Notes
        -----
        The value for the AWG frequency is entered and returned in MHz. 
        Internally it is saved in GHz.
    
        """

        # Check if you want to set or return values.
        if AWG_freq is None:
            if self.calibration_dictionary['AWG frequency'] is None:
                return None,None,None,None
            return self.calibration_dictionary['AWG frequency'], self.calibration_dictionary['Amplitude corrected channel'], self.calibration_dictionary['AWG chI'],self.calibration_dictionary['AWG chQ']
        
        
        # Set the values.
        self.__dict_access('AWG frequency', self.np.round(AWG_freq, 9))
        self.__dict_access('AWG chI', int(chI))
        self.__dict_access('AWG chQ', int(chI+1) )
        
        #calculate Sidebands
        self.Sidebands()
        
    
    #--------------------------------------------------------------------------    
    
    def Sidebands(self):
        """
        Calculate and set the frequencies for the sidebands of the calibration.
    
        Parameters
        ----------
        none
    
        Returns
        -------
        Tuple: tuple
            This function returns the frequencies in the sidebands a tuple:
            
            (LSB_freq, carrier_freq, RSB_freq)
    
        Notes
        -----
        First it is checked if a value for the frequency and the AWG frequency
        have been entered. Otherwise the calculation would fail. The 
        frequencies are calculated based on which sideband is being calibrated.
    
        """
        
        # Check if initial setup is done.
        if self.calibration_dictionary['Frequency'] is None or self.calibration_dictionary['AWG frequency'] is None:
            #print('Setup not complete')
            return 0, 0, 0
        else:
            # Calculate the frequencies of the sidebands.
            AWG_freq = self.calibration_dictionary['AWG frequency']*1e-3
            if self.calibration_dictionary['Sideband'] == 'RSB':
                self.__calibration_dictionary['LSB'], self.__calibration_dictionary['Carrier'], self.__calibration_dictionary['RSB'] = self.np.round_([self.calibration_dictionary['Frequency'] - 2*AWG_freq, 
                                                                    self.calibration_dictionary['Frequency'] - AWG_freq, 
                                                                    self.calibration_dictionary['Frequency']], 
                                                                    9)
                return self.calibration_dictionary['LSB'], self.calibration_dictionary['Carrier'], self.calibration_dictionary['RSB']
            elif self.calibration_dictionary['Sideband'] == 'LSB':
                self.__calibration_dictionary['LSB'], self.__calibration_dictionary['Carrier'], self.__calibration_dictionary['RSB'] = self.np.round_([self.calibration_dictionary['Frequency'], 
                                                                    self.calibration_dictionary['Frequency'] + AWG_freq, 
                                                                    self.calibration_dictionary['Frequency'] + 2*AWG_freq], 
                                                                    9)
                return self.calibration_dictionary['LSB'], self.calibration_dictionary['Carrier'], self.calibration_dictionary['RSB']
            else:
                print('Sideband not yet specified!')
                raise ValueError
    

    #--------------------------------------------------------------------------

    def print_parameters(self):
        """
        Print all the parameters that have already been initialized.
    
        Parameters
        ----------
        none
    
        Returns
        -------
        nothing: None
    
        Notes
        -----
        Depending on the setup, a different amount of parameters is printed.
        The output looks always at least like:
        
        offI: {}V
        offQ: {}V
        amp_ratio: {}
        amp_channel: {}
        phase_corr: {} deg
        
        If the AWG has been initialized the following is added:
        AWG_frequency: {} MHz
        chI: {}
        chQ: {}
        
        If the sidebands have been initialized the following is added:
        LSB: {} GHz
        LO: {} GHz
        RSB: {} GHz
        calibrated for: {}
    
        """
        
        text = '\noffI: {}V\noffQ: {}V\namp_ratio: {}\namp_channel: {}\nphase_corr: {} deg\n'.format(*self.cal_par_list())
        text2, text3 = '', ''
        
        if self.calibration_dictionary['AWG frequency'] is not None:
            text2 = 'AWG_frequency: {} MHz\nch_to_corr:{}\nchI: {}\nchQ: {}\n'.format(*self.AWG_parameters())
        if self.calibration_dictionary['Frequency'] is not None:
            text3 = 'LSB: {} GHz\nLO: {} GHz\nRSB: {} GHz\ncalibrated for: {}\n'.format(*self.Sidebands(),self.calibration_dictionary['Sideband'])
            
        print(text + text2 + text3)
    
    #--------------------------------------------------------------------------    
    def cal_amplitude(self,cal_amplitude = None):
        if cal_amplitude is None:
            return self.calibration_dictionary['Calibration amplitude']
        
        self.__dict_access('Calibration amplitude',self.np.round(cal_amplitude,12))

        
    
    
    #--------------------------------------------------------------------------
    def save(self, fname, Force = False):
        """
        Save the calibration into a file.
    
        Parameters
        ----------
        fname : string
            The path and filename for where to save the file.
        Force : bool, optional
            Defines if an already existing file should be overwritten 
            (default: False).
    
        Returns
        -------
        nothing : None
    
        Notes
        -----
        First it is checked if a value for the frequency and the AWG frequency
        have been entered. Otherwise the calibration is not yet completed. Then
        the folder is search and created if needed. The file extension is then 
        set to '*.cal'. If a file with the given name already exists in this 
        directory and the parameter Force is True, then the existing file is
        overwritten.
    
        """
        
        import os
        from UtilitiesLib import dictionarytofile
        
        if self.calibration_dictionary['Frequency'] is None or self.calibration_dictionary['AWG frequency'] is None:
            print('Calibration not complete, impossible to save!\n')
            raise Exception('CALERR')
        
        
        #Split fname in folder and filename
        folder = os.path.split(fname)[0]
        file_name = os.path.split(fname)[1]

        #Create directory if not already existent
        if not os.path.exists(folder) and folder:
            os.makedirs(folder)

        # Check for file extension
        if file_name[-4:].lower() != '.cal':
            file_name += '.cal'

        # Append Folder and be adaptive to windows, etc.
        file_name = os.path.normpath(os.path.join(folder, file_name))

        # Check for Overwrite
        if not Force:
            if os.path.isfile(file_name):
                print('File already exists!\n')
                raise Exception('FILEEXISTSERR')
        
        dictionarytofile(self.calibration_dictionary,fname)

    #--------------------------------------------------------------------------

    