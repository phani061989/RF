# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:21:49 2019

@author: User
"""

from UtilitiesLib import progressive_plot_2d
#from IPython import display
import lmfit
import Instruments
import time

class IQCal_base(object):
    from .CALPAR import CalibrationParameters
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    
    
    #----------------------------------------------------------------------------------------------------------------------------Initialization of the Class-----------------------
    
    def __init__(self,sgLO, SpecAna,OS='W', calibration = None):
        """
        This object contains the calibration parameters of an IQAM.
        
        It is possible to initialize the parameters when initializing the 
        object or do it later with the proper functions.
        
        Parameters
        ----------
        AWG : object
            Initialized AWGSIG object
        sgLO : initialized Instrumens.SG
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
        
        self.connect_specana(SpecAna,OS)
        
        
        self._sgLO = sgLO
        self._SpecAnaID = None
        
        
        if calibration is not None:
            self.load_calibration(calibration)
        else:
            self.calibration = self.CalibrationParameters()
        
        #self.AWG_calibration_amplitude( AWG_channel_cal_amplitude)  # Amplitude of the AWG
        self.__MINFR = 3 #GHz
        self.__MAXFR = 12 #GHz
        
        
        
    def load_calibration(self,calibration):
            if type(calibration)==str: #User inserted a path
                from UtilitiesLib import filetodictionary
                cal_dict = filetodictionary(calibration)
            else:
                cal_dict = calibration
        
            self.calibration = self.CalibrationParameters(dictionary=cal_dict)        
        
        
    

    def connect_specana(self,ID=None,OS='W'):
        """Connect the specified spectrum analyzer using the driver for the  specified operative system
        
        args:
            - ID: 'SHN' for Signal Hound N
                  'RS' for Rohde and Schwarz (ignores OS)
            - OS: 'W' (def) for windows
                  'S' for Linux
        """
        ID = ID.upper()
        OS = OS.upper()
        
        if ID is None and self._SpecAnaID is None:
            print('No spectrum analyzer ID inserted\n')
            return
    
        if ID is None:
            ID = self._SpecAnaID
        
        if ID == 'RS':
            
            self._SpecAna = Instruments.SA('SA1','').instr
            self._SpecAna._inst.timeout=5e3
        elif ID[:2] == 'SH':
            from SIGNALHOUND import SignalHound
            self._SpecAna = SignalHound(ID,OS=OS)
        else:
            print('Wrong spectrum analyzer specified')
            raise Exception('SpecAnaERR')
        
        self._SpecAnaID = ID
        
    def close_SA_connection(self):
        """
        End the connection with the spectrum analyzer. Always call at the end 
        of the program!
    
        Parameters
        ----------
        none : None
    
        Returns
        -------
        nothing : None
    
        Notes
        -----
        This function closes the connection with whatever spectrum analyzer is 
        currently in use. It has to always be called at the end of the program 
        to ensure that the spectrum analyzer doesn't keep running indefinetely.
    
        """
        
        if self._SpecAnaID == 'RS':
            self._SpecAna.close()
        else:
            self._SpecAna.CloseDevice()
            
        self._SpecAna = None
            
        
    #--------------------------------------------------------------------------- Exceptions ------------------------------------------
    class __IQEXC(Exception):
        pass
    
    class SLOPEEXC(__IQEXC):
        def __init__(self,expression,message):
            self.expression=expression
            self.message=message
    
    
    #----------------------------------------------------------------------------------------------------------------------------Initialization of the instruments-----------------
    
    
    


    #--------------------------------------------------------------------------
    
    def set_frequency(self, frequency, SB = 'R'):
        """
        Set the up-mixed frequency for the experiment. It is possible to 
        specify the sideband that is calibrated.
    
        Parameters
        ----------
        frequency : float
            The calibrated frequency for the experiment. The value is given in 
            (GHz).
        SB : string, optional
            Defines which sideband is used for the calibration (default: 'R').
            The used convention is:
            'R' or 'r':         right
            'L' or 'l':         left
    
        Returns
        -------
        nothing: None
    
        """
        
        self.calibration.frequency(frequency, SB)





    #--------------------------------------------------------------------------
    
    def set_LO(self, power = 13):
        """
        This function sets the frequency and power of the LO signal generator.
    
        Parameters
        ----------
        power : float, optional
            The power provided by the LO (default: 13). The value is given in 
            (dbm).
        channel : int, optional
            The channel of the LO (default: 1).
    
        Returns
        -------
        nothing: None
    
        """
        
        
        
        # Parameters
        LOfreq = self.calibration.Sidebands()[1]
        
        # LO setup
        
        self._sgLO.power(power)
        self.calibration.LOPWR = power
        self._sgLO.frequency(self.np.round(LOfreq,8))
        self._sgLO.instr.reference('EXT')
        self._sgLO.output(1)
        
        try:
            self._sgLO.instr.pulse_triggered(0, 0, 0)
            self._sgLO.instr.alc('ON')
        except AttributeError:
            pass
        
        
        


 
    #--------------------------------------------------------------------------
    




    #----------------------------------------------------------------------------------------------------------------------------Measurement of the Sidebands----------------------
    
    def measure_SB(self, plot = False, print_diff = False, bands = 'ALL', averages=5,peak_span=5,ref_level=0,BW=100e3):
        """
        Measure the power in the sidebands. 
        
        Parameters
        ----------
        plot : bool, optional
            This defines if the measured power should be plotted (default: False).
        print_diff : bool, optional
            This defines if the power difference between the two sidebands 
            is to be printed (default: False).
        bands : string, optional
            This string sets which sidebands are to be measured (default: ALL):
            'ALL':              Measure all three peaks
            'R' or 'r':         Measure only the right sideband
            'C' or 'c':         Measure only the carrier
            'L' or 'l':         Measure only the left sideband
        
        averages:           default to 5
        peak_span:          default to 5 MHz
        ref_level:          default is 0 dBm
        BW:                 default to 100e3 Hz (RBW of the SA)
            Only the 'RS' spectrum analyzer can use averages. The 
            Signalhound only uses the peak_span variable.
        
        Returns
        -------
        Tuple: tuple
            The return value of this function is a tuple that always contains 
            three values, depending on which bands were measured:
            'ALL':              (max left peak, max center peak, max right peak)
            'R' or 'r':         (0, 0, max right peak)
            'C' or 'c':         (0, max center peak, 0)
            'L' or 'l':         (max left peak, 0, 0)
        
        Notes
        -----
        The parameter bands is only useful if the Signalhound is used. The 'RS'
        spectrum analyzer always measures the entire width of the sidebands.
        
        """
        
        
        if self._SpecAnaID == 'RS':
            left, middle, right = self.__measure_SB_RS(averages, peak_span, plot,BW)
        else:
            left, middle, right = self.__measure_SB_SH(ref_level,peak_span, plot, bands,BW)
        
        if print_diff:
            print("The difference between the sidebands is {0:.5f} dB".format(abs(right - left)))
        
        return left, middle, right
    

    def __measure_SB_SH(self, ref_level,peak_span, plot, bands,BW):
        """
        This function measures the power in the sidebands using a SignalHound. 
        It can be chosen which sideband is measured.
        
        """
        
        peak_span /= 1e3        
        
        # Set up the Spectrum Analyzer
        self._SpecAna.default_settings()
        self._SpecAna.Reference()
        self._SpecAna.ConfigLevel(ref_level)
        self._SpecAna.BandWidth(BW)                                          # in Hz
        
        # Set the variables
        LSB = self.calibration.LSB()
        Carrier = self.calibration.carrier()
        RSB = self.calibration.RSB()
        
        #------------------------------------------------------        
        
        if bands == 'ALL':
            #Measure left Sideband
            self._SpecAna.CenterSpan(LSB, peak_span)
            self._SpecAna.Initiate()
            measL = self._SpecAna.GetSweep()
        
            #------------------------------------------------------
    
            #Measure Carrier
            self._SpecAna.CenterSpan(Carrier, peak_span)
            self._SpecAna.Initiate()
            measC = self._SpecAna.GetSweep()
            
            #------------------------------------------------------
    
            #Measure right Sideband
            self._SpecAna.CenterSpan(RSB, peak_span)
            self._SpecAna.Initiate()
            measR = self._SpecAna.GetSweep()
            
            #------------------------------------------------------
            if plot:
                self.plt.figure()
                measL.plot(engine = 'p')
                measC.plot(engine = 'p')
                measR.plot(engine = 'p')
                
                self.plt.xlim([measL.x[0] - peak_span, measR.x[-1] + peak_span])
                
                min_y = self.np.min((measL.y.min(), measC.y.min(), measR.y.min(), ))
                max_y = self.np.max((measL.y.max(), measC.y.max(), measR.y.max(), ))
                
                self.plt.ylim([min_y, max_y + 2])
                self.plt.show()
            
            #------------------------------------------------------       
        
            #return data
            return measL.y.max(), measC.y.max(), measR.y.max()
            
        elif bands.upper() == 'L':
            #Measure left Sideband
            self._SpecAna.CenterSpan(LSB, peak_span)
            self._SpecAna.Initiate()
            measL = self._SpecAna.GetSweep()
            
            #------------------------------------------------------
            if plot is True:
                measL.plot(engine = 'p')
                
                self.plt.xlim([measL.x[0] - peak_span, measL.x[-1] + peak_span])
                
                min_y = measL.y.min()
                max_y = measL.y.max()
                
                self.plt.ylim([min_y, max_y + 2])
            
            #------------------------------------------------------       
        
            #return data
            return measL.y.max(), 0, 0
                
        elif bands.upper() == 'C':
            #Measure Carrier
            self._SpecAna.CenterSpan(Carrier, peak_span)
            self._SpecAna.Initiate()
            measC = self._SpecAna.GetSweep()
            
            #------------------------------------------------------
            if plot is True:
                measC.plot(engine = 'p')
                
                self.plt.xlim([measC.x[0] - peak_span, measC.x[-1] + peak_span])
                
                min_y = measC.y.min()
                max_y = measC.y.max()
                
                self.plt.ylim([min_y, max_y + 2])
            
            #------------------------------------------------------       
        
            #return data
            return 0, measC.y.max(), 0
            
        elif bands.upper() == 'R':
            #Measure right Sideband
            self._SpecAna.CenterSpan(RSB, peak_span)
            self._SpecAna.Initiate()
            measR = self._SpecAna.GetSweep()
            
            #------------------------------------------------------
            if plot is True:
                measR.plot(engine = 'p')
                
                self.plt.xlim([measR.x[0] - peak_span, measR.x[-1] + peak_span])
                
                min_y = measR.y.min()
                max_y = measR.y.max()
                
                self.plt.ylim([min_y, max_y + 2])
            
            #------------------------------------------------------       
        
            #return data
            return 0, 0, measR.y.max()
            
            

        





    def __measure_SB_RS(self, ave, peaks_span, plot,BW):
        """
        This function measures the power in the sidebands using the Rohde Schwarz
        spectrum analyzer.
        
        """
        
        import DataModule as dm
        
        self._SpecAna._inst.timeout= 5e3 + ave*0.2e-3
        # Set the variables
        peaks_span /= 1e3                                                       #MHz -> GHz
        LSB = self.calibration.LSB()
        carrier = self.calibration.carrier()
        RSB = self.calibration.RSB()
        AWG_freq = self.calibration.AWG_parameters()[0]
        
        # Initialize the spectrum analyzer
        self._SpecAna.f_center(carrier*1e9)
        self._SpecAna.f_span(2.5e6*AWG_freq)
        self._SpecAna.averages(ave)
        self._SpecAna.bw_res(BW)
        #self._SpecAna.Single()
        
    
        
        # Start the spectrum analyzer
        #self._SpecAna.run()
        #self.time.sleep(0.001)
        #while self._SpecAna.count() < ave:
            #self.time.sleep(0.0001)
        
        self._SpecAna.sweep_time('auto')
        self._SpecAna.initiate_data()
        ts = time.time()    # Second counter
        while time.time()-ts<10:
            time.sleep(1e-3)
            if self._SpecAna.com('*OPC') == 1:
                break
        
        # Completing the measurement
        x, y, trash = self._SpecAna.read_data()
        tmp = dm.data_table([x/1e9, y], ['Frequency (GHz)', 'PSD (dBm)'])
        
        # Obtain the data for the LSB, carrier and RSB
        tmp.select([LSB - peaks_span, LSB + peaks_span])
        left = tmp.return_y().max()
        
        tmp.select([carrier - peaks_span, carrier + peaks_span])
        center = tmp.return_y().max()
        
        tmp.select([RSB - peaks_span, RSB + peaks_span])
        right = tmp.return_y().max()
        
        # Plot the measurement if necessary
        if plot:
            tmp.select()
            tmp.plot(engine = 'p')
        
        return left, center, right


    
    #----------------------------------------------------------------------------------------------------------------------------Calibration of the setup--------------------------
        


    def __calibrate_for_minimum(self, dict_pointer,band,index,step, timeout,plot,min_points,  args):
        """
        Adjust the sideband  until the peak hight of 
        the sideband to be calibrated is in a minimum. If no minimum is found 
        after timeout (s), the current value will be returned.
        
               
        """
        min_points = int(min_points)
        
        def measure(x):
            if dict_pointer == 'Offset chI':
                self.calibration.offI = x
            elif dict_pointer == 'Offset chQ':
                self.calibration.offQ = x
            elif dict_pointer == 'Amplitude ratio':
                self.calibration.ratio = x
            elif dict_pointer == 'Phase correction chQ':
                self.calibration.phase = x
            else:
                print('Unexpected error')
                raise ValueError
                
            self.apply_correction()
            return x,self.measure_SB(False,False, band, *args)[index]
        
        #------------------------------------------------------
        def detect_slope(y0,y1,y2):
                slope1 =  y1 - y0
                slope2 =  y2 - y1
                
                
                if slope1 > 0 and slope2 > 0:
                        status =  'up'
                elif slope1 < 0 and slope2 < 0:
                        status = 'down'
                elif slope1 > 0 and slope2 < 0:
                        status = 'max'
                elif slope1 < 0 and slope2 > 0:
                        status = 'min'
                else:
                        raise self.SLOPEEXC('cal ratio values','slope1 and 2 are zero')
                    
                    
                return status
        
        def refine_scan(x,y,min_points):
            
            if min_points %2 == 0:
                min_points+=1
                
            if len(x)<min_points:
                sweep = self.np.linspace(self.np.min(x),self.np.max(x),min_points)
            else:
                x,y = self.np.array(x),self.np.array(y)
                indexes = self.np.argsort(x)
                return x[indexes],y[indexes]
            
            for s in sweep:
                try:
                    x.index(s)
                except ValueError:
                    x.append(s)
                    y.append(measure(x[-1])[1])
            
            x,y = self.np.array(x),self.np.array(y)
            indexes = self.np.argsort(x)
            return x[indexes],y[indexes]
        
        #------------------------------------------------------
        
        start = self.time.time()
        x,y = [],[]
        
        starting_point = self.calibration.calibration_dictionary[dict_pointer] #at the begin the central point is x[1],y[1]
        
        x.append(starting_point-step)
        y.append(measure(x[0])[1])
        x.append(x[-1]+step)
        y.append(measure(x[1])[1])
        x.append(x[-1]+step)
        y.append(measure(x[2])[1])
        
        if plot:
            progressive_plot_2d(x,y,'o')
        
        slope = detect_slope(*y)
        
        try:
            while(1):
                
                if  slope == 'up':
                    starting_point -=step
                    tmp=measure(starting_point-step)  #adding 1 point on the left
                    x.insert(0,tmp[0])
                    y.insert(0,tmp[1])
                    
                    slope = detect_slope(*y[:3])
                    if plot:
                        progressive_plot_2d(x,y,'o')
                    continue
                if slope == 'down':
                    starting_point +=step
                    tmp=measure(starting_point+step) #adding 1 point on the right
                    x.append(tmp[0])
                    y.append(tmp[1])
                    slope = detect_slope(*y[-3:])
                    if plot:
                        progressive_plot_2d(x,y,'o')
                    continue
                if slope =='min':
                    
                    x,y = refine_scan(x,y,min_points)
                    if plot:
                        progressive_plot_2d(x,y,'o')
                    
                    return [x[self.np.argmin(y)],self.np.min(y)],x,y,False
                    
                elif slope == 'max':
                    import random
                    starting_point += random.uniform(-2*step, 2*step)
                
                if self.time.time() - start > timeout:
                    print('TIMEOUT reached')
                    center = x[-2]
                    return center,x,y,True
                
        except KeyboardInterrupt:
                return 0,x,y,True
            
    def _fit_data(self,x,y,init=None,points=10001,plot=True):
        #lorentzian_fit_pars_labels = ['center', 'width', 'offset', 'amplitude']
        def func2(x,pars):
    
            y=pars['width']**2/((x-pars['center'])**2 +pars['width']**2 )   
    
            return pars['amplitude'] * y/y.max() + pars['offset']
    
    
    
    
        def diff_func2(pars,x,y):
            return y-func2(x,pars)
    
        if init is None:
            p0l = lmfit.Parameters()
            p0l.add('center',x[self.np.argmin(y)])
            
            cen_index = self.np.argmin(y)
            
            p0l.add('width',abs(x[cen_index-1]-x[cen_index+1]))
            p0l.add('offset',self.np.max(y))
            p0l.add('amplitude',self.np.min(y)-self.np.max(y))
    
    
        results = lmfit.minimize(diff_func2,p0l,args=(x,y))
        
        x2=self.np.linspace(x[0],x[-1],points)
        y2=func2(x2,results.params)
        
        if plot:
            self.plt.plot(x,y,'bo')
            self.plt.plot(x2,y2,'-r')
        
        return x2[self.np.argmin(y2)],p0l,results


    
    def _check_amp_ratio_limit(self):
        """When amplitude ratio is larger than 1, it is better to assign its reciprocalto the other quadrature channel"""
        tmp = self.calibration.ratio
        
        if  tmp > 1: #if amp_corr>1 then set chQ as corr channel and 1/ratio as amp_corr
            self.calibration.ratio= 1/tmp
            self.calibration.amp_corr_ch = 'chQ'
            
            
    


    #--------------------------------------------------------------------------
    
    def calibrate_offset(self, step = 0.1, min_points=11,timeout = 2,  plot = True, averages=5,peak_span=5,ref_level=0,BW=100e3):
        """
        Calibrates the offset of the inputs in I and Q. The peak at the carrier
        frequency should be minimized. 
        
        Parameters
        ----------
        step : float, 0.1V def
            Gives the step width the algorithm takes between changes in the 
            offset 
        min_points: int,11 def
            after finding a minimum, the algorithm will ensure that 
            at least min_points are measured between the start and minimum
        timeout: float (s), def 10
            If a minimum can' t be found after timeout (s), the last value will
            be returned
        plot : bool, def True
            It shows the plots of the minimization procedure
        
        averages:           default to 5
        peak_span:          default to 5 MHz
        ref_level:          default is 0 dBm
        BW:                 default to 100e3 Hz (RBW of the SA)
            Only the 'RS' spectrum analyzer can use all values. The 
            Signalhound only uses the peak_span variable.
        
        Returns
        -------
        Tuple: tuple
            The return value of this function is a tuple that always contains 
            three lists:
            
            [offI,offQ]:        The offsets values, already inserted in 
                                calibration.calibration_dictionary
            [xI,yI]:            the x and y axis of the minimization for offI
            [xQ,yQ]:            the x and y axis of the minimization for offQ
        
        
        
        """
        
        
        
            
        
        
        try:
            if plot is True:
                self.plt.figure()
                self.plt.xlabel('Offset chI (V)')
                self.plt.ylabel('Carrier (dBm)')
            
            #step1 : measure at least min_points around the minimum
            cenI,xI,yI,timeout_flag = self.__calibrate_for_minimum('Offset chI','C',1,step,timeout,plot,min_points,[averages,peak_span,ref_level,BW])
            #step 2: lorentzian fit to find center
            if not timeout_flag:
                cenI,p0,res = self._fit_data(xI,yI)
            #step 3: set the center
            self.calibration.offI = cenI
            
            if plot is True:
                self.plt.figure()
                self.plt.xlabel('Offset chQ (V)')
                self.plt.ylabel('Carrier (dBm)')
            
            #step 4: repeat for chQ
            cenQ,xQ,yQ,timeout_flag = self.__calibrate_for_minimum('Offset chQ','C',1,step,timeout,plot,min_points,[averages,peak_span,ref_level,BW])
            
            if not timeout_flag:
                cenQ,p0,res = self._fit_data(xQ,yQ)
            
            self.calibration.offQ=cenQ
            
            return [cenI,cenQ],[xI,yI],[xQ,yQ]
        
        except KeyboardInterrupt:
            return [cenI,cenQ],[xI,yI],[xQ,yQ]




    #--------------------------------------------------------------------------


    def calibrate_amp_ratio_and_phase(self, step_ratio = 0.1, step_phase = 5, min_points=11,timeout = 2,  plot = True, averages=5,peak_span=5,ref_level=0,BW=100e3):
        """
        Calibrates the amplitude ration and phase correction. The peak at the 
        unwanted sideband frequency will be minimized. 
        
        Parameters
        ----------
        step_ratio : float, 0.1 def
            Gives the step width the algorithm takes between changes in the 
            amplitude ratio 
        step_phase : float (deg), 5 def
            Gives the step width the algorithm takes between changes in the 
            phase correction 
        min_points: int,11 def
            after finding a minimum, the algorithm will ensure that 
            at least min_points are measured between the start and minimum
        timeout: float (s), def 10
            If a minimum can' t be found after timeout (s), the last value will
            be returned
        plot : bool, def True
            It shows the plots of the minimization procedure
        
        averages:           default to 5
        peak_span:          default to 5 MHz
        ref_level:          default is 0 dBm
        BW:                 default to 100e3 Hz (RBW of the SA)
            Only the 'RS' spectrum analyzer can use all values. The 
            Signalhound only uses the peak_span variable.
        
        Returns
        -------
        Tuple: tuple
            The return value of this function is a tuple that always contains 
            three lists:
            
            [ratio,phase]:        The offsets values, already inserted in 
                                calibration.calibration_dictionary
            [xAR,yAR]:            the x and y axis of the minimization for the ratio
            [xPH,yPH]:            the x and y axis of the minimization for the phase
        
        
        
        """
        
        
        
        band_label = self.calibration.frequency()[1]
        if band_label == 'RSB':
            band = 'L'
            index = 0
        else:
            band = 'R'
            index = 2
        
        
        try:
            if plot is True:
                self.plt.figure()
                self.plt.xlabel('Amplitude ratio')
                self.plt.ylabel(band_label+' (dBm)')
            
            #step1 : measure at least min_points around the minimum
            cenI,xI,yI,timeout_flag = self.__calibrate_for_minimum('Amplitude ratio',band,index,step_ratio,timeout,plot,min_points,[averages,peak_span,ref_level,BW])
            #step 2: lorentzian fit to find center
            if not timeout_flag:
                cenI,p0,res = self._fit_data(xI,yI)
                
            #step 3: set the center
            self.calibration.ratio=cenI
            
            if plot is True:
                self.plt.figure()
                self.plt.xlabel('Phase correction chQ (deg)')
                self.plt.ylabel(band_label+ ' (dBm)')
            
            #step 4: repeat for phase
            cenQ,xQ,yQ,timeout_flag = self.__calibrate_for_minimum('Phase correction chQ',band,index,step_phase,timeout,plot,min_points,[averages,peak_span,ref_level,BW])
            if not timeout_flag:
                cenQ,p0,res = self._fit_data(xQ,yQ)
                
            self.calibration.phase=cenQ
            self._check_amp_ratio_limit()
            return [cenI,cenQ],[xI,yI],[xQ,yQ]
        
        except KeyboardInterrupt:
            self._check_amp_ratio_limit()
            return [cenI,cenQ],[xI,yI],[xQ,yQ]



    #----------------------------------------------------------------------------------------------------------------------------Close connectino to the spectrum analyzer---------
           
    
            
    
    #----------------------------------------------------------------------------------------------------------------------------Save calibration for later use--------------------
        
    def save_calibration(self, filename, overwrite = False):
        """
        Save the calibration into a file.
    
        Parameters
        ----------
        filename : string
            The path and filename for where to save the file.
        overwrite : bool, optional
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
        
        self.calibration.save(filename, overwrite)
    #--------------------------------------------------------------------------
    
    def do_calibration(self, fitpoints=7,show_steps = True, save = True,**args):
        """
        This function executes 3 minimizations pairs in a row with different steps.
        It has been written by following user experiences, it could be still 
        possible to get a better calibration by manually do it.
        
        The function will run the following calibrations and steps:
            1) offset with 0.1V step
            2) amp and phase with 0.1 and 5
            3) offset with 0.01V step
            4) amp and phase with 0.01 and 0.1
            5) offset with 0.5 mV step
            6) amp and phase with 0.0005 and 0.01
    
        args:
            - fitpoints: int, def 7 
                see min_points in calibration_offset function
            - show_steps: bool, True
                plots the minimization routine
            - save: bool, True
                it saves a file at the end of the calibration with the following name:
                    ./calibrations/timestr-cal-AWG-[freq]MHz-freq-[cal_freq]GHz-SB-[Sideband]
        """
        
        if self.calibration.AWG_parameters()[0] is None or self.calibration.frequency()[0] is None:
            print('Init not completed\n')
            raise Exception('MISSINGINIT')
        
        self.apply_correction() 
        
        # Start the calibration up with the right parameters.
        
        print('Before the calibration the signal looks like this. \n')
        self.measure_SB(True, True)
        
        
        # Do the calibration
        tmp1=self.calibrate_offset(0.1,fitpoints,**args)
        tmp2=self.calibrate_amp_ratio_and_phase(0.1,5,fitpoints,**args)
        
        tmp1=self.calibrate_offset(0.01,fitpoints,**args)
        tmp2=self.calibrate_amp_ratio_and_phase(0.01,0.1,fitpoints,**args)
        
        tmp1=self.calibrate_offset(0.0005,fitpoints)
        tmp2=self.calibrate_amp_ratio_and_phase(0.0005,0.01,fitpoints,**args)
        
        print('After the calibration the signal looks like this. \n')
        self.calibration.results = self.measure_SB(True, True)
        
        # Save the calibration to file
        if save == True:
            
            fr = self.calibration.frequency()
            awg = self.calibration.AWG_parameters()[0]
            timestr = self.time.strftime('%Y-%m-%d')
            cal_file = 'calibrations/cal-AWG-{}MHz-freq-{}GHz-SB-{}-'.format(awg,fr[0],fr[1] )
            cal_file += timestr + '.cal'
            self.save_calibration(cal_file, True)
        
        # Disconnect from the Spectrum Analyzer after finishing.
        #self.close_SpecAna_connection()
        
    
    #----------------------------------------------------------------------------------------------------------------------------Measure set---------------------------------------
    

    
