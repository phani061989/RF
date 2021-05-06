# -*- coding: utf-8 -*-
"""
Module for VNA Measurements

Author: Oscar Gargiulo
Date: 08.Apr.2021
"""

import time

import numpy as np

from DataModule.data_table import data_table
from DataModule.data_complex import data_complex



class VNA(object):
    """VNA Instrument class.

    Provides an object to measure with the VNAs in the Kirchmair Lab.

    Parameters
    -----------
    id_string : str
        Name of the VNA. Right now choose between 'VNA1', 'VNA2' or 'VNA3'.
        You can also specify an IP, however then you have to manually specify
        the driver for the chosen IP address
    cryostat : str
        Name of the Cryostat. Choose between 'Freezer' and 'Cooler'
    driver : object, optional
        Driver for the VNA. Only needed if VNA is not already in IPList, or
        you specified it with an IP.
    database : bool
        Connect to database to display current status and check usage
    """

    def __init__(self, ip='192.168.187.123', driver=None, **kwargs):

        # Device Parameters ####################################################
        self.ip = ip
        
        self.kwargs = kwargs

        # Database parameters ##################################################
        self.devtype = 'VNA'
        self.ch = 1  # Not required here, however added for compatibility to CS
        

        # Load driver ##########################################################
        
        if driver is None:
            import E5071C
            self.driver = E5071C.E5071C
        else:
            self.driver = driver

        
        # Init device
        self.instr = None
        self.init_driver()

        # Print information if connection is succesful
        print("Identify: " + self.identify())  # Identify at beginning
        print("IP: " + self.ip)
        
    def identify(self):
        return self.instr.identify()

    def init_driver(self):
        """Init connection to device."""
        self.instr = self.driver(self.ip, **self.kwargs)

    def close_driver(self):
        """Close connection to device"""
        self.instr.close()
        self.instr = None

    # Measurement functions ####################################################
    def meas(self, f_range, npoints=1601, navg=10, power=-50, BW=1e3,
             Spar='S21', Format='MLOG', power_port2=False):
        """Measure specified Spar with specified format.

        Only supported by VNA1 and VNA2 right now.

        Note
        ------
        Highly recommened to use :func:`~VNA.VNA.meas_complex_avg` instead
        since you lose the phase information with this method.

        Parameters
        -----------
        f_range : array
            Frequency range [[f_start, f_stop]]
        npoints : int
            Number of Points
        navg : int
            Number of averages
        power : float
            VNA output power in dBm
        BW : int
            IF Bandwidth in Hz
        Spar : str
            S Parameter to measure. E.g. 'S21'
        Format : str
            Format for measurement. Choose from
             | 'MLOG' : magnitude in dB
             | 'PHAS': phase
             | 'MLIN': linear magnitude
             | 'REAL': real part of the complex data
             | 'IMAG' imaginary part of the complex data
             | 'UPH': Extended phase
             | 'PPH': Positive phase

        Returns
        --------
        DataModule.data_table
            data_table with VNA data
        """
        f, y = self.instr.meas(f_range=f_range,
                               npoints=npoints,
                               navg=navg,
                               power=power,
                               BW=BW,
                               Spar=Spar,
                               Format=Format,
                               power_port2=power_port2)

        units = {'MLOG': 'dB',
                 'PHAS': 'deg',
                 'MLIN': '',
                 'REAL': 'V',
                 'IMAG': 'V',
                 'UPH': 'deg'}
        return data_table([f, y],
                          ['Frequency (GHz)',
                           '{S} ({unit})'.format(S=Spar,
                                                 unit=units[Format])])

    def meas_complex_avg(self, f_range,
                         npoints=1601,
                         navg=10,
                         power=-50,
                         Spar='S21',
                         BW=1e3,
                         autoSave=True,
                         filename='measurements/measurement.dm',
                         Use_Date=True,
                         overwrite=False,
                         #Temp_reading=True,
                         #Temperature_file_path='/home/measurement/DataLogging'

                         parameters={},
                         power_port2=False,
                         scale="lin"):
        """VNA Measurement with complex data format.

        Automatically save as complex datamodule (Voltages) if not specified
        otherwise (autoSave=False).

        Parameters
        -----------
        f_range : [float, float]
            Frequency range in GHz. [f_start, f_stop]
        npoints : int
            Number of Points
        navg : int
            Number of averages
        power : float
            VNA output power in dBm
        Spar : str
            S Parameter to measure. E.g. 'S21'
        BW : int
            IF Bandwidth in Hz
        autoSave : bool
            Automatically save data to disk (RECOMMENDED!!)
        filename : str
            Filepath + Filename. E.g. '/measurements/test'
        overwrite : bool
            Force override
        Use_Date : bool
            Add Date as leading part of filename
        parameters : dict
            Additional parameters to save in dm
        power_port2 : bool
            Power excitation on port2
        scale : "lin", "log"
            Scale for frequencies

        Returns
        --------
        datamodule object
            Returns datamodule object if autoSave is set to False
        """
        # Initialize datamodule
        time_start = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

        

        f, v_re, v_im = self.instr.meas_complex(f_range=f_range,
                                                npoints=npoints, navg=navg,
                                                power=power, Spar=Spar,
                                                BW=BW,
                                                power_port2=power_port2,
                                                scale=scale)

        # Save in datamodule
        data = data_complex(f, np.array(v_re) + 1j * np.array(v_im))
        data.time_start = time_start
        data.time_stop = time.strftime(data.date_format, time.localtime())
        
        data.load_var(f, v_re, v_im)
        data.insert_par(frange=f_range, npoints=npoints, navg=navg,
                        power=power, Spar=Spar, BW=BW,
                        device=self.identify(),
                        **parameters)

        

        # Save
        if autoSave is True:
            data.save(filename, Use_Date, overwrite)
            return data
        else:
            return data

    def meas_complex_segm(self,
                          segments,
                          navg=1,
                          power=-50,
                          Spar='S21',
                          BW=1e3,
                          filename='measurements/measurement.dm',
                          Use_Date=True,
                          overwrite=False,
                         
                          parameters={},
                          autoSave=True,
                          ):
        """VNA measurement with segments in complex data format.

                If optional entries in segment dictionary are not given it will take
                the overall BW and power set the arguments of this function.

                Parameters
                -----------
                segments : list
                    [segment1, segment2, ...] where
                    segment1 = {'start': 1, 'stop': 10, 'npoints': 1601,
                                'BW':1000 (optional), 'power': -20 (optional)}
                npoints : int
                    Number of points
                navg : int
                    Number of averages
                power : int, float
                    RF power in dBm
                BW : int, float
                    IF bandwidth
                Spar : str
                    S Parameter to measure. 'S21' default
                """
        # Initialize datamodule
        time_start = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

        

        # Measure
        f, v_re, v_im = self.instr.meas_complex_segm(segments=segments,
                                                     navg=navg,
                                                     power=power,
                                                     Spar=Spar,
                                                     BW=BW)

        # Save in datamodule
        data = data_complex(f, np.array(v_re) + 1j * np.array(v_im))
        data.time_start = time_start
        data.time_stop = time.strftime(data.date_format, time.localtime())
        
        data.insert_par(segments=segments, navg=navg,
                        power=power, Spar=Spar, BW=BW,
                        device=self.identify(),
                        **parameters)

        

        # Save
        if autoSave is True:
            data.save(filename, Use_Date, overwrite)
            return data
        else:
            return data

    def get_current_measurement_cplx(self):
        """Gets current complex measurement data and saves to cplx. datamodule
        """
        data = data_complex()
        re = np.asarray(self.instr.trace_read(1)[0], dtype='float')
        im = np.asarray(self.instr.trace_read(2)[0], dtype='float')
        try:
            # Try to get frequency range (not supported by E5080)
            x = np.asarray(self.instr.freq_read(), dtype='float')
        except:
            # If not supported, return zeros for setting later manually
            print("Could not receive frequencies. Set to 0. Please"
                  "add manually")
            x = np.zeros_like(re)
        data.x= x / 1e9
        data.value =  re+1j* im
        data.time_stop = time.strftime(data.date_format, time.localtime())
        data.par = self.get_parameters()
        return data

    def get_current_measurement(self):
        """Gets current measurement data and saves to datamodule"""
        dat = data_table()
        y = np.asarray(self.instr.trace_read()[0], dtype='float')
        try:
            # Try to get frequency range (not supported by E5080)
            x = np.asarray(self.instr.freq_read(), dtype='float')
        except:
            # If not supported, return zeros for setting later manually
            print("Could not receive frequencies. Set to 0. Please"
                  "add manually")
            x = np.zeros_like(y)
        dat.x = x / 1e9
        dat.y =  y
        dat.time_stop = time.strftime(dat.date_format, time.localtime())
        dat.par = self.get_parameters()
        return dat

    # Database functions #######################################################
    def get_parameters(self):
        pars = self.instr.read_settings()
        d = {'f_start': '{:.4f} GHz'.format(pars['f_start (GHz)']),
             'f_stop': '{:.4f} GHz'.format(pars['f_stop (GHz)']),
             'Power': '{:.0f} dBm'.format(pars['power (dBm)']),
             'Points': '{:.0f}'.format(pars['Points']),
             'Averages': '{:.0f}'.format(pars['averages']),
             'Format': '{}'.format(pars['Format']),
             'IF Bandwidth': '{:.2f} kHz'.format(pars['IF - BW (Hz)'] / 1e3),
             'RF Power': '{}'.format('ON' if int(pars['output']) == 1 else
                                     'OFF'),
             'S-Parameter': '{}'.format(pars['S_parameter ']),
             }
        return d

    # Additional class for USB VNA #############################################
    def measure_spectrum(self,
                  f_start,
                  f_stop,
                  npoints=1601,
                  navg=10,
                  port='B',
                  BW='auto',
                  filename='measurements/test_SA',
                  Use_Date=True,
                  overwrite=False,
                  
                  parameters={},
                  autoSave=True
                  ):
        """Measure spectrum using the SA functionality of the VNA while sending
        a CW signal on source_port.
        
        Parameters
        ------------
        f_start : float
            Start frequency in Hz
        f_stop : float
            Stop frequency in Hz
        npoints : int
            Number of points
        navg : int
            Number of averages
        port : 'A', 'B'
            Port to acquire spectrum. A is Port1, B is Port 2.
        BW : int,'auto'
            Resolution Bandwidth in Hz. Use 'auto' for automatic detection.
        filename : str
            Filename to save data
        """
        time_start = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

        # Update monitor values
        pars = {'f_start': '{:.6e} Hz'.format(f_start),
                'f_stop': '{:.6e} Hz'.format(f_stop),
                'Averages': '{:.0f}'.format(navg),
                'Bandwidth': BW,
                'SA Port': port,
                'RF Power': 'ON',
                }
        
        # Measure
        f, power = self.instr.measure_spectrum(f_start=f_start,
                                              f_stop=f_stop,
                                              npoints=npoints,
                                              navg=navg,
                                              port=port,
                                              BW=BW
                                              )

        # Save in datamodule
        data = data_table([f, power], ['Frequency (Hz)', 'PSD (dBm)'])
        data.time_start = time_start
        data.time_stop = time.strftime(data.date_format, time.localtime())
        
        data.insert_par(**pars)
        data.insert_par(**parameters)

        

        # Save
        if autoSave is True:
            data.save(filename, Use_Date, overwrite)
            return data
        else:
            return data

    def measure_spectrum_source(self,
                  f_start,
                  f_stop,
                  source_power,
                  source_freq,
                  source_port=1,
                  npoints=1601,
                  navg=10,
                  port='B',
                  BW='auto',
                  filename='measurements/test_SA',
                  Use_Date=True,
                  overwrite=False,
                  parameters={},
                  autoSave=True
                  ):
        """Measure spectrum using the SA functionality of the VNA while sending
        a CW signal on source_port.
        
        Parameters
        ------------
        f_start : float
            Start frequency in Hz
        f_stop : float
            Stop frequency in Hz
        source_power : float
            Power for signal generator
        source_freq : float
            Source frequency in Hz
        source_port : 1,2
            Signal generator on port 1 or 2
        npoints : int
            Number of points
        navg : int
            Number of averages
        port : 'A', 'B'
            Port to acquire spectrum. A is Port1, B is Port 2.
        BW : int,'auto'
            Resolution Bandwidth in Hz. Use 'auto' for automatic detection.
        filename : str
            Filename to save data
        """
        time_start = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

        # Update monitor values
        pars = {'f_start': '{:.6e} Hz'.format(f_start),
                'f_stop': '{:.6e} Hz'.format(f_stop),
                'Source Power': '{:.0f} dBm'.format(source_power),
                'Source Freq': '{:.6e} Hz'.format(source_freq),
                'Source Port': source_port,
                'Averages': '{:.0f}'.format(navg),
                'Bandwidth': BW,
                'SA Port': port,
                'RF Power': 'ON',
                }
        
        # Measure
        f, power = self.instr.measure_spectrum_source(f_start=f_start,
                                                      f_stop=f_stop,
                                                      source_power=source_power,
                                                      source_freq=source_freq,
                                                      source_port=source_port,
                                                      npoints=npoints,
                                                      navg=navg,
                                                      port=port,
                                                      BW=BW
                                                      )

        # Save in datamodule
        data = data_table([f, power], ['Frequency (Hz)', 'PSD (dBm)'])
        data.time_start = time_start
        data.time_stop = time.strftime(data.date_format, time.localtime())
        
        data.insert_par(**pars)
        data.insert_par(**parameters)

        
        # Save
        if autoSave is True:
            data.save(filename, Use_Date, overwrite)
            return data
        else:
            return data
