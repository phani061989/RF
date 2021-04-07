# -*- coding: utf-8 -*-
__version__ = '2.5.3'

"""
Created on Sat Sep  2 16:28:12 2017

@author: Oscar,Moritz

v2.5.3 - OSC:
    - inserted the option to change the BW of the spectrum analyzer, 
    defaulted to 100 kHz

v2.5.2 - OSC:
    - inserted a load_calibration() method in IQCal_base
    - added some utility sublibrary that contains functions
    - added a function that plots the results stored in all the calibration files
    in a given folder
    - added a function that re-check the calibration of files conteined in a folder,
    for a certain amplitude
    - the function connect_specana() can be used to reconnect the spectrum analyzer
    after closing the connection.
    
v2.5.1 - OSC:
    - .calibration.results is now a property
    - the function do_calibration will automatically add the results to the calibration dictionary
    
v2.5.0 - OSC:
    - modified for the use of multi SignalHounds on the same pc
v2.4.2 - OSC:
    - BUGFIX in measure_SB with RS, averages were fixed to 1
v2.4.1 - OSC:
    - removed redefinition of functions that are in IQCAL_base 
v2.4.0 - OSC:
    - changed init function, now the LO must be initialized and passed to the function (avoids re-init issues)
    - renamed the close_SA_connection function
v2.3.1 - OSC:
    - inserted the amp_corr_ch property in CALPAR
    - the function __check_amp_ratio_limit(...) is now _check_amp_ratio_limit(...)
    
v2.3.0 - OSC:
    - amplitude ratio will always be <1, the corrected channel will be automatically selected
    - ChQ is fixed to be chI+1
    
v2.2.1 - OSC:
    - Introduced titles to progressive plots
    - Introduced a warning if amp_ratio is negative

v2.2.0 - OSC:
    - Old routine has been renamed to IQCAL_OLD and it is not usable anymore (just keeping the code)
    - Created a base class that connects to the SA and separated the AWG class in AWGSIGD and ZI
    - added the calibration using ZI AWG
    
v2.1.0 - OSC:
    - improved dictionary parameters settings
    
v2.0.1 - OSC:
    - bugfixes and rewrote the help
v2.0.0 - OSC:
    - rewrote the structure of the library
    
v1.5.0 - OSC:
    - fixed some optional choice for measuring the three bands
    - fixed a bug about the level reference for Signal Hound measurements
    - compressed the calibration code from 4 functions to only 2
    
v1.4.0 - OSC:
    - corrected the function initialize_calibration
    - corrected the function do_calibration
    - it is possible to chose the amplitude corrected channel in the AWG setup
    
v1.3.0 - OSC:
    - adapted to the Instruments library
    - changed to python dictionary
    
v1.2.2 - OSC:
    - inserted a function to query or set the calibration amplitude

v1.2.1 - OSC:
    - removed checks in the AWG frequency, they are performed by the AWG class
    - inserted debug mode in the calibration functions
    - inserted custom exceptions

v1.2.0 - OSC:
    - the correction is applied by the AWG class
    - renamed apply_corr to apply_correction

v1.1.1 - OSC:
    - improved the calibration routine for speedup and quality

v1.1.0 - MOR:
    - added documentation for all classes and functions
    - improved the calibration procedure for the offset, phase and ratio
    - added new functionality to the measure_SB() function: you can now measure
      whatever sideband you wnat
    - added functionality to measure the signal for a grid of offset values
    - added function to initialize the entire calibration
    - added function to run the entire calibration

v1.0.1 - OSC:
    - added compatibility with SignalHound spectrum analyzer
    - added a lower edge of -95 dBm to stop the automation

v1.0.0 - OSC:
    - LIbrary used to perform the calibration of an IQ mixer
"""