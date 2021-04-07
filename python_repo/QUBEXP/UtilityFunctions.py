from SensorReader import SensorReader
import keysightSD1 as sd1
#from UtilitiesLib import *
import numpy as np
import DataModule as dm
import time

def load_HVI(path,dig,awg):
    """Function used to open the HVI and load it in the boards"""
    HVI = sd1.SD_HVI()
    HVI.open(path)
    
    tmp = HVI.assignHardwareWithIndexAndSlot(0,dig._chassis,dig._slot)#DIG has index 0
    if tmp <0:
        print('Error in HW assignment (DIG): {}\n'.format(tmp))
        raise Exception('HVIERR')
    tmp= HVI.assignHardwareWithIndexAndSlot(1,awg._chassis,awg._slot)#AWG has index 1
    if tmp<0:
        print('Error in HW assignment (AWG): {}\n'.format(tmp))
        raise Exception('HVIERR')
    tmp = HVI.compile()
    if tmp<0:
        print('Error in HVI compiling: {}\n'.format(tmp))
        raise Exception('HVIERR')
    tmp = HVI.load()
    if tmp<0:
        print('Error in loading the HVI in the boards: {}\n'.format(tmp))
        raise Exception('HVIERR')
    return HVI
        
        


def read_temperature(CryoID,file_path='c:/temperature',BaseSensor='T_MC_RuOx'):
    
        
    sr = SensorReader(CryoID,BaseSensor,file_path)
    sr.update()
    
    
        
    return sr.base_temp(),sr.last_update()
    

def digits_to_Vp(digits,scale=None):
        
        return digits/(2**15-1)*scale
    
def set_phase_in_reg_deg(angle,phase_corr=0):
        '''conversion from deg to register value'''
        tot = (angle+phase_corr) % 360
    
        return np.int( (2**32)/360*tot )

def set_phase_in_reg_rad(angle,phase_corr=0):
        '''conversion from rad to register value'''
        angle = np.rad2deg(angle)
        return set_phase_in_reg_deg(angle,phase_corr)
    
def create_flux_pulse(length,slope=20,SF=1.):
        """This function creates a flux pulse with a smooth raising time
        NOTE: the slopes are added to the length
        """
        n_points_slope = np.int(slope*SF)
        n_points_flat = np.int(length*SF)-n_points_slope*2
        
        t_axis = np.linspace(0,1,n_points_slope)
        signal1 = np.abs(np.sin(t_axis*np.pi/2))
        signal2 = np.ones(n_points_flat)
        signal3 = 1-np.abs(np.sin(t_axis*np.pi/2))
    
        return np.hstack((signal1,signal2,signal3))

def create_double_flux_pulse(length,slope=20,amp1=1,amp2=None,SF=1):
        """This function creates a double flux pulse with a smooth raising time,
        
        - amp1 is the amplitude of the pulse (def 1, normalized to +-1)
        - amp2 is the amplitude of the second half of the pulse (def is None that means -amp1)
        
        NOTE: the slopes are added to the length
        """
        #n_points_slope = np.int(slope*SF)
        #n_points_flat = np.int(length/2*SF)
        
        if amp1 <-1 or amp1 >1:
            print('Error: amp1 must be [-1,1]')
            raise Exception('FLAMP')
        
        if amp2 is None:
            amp2=-amp1
        
        if amp2 <-1 or amp2 >1:
            print('Error: amp2 must be [-1,1]')
            raise Exception('FLAMP')
        
        
        n_points_slope = np.int(slope*SF)
        n_points_flat = np.int(length*SF/2)-n_points_slope*2
        
        t_axis = np.linspace(0,1,n_points_slope)
        signal1 = np.abs(np.sin(t_axis*np.pi/2))*amp1
        signal2 = np.ones(n_points_flat)*amp1
        signal3 = amp1-np.abs(np.sin(t_axis*np.pi/2))*(amp1-amp2)
        signal4 = np.ones(n_points_flat)*amp2
        signal5 = amp2+np.abs(np.sin(t_axis*np.pi/2))*-amp2
    
        return np.hstack((signal1,signal2,signal3,signal4,signal5))


    
    
def create_compensated_flux_pulse(length,overshoot,tau,slope1=20,SF=1.):
        """This function creates a flux pulse with a smooth raising time
        NOTE: the slopes are added to the length
        """
        
        if slope1<0:
            print('slope1 cannot be negative\n')
            raise ValueError
        
        
        if tau<0:
            print('tau cannot be negative\n')
            raise ValueError
        
        if length<0:
            print('length cannot be negative\n')
            raise ValueError
            
        n_points_slope1 = np.int(slope1*SF)
        
        n_points_signal = np.int(length*SF-n_points_slope1*2)
        if n_points_signal <0:
            print('length cannot be smaller than slope1+slope2\n')
            raise ValueError
        
        if slope1>0:
            t_axis = np.linspace(0,1,n_points_slope1)
            signal1 = np.abs(np.sin(t_axis*np.pi/2)) #raising
        
        
            t_axis = np.linspace(0,length-slope1*2,n_points_signal-n_points_slope1)
            signal2 = (1-1/overshoot)*np.exp(-t_axis/tau)+1/overshoot
        
        
            t_axis = np.linspace(0,1,n_points_slope1)
            signal3 = signal2[-1]-signal2[-1]*overshoot*np.abs(np.sin(t_axis*np.pi/2))
            t_axis = np.linspace(0,length-slope1*2,n_points_signal-n_points_slope1)
            signal4 = signal3[-1]+(1-np.exp(-t_axis/tau))*abs(signal3[-1])
            
            y= np.hstack((signal1,signal2,signal3,signal4))
        elif slope1==0:
            t_axis = np.linspace(0,length,n_points_signal)
            signal1 = (1-1/overshoot)*np.exp(-t_axis/tau)+1/overshoot
            
            start_point = signal1[-1]-signal1[-1]*overshoot
            signal2 = start_point+(1-np.exp(-t_axis/tau))*abs(start_point)
            y= np.hstack((signal1,signal2))
        else:
            print("slope1 can't be negative\n")
        
        
        return y

def create_compensated_flux_pulse_double_tau(length,overshoot,tau1,tau2,overshoot2=None,tau12=None,tau22=None,SF=1.):
        """This function creates a flux pulse with a smooth raising time
        NOTE: the slopes are added to the length
        """
        
        
        if tau1<0 or tau2<0:
            print('tau cannot be negative\n')
            raise ValueError
        
        if length<0:
            print('length cannot be negative\n')
            raise ValueError
        
        if overshoot2 is None:
            overshoot2=overshoot
        
        if tau12 is None:
            tau12 = tau1
        
        if tau22 is None:
            tau22 = tau2

        
        n_points_signal = np.int(length*SF)
        if n_points_signal <0:
            print('length cannot be smaller than slope1+slope2\n')
            raise ValueError
        
        
        t_axis = np.linspace(0,length,n_points_signal)
        tmp = (np.exp(-t_axis/tau1)+np.exp(-t_axis/tau2))/2
        signal1 = (1-1/overshoot)*tmp+1/overshoot
        
        start_point = signal1[-1]-signal1[-1]*overshoot2
        tmp = (np.exp(-t_axis/tau12)+np.exp(-t_axis/tau22))/2
        signal2 = start_point+(1-tmp)*abs(start_point)
        y= np.hstack((signal1,signal2))
        
        
        return y
    
def create_compensated_flux_pulse_double_os(length,overshoot1,overshoot2,tau,slope1=20,SF=1.):
        """This function creates a flux pulse with a smooth raising time
        NOTE: the slopes are added to the length
        """
        
        if slope1<0:
            print('slope1 cannot be negative\n')
            raise ValueError
        
        
        if tau<0:
            print('tau cannot be negative\n')
            raise ValueError
        
        if length<0:
            print('length cannot be negative\n')
            raise ValueError
            
        n_points_slope1 = np.int(slope1*SF)
        
        n_points_signal = np.int(length*SF-n_points_slope1*2)
        if n_points_signal <0:
            print('length cannot be smaller than slope1+slope2\n')
            raise ValueError
        
        if slope1>0:
            t_axis = np.linspace(0,1,n_points_slope1)
            signal1 = np.abs(np.sin(t_axis*np.pi/2)) #raising
        
        
            t_axis = np.linspace(0,length-slope1*2,n_points_signal-n_points_slope1)
            signal2 = (1-1/overshoot1)*np.exp(-t_axis/tau)+1/overshoot1
        
        
            t_axis = np.linspace(0,1,n_points_slope1)
            signal3 = signal2[-1]-signal2[-1]*overshoot2*np.abs(np.sin(t_axis*np.pi/2))
            t_axis = np.linspace(0,length-slope1*2,n_points_signal-n_points_slope1)
            signal4 = signal3[-1]+(1-np.exp(-t_axis/tau))*abs(signal3[-1])
            
            y= np.hstack((signal1,signal2,signal3,signal4))
        elif slope1==0:
            t_axis = np.linspace(0,length,n_points_signal)
            signal1 = (1-1/overshoot1)*np.exp(-t_axis/tau)+1/overshoot1
            
            start_point = signal1[-1]-signal1[-1]*overshoot2
            signal2 = start_point+(1-np.exp(-t_axis/tau))*abs(start_point)
            y= np.hstack((signal1,signal2))
        else:
            print("slope1 can't be negative\n")
        
        
        return y