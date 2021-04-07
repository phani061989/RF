#==============================================================================
# # -*- coding: utf-8 -*-
#==============================================================================
"""
Created on Wed Jan 22 14:42:31 2014

@author: Oscar Gargiulo


This library is a colleciton of utilities functions, it also imports some phisical constants and the most used libraries in python



Last changes:
v1.4.3 - OSC:
    - wrote the function IQ_imbalance that evaluate the rotarion matrix given amplitude correction and phase
    
v1.4.2 - OSC:
    - progressive_plot display labels and titles
    
v1.4.1 - OSC:
    - improved the load_datamodules_in_folder, it is now possible to access keywords in nested dictionaries
    
v1.4.0 - OSC:
    - inserted functions to convert dictionary to list and the opposite
    - inserted functions to save dictionary to file and opposite
    - removed some obsolete functions
    
v1.3.1 - OSC:
    - inserted a function to send emails using the group account
    
v1.3.0 - OSC:
    - erased some obsolete function
    - modified the function IQ2PHAM that converts I,Q points in phase and amplitude
v1.2.4 - OSC:
    - added the functions initialize_shared_gen and check_shared_gen, useful for a shared device with limited user connections like the ANAPICO 4 channels (Fanta4)
    
v1.2.3 - OSC:
    - added a function that can be used to insert the Cryostat temperature information in a data module and the time informations
    
v1.2.2 - OSC:
- modified the function that loads datamodules in a folder, added more options to it

v1.2.1 - OSC:
- added two functions to convert dbm in peak voltage and viceversa

v1.2.0 - OSC:
- renamed some function and added some help to them, removed some function related to some instruments that we don't have anymore (Tektronik AWG)
- added a function that load all the datamodules in a folder and eventually also one or more specified parameters

v1.1.6 - DAZ:
- added fuction to give segments to use for VNA measurement
    segments_vna(...)

v1.1.5 - DAZ:
- added function to read out information of complex dm
    multi_dm_cplx_readout()


v1.1.4 - oscar:
- added the function font_list()

v1.1.3 - DAZ:
- added function to import csv file containing freq., re and im data to complex
    datamodule (to replace fct added in v1.1.2)

v1.1.2 - DAZ:
- added function to combine two dat files (typ. from HFSS) to cplx. datamodule

v1.1.1 - DAZ :
- added function to combine real and imaginry datamodule to complex datamodule
- added function to combine several complex datamodule (typ. over diff. span)



v1.1.0:
the bins_creation function returns the bins 
"""

print('UtilitiesLib v1.4.3')
import numpy as np
import time
import matplotlib.pyplot as plt
import DataModule as dm
import pandas as pd


electron_mass = 9.1093826e-31  #Kg
e_charge = 1.60217653e-19  #Coulomb
hbar = 1.054571726e-34 #Js
speed_of_light = 299792458 #m/s
kb = 1.3806488e-23 #m2 Kg s-2 K-1
n_avogadro = 6.02214129e23 #mol-1

def export_dat_file(Filename,Separator,Header,*args):
    """function export_dat_file(Filename,Separator,Header,*args):

- The function write n column separated by 'Separator'. It is possible to pass
 a header.
The header can be a list of strings, one for each column.

Examples:

export_dat_file('test','\t','example1',x,y,z)

export_dat_file('test','\t',['colx','coly','colz'],x,y,z)

export_dat_file('test','\t',None,x,y,z)

"""

 
    try:    
        f=open(Filename,'w')
    except IOError:
        print("Writing Error, disk protected or full")
        return

    # I find it easier if I create a matrix from the data
    M = args[0]
    
    for i in np.arange(1,len(args)):
        M = np.vstack((M,args[i]))
    M=M.transpose()

    #creating the Columns Header
    tmp=''    
    if (Header != None    ) or Header != '':
        if (type(Header) == list) or (type(Header) == tuple) or (type(Header) == np.ndarray):

            for a in Header:
                tmp=''.join((tmp,str(a),Separator))
            tmp = tmp[:-1]+'\n'
        else:
            tmp = Header+'\n'
    
    f.write(tmp)
    
    
    #Writing the data
    for b in M:
        tmp=''
        for a in b:
            tmp=''.join((tmp,repr(a),Separator))
        tmp = tmp[:-1]+'\n'
        f.write(tmp)
    
    #closing the file
    f.close()
            


def import_dat_file(filename,separator,skip_lines,*args):
    """function M=import_dat_file(filename,separator,skip_lines,*columns)

    The function read a N column files separated by 'separator',return a matrix

    Examples: 
        1) The file has two lines of description, we want only the first and 
        fifth column, the columns are separated by a tabulation:

            M=import_dat_file('fp.dat','\\t',2,1,5)
    
        2) The file doesn't have description, we want all the file:

            M=import_dat_file('z:\\my_files\\file.dat','\\t',0,'a')

"""

    #Check and open 
    try:
        f=open(filename,'r')
    except IOError:
        print("File not found")
        return 0

    
    
    
    #skipping the initial lines as specified
    for i in range(skip_lines):
            a=f.readline()

            
    index=f.tell()
    
    #reading the specified columns
    if args[0]=='a' or args[0]=='A':
        a=f.readline()
        a=a[:-1].split(separator)
        args=np.arange(len(a))
        f.seek(index)
    else:
        args=np.array(args)-1
        
    #creating arrays
    x,z=np.ndarray(len(args)),np.ndarray(len(args))

    while True:
        a=f.readline()
        if a=='' or a=='\n': # When the file ends, a is '', a file can end with a '\n' if it has been saved in this way
            break
        

        a=a[:-1].split(separator)
        
        for i in range(len(args)):
                z[i]=np.double(a[args[i]])
        
        
        x=np.vstack((x,z))
        
    return x[1:]
    
    
    
def export_dm_textfile(data,Filename,Separator='\t',Header=''):
    """function export_dm_textfile(data,Filename,Separator='\t',Header='')
    
    This function will create a text file where the columns contain the data 
    from the data module.
    
    default separator is tabulation,
    Header can be None or empty string, or text or a list/tuple with a string 
    for each column
    
    Examples:    
    export_dm_textfile(data,'test','\t','example1')

    export_dm_textfile(data,'test','\t',['colx','coly','colz'])

    export_dm_textfile(data,'test','\t',None)"""
    
    if type(data) == type(dm.data_2d()):
        export_dat_file(Filename,Separator,Header,data.x,data.y)
    elif type(data) == type(dm.data_3d()):
        arguments=[]
        arguments.append(data.x)
        arguments.append(data.y)
        for a in data.z:
            arguments.append(a)
        export_dat_file(Filename,Separator,Header,*arguments)
    elif type(data) == type(dm.data_cplx()):
        export_dat_file(Filename,Separator,Header,data.x,data.value)
    else:
        print('data module not recognized')

  
    
#-------------------------------------------------------------------------------

def create_frequency_axis(Time_axe,Mode='n'):
    """function f_axe=create_frequency_axe(Time_axe,[Mode='m'])\n
    The function will evaluate the fourier space frequency axe using time_axe 
    as model

    - Time_axe: array of numbers representing a signal time axe
    - Mode: default is mirror, the frequency axe will be [-FS/2,FS/2] 
    where FS is the sampling frequence. The other mode is [0,FS], to use 
    mirroring, use 'm' as Mode.

    Examples:

    - f_axe=create_frequency_axe(time_axe)
    - f_axe=create_frequency_axe(time_axe,1)
"""
    
    sampling_freq=1./(Time_axe[1]-Time_axe[0])
    
    if Mode=='m' or Mode=='M':
        return np.linspace(-sampling_freq/2,sampling_freq/2,Time_axe.size)
    else:
        return np.linspace(0,sampling_freq,Time_axe.size)

#-------------------------------------------------------------------------------

def create_time_axis(Time,SF=25):
    """function t_axe=create_time_axe(Time,[SF=25 (GHz)])
    
    The function will create a time axe from 0 to Time, with a step equal to 
    the sampling period 1/SF


"""
    
    return np.linspace(0,Time,Time*SF)
    

#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------

def bins_creation(Measured,Expected,Binsnumber,Mode=0,Plot=True,Amp=100):
    """function err,var=bins_creation(Measured,Expected,Binsnumber,[Mode=0],
    [Plot=True],[Amplitude=100 (%)])
    
Function used to create a distribution of the Amplitude and Phase measures 
(array), the function will return the mean and variance. Moreover it will plot 
the distribution

- Measured: is the array containing the measures
- Expected: is the array containing the expected values that every measure 
should have or the single number that ALL measures should have
- Binsnumber: is the number of bins, or the resolution of the distribution, 
the x-axis will be splittend in Binsnumber parts
- Plot: at the end of the elaboration it will display the bins (default=True)
- Amp: is the value of the amplitude, it is used only in the plot title
- Mode:

    -0 is for amplitude distribution
    -1 is for phase distribution

    """
    deltaf=Measured-Expected
    
    if Mode==1:
        deltaf[np.abs(deltaf-2*np.pi)<np.abs(deltaf)]-=2*np.pi
        deltaf[np.abs(deltaf+2*np.pi)<np.abs(deltaf)]+=2*np.pi

                    
                
        

    #x-axis creation
    stf=(np.max(deltaf)-np.min(deltaf))/Binsnumber

    binsf=np.zeros(Binsnumber)


    xaxisf=np.arange(Binsnumber)*stf*1.01+np.min(deltaf)
    
    #every bins will be filled

    for i in range(xaxisf.size-1):
        af,bf=deltaf>xaxisf[i],deltaf<xaxisf[i+1]
        af=af&bf
        binsf[i]=deltaf[af].size

    #let's plot the results
    if Plot==True:
        plt.figure()
        plt.stem(xaxisf,binsf)
        if Mode==0:    
            plt.title('$\Delta A - Amplitude$ '+str(int(Amp))+'$\%$')
            plt.xlabel('$\Delta A$')
        else:
            plt.title('$\Delta \phi - Amplitude $'+str(int(Amp))+'$\%$')
            plt.xlabel('$\Delta\phi$')
            
        plt.ylabel('Counts')

    #normalizing and evaluating average and variance

    dx=np.hstack(([0],np.diff(xaxisf)))
    N=np.sum(binsf*dx)    
    err=np.sum(xaxisf*binsf*dx)/N
    sigma= np.sqrt(np.sum(((xaxisf-err)**2)*binsf*dx)/N)

    return (xaxisf,binsf),(err,sigma)



#-------------------------------------------------------------------------------
            
def cut_pulse(Signal,SF,*args):
    """function y1,y2,... = cut_pulse(Signal,SF (Gigasamples/s),*args (ns))
    
This function slice a sequence 'Signal' of pulses separated by a "wait time".

- FS is the sampling frequency of the time axe, 
- Edge should be just after the noise
- args is the time position (in ns) of the wait time (it is necessary that it 
is after one pulse and before the other one) if there is only one pulse it can 
be omitted    
    
- The output is a list of the size of args+1:
    Examples:
        1) with no args, the function cuts the wait time at the begin of 
        the signal:
        
            - y=cut_pulse(Sig1,50,0.03)

        2) Two signals, wait time at the begin and between them, the first 
        pulse is 100ns and the wait time is 5ns:
        
            - y1,y2=cut_pulse(Sig1_2,50,0.03,107)
    """    
    

    args=args[0]
    l=len(args)
    print(l)
    if l>0:
        #args=hstack((0,args,Signal.size))
        #args=round_(args)
        y=[]
    
        for i in range(l):
            
            #slice signal   
            start=np.int( np.round(np.float(args[i][0])*SF,10) )
            stop=np.int(np.round(np.float(args[i][1])*SF,10))
#            print(start)
#            print(args[i][1])
            p=Signal[start:stop]
    
            y.append(np.hstack((np.zeros(start),p)))
            
        return y
        
    else:
        
        
        return Signal
        
#-------------------------------------------------------------------------------
    
def cut_pulse_auto(Signal,Edge,Minimum=0,smith_ist=0,Mode='ZP' ):
    """This function slice a sequence 'Signal' of pulses separated by a 
    "wait time".

    NOTE: there should be a wait time before and after the sequence, too



- Noise should be inside +Edge and -Edge

- The output is a list containing the number of pulses:

- Mode:
    - 'ZP': zero-padding, used to preserve the absolute phase, but not needed 
    for a relative phase
    - 'C': cut, the pulse will be cutted, less memory used and faster
    
- Minimum: minimum number of samples of the cutted signal, helps to avoid noise 
and bumps. Default=0

    
    Examples:
        1) list of pulses     
            - y=cut_pulse(Sig1,0.02)
            first pulse: y[0]
            second pulse: y[1]
            ...
            last pulse y[len(y)-1]

        2) Separated pulse for a sequence of 3 pulses:
        
            - y1,y2,y3=cut_pulse(Sig1_3,0.04)
    """

    y,rest = [],[]
    indexstart,cutted = 0,0
    
    
    """    
    if Edge<=0:
        print("Error: Edge must be positive")
        return
    """
    Mode = Mode.upper()
    
    if (Mode != 'ZP' or Mode != 'C')== False:
        print("Error: wrong Mode inserted")
        return

    state=0 #programmed as a finite state machine
    
    for i in range(len(Signal)):
        
        
        if state==0:
            if Signal[i]>Edge:
                state=1
                indexstart=i
                if Mode=='ZP':
                        rest.append( np.hstack((np.zeros(cutted),Signal[cutted:indexstart])) )
                        
                        
                else:
                        rest.append(Signal[cutted:indexstart])
                
        
        elif state==1:
            if Signal[i]<Edge-smith_ist:
                state=0
                
                if (i-indexstart)> Minimum:
                    
                        
                    if Mode=='ZP':
                        y.append( np.hstack(( np.zeros(indexstart),Signal[indexstart:i] ))  )
                        
                        
                    else:
                        y.append( Signal[indexstart:i] )
                    
                    cutted = i

    if indexstart < cutted:    
        if len(Signal)-cutted > Minimum:
            if Mode=='ZP':
                rest.append( np.hstack((np.zeros(cutted),Signal[cutted:])) )
                        
                        
            else:
                rest.append(Signal[cutted:])
            
    else:
        if len(Signal)-indexstart > Minimum:
            if Mode=='ZP':
                y.append( np.hstack(( np.zeros(indexstart),Signal[indexstart:] ))  )
                        
                        
            else:
                y.append( Signal[indexstart:] )
                
                    
                
    return y,rest

#-------------------------------------------------------------------------------
'''
def delay_measure(Square_pulses,SF=25,Edge=0.2):
    """function delay (ns)=delay_measure(Square_pulses,[SF=25 (Gigasamples/s)][Edge=0.2])
    
This function will measure the delay between the begin of Square_pulses and the end of the glitch

- Square_pulses should start as 1 logic and should have a glitch of 0 logic
- SF Sampling frequency in Gigasamples/s (default=25)
- Edge: Set it between the logic 1 and the logic 0 (default=0.2)


    """
    
    s=Square_pulses>Edge
    flag=0
    for i in range(2,s.size):
        
        if s[i]==True:
            flag=i
            break
    
    return double(flag)/SF
'''
#-------------------------------------------------------------------------------
def wait_time_measure(Signal,SF=25,Edge=0.03):
    """this function can be used to evaluate the waiting time at the begin of a signal, given the Edge (y-axis units)"""
    
    
    indexstart=0
    
    #while indexstart<Signal.size:
            
    for i in range(indexstart,Signal.size):
        if Signal[i]>Edge or Signal[i]<-Edge:
            indexstart=i
            break
        indexstart=i
    
    

    return np.double(indexstart)/SF

#-------------------------------------------------------------------------------
def load_scope_file(Filename,Scope,Time_axis=False):
    '''This function will load a wave from a Matlab file saved with the scope 
    in the default pc_folder shared with it.
The Wfm.dat extension will be added automatically.

- Scope: is the name of the variable containing the Scope initialized Class (Scope library)
- Time_axis: if it is True, the time axe will be generated and returned as first output (default=False)

'''

    
    path=Scope.DEF_PC_FOLDER
    
    
    try:
        f=open(path+'\\'+Filename+'Wfm_hdr.dat','r')
    except IOError:
        print("File not found")
        return 0
    
    Samples=np.int64(f.readline())
    SF=1./np.double(f.readline())
    f.readline()
    f.readline()
    start_time=np.double(f.readline()    )
    
    try:
        f=open(path+'\\'+Filename+'Wfm.dat','r')
    except IOError:
        print("File not found")
        return 0
    
    #y=array([])    
    y=np.ndarray(Samples)
   
    
    for i in range(Samples):
        y[i]=np.double(f.readline())
        
    f.close()
    
    if Time_axis==True:
        x=np.linspace(start_time,Samples/SF,Samples)
        
        return x,y        
    else:
        return y
        
#--------------------------------------------------------------------------------------------------------------------------


    
def cut_wait_time(Signal,Edge=0.01):
    """This function is used to cat slices of a signal separated by a 
    "wait time", given the Edge (y-axis units)"""
        
        #Edge=0.03
        
    indexstart=0
    
    #while indexstart<Signal.size:
            
    for i in range(indexstart,Signal.size):
        if Signal[i]>Edge or Signal[i]<-Edge:
            indexstart=i
            break
        indexstart=i
    
    '''flag=0
    for i in range(indexstart,Signal.size):
        if (Signal[i]<Edge and Signal[i]>-Edge):
            flag+=1

            if flag>20:
                indexstop=i
                break
        else:
            flag=0        '''

    return Signal[indexstart:],indexstart
          
        


#-----------------------------------------------------------------------------

def amplitude_phase(Signal,Freq,Type='s',SF=25,Phaseref=0,Ampref=1,Sigma=0):
    """------------------------------------------------------------------\n    
    function a,p=amplitude_phase(Signal,Freq (GHz),[Type='s'],[SF=25 (Gigasamples/s)],[Phaseref=0 (rad)],[Ampref=1],[Sigma=time/6]):
    
    The function evaluate the amplitude and phase difference of Signal comparing it with a wave created locally with amplitude=Ampref and phase=Phaseref.
    
    - Freq is the frequency of the signal
    - SF is the sampling frequency of the signal (25 Gigasamples/s default)
    - Type:
        - 's': square wave AM (default)
        - 'g': gaussian wave AM
        - 't': triangle wave AM
    
    - Sigma: only in case of a gaussian type, sigma can be inserted (time/6 as default)
    """
    import PULSE
    
    if Ampref<=0:
            print("Error: Ampref cannot be zero or negative")
            return
     
     
    if Phaseref<0:
        Phaseref+=2*np.pi
        
        
        #A_Sig=abs(fft.fft(Sig))#c_sig))
        #print(repr(double(wait_samples)/SF))


    ref= PULSE.Pulse()
    ref.setsampling(SF)
    ref.setlength(Signal.size,"s")
    ref.settone(Freq,Phaseref)
    ref.setamplitude(Ampref)
    ref_90=ref.copy()
    ref_90.settone(Freq,Phaseref+np.pi/2)
    ref=ref.generate()
    ref_90=ref_90.generate()
  
    if Type=='S' or Type=='s':
        c=np.complex(np.sum(ref*Signal[:ref.size]),np.sum(ref_90*Signal[:ref.size]))
        p= np.angle(c)
        
        a=np.abs(c)*2/ref.size
#        if p <0:
 #           p+=2*numpy.pi
    
        
        return a,p
        
    elif Type=='G' or Type=='g':
        c_sig,wait_samples= cut_wait_time(Signal)
        p=np.angle(np.complex(np.sum(ref*Signal[:ref.size]),np.sum(ref_90*Signal[:ref.size])))
        
        ref= PULSE.Pulse()
        ref.setsampling(SF)
        ref.setlength(c_sig.size,"s")
        ref.setsigma(Sigma)
        ref.settone(Freq,0,"g")
        ref.setamplitude(Ampref)
        

        
        """            
    elif Type=='T' or Type=='t':
        c_sig,wait_samples=self.__cut_wait_time(Signal)
        p=angle(complex(sum(ref*Signal[:ref.size]),sum(ref_90*Signal[:ref.size])))
        
        ref=self.triangle_pulse(double(c_sig.size)/SF,Freq,Phaseref,Ampref)    
        """         
    else:
        print("Error: Type inserted is not valid")
        
        return
    
    
    del ref_90
    
    w= np.ndarray(wait_samples)
    w[:]=0

    ref=np.hstack((w,ref))



    
           
    #A_Ref=abs(fft.fft(ref[:c_sig.size]))
    #a=max(A_Sig)/max(A_Ref)
    a= np.sqrt(np.sum(np.double(Signal)**2)/np.sum(np.double(ref)**2))
    #savez('test',y1=Signal,y2=ref)
    
    
    return a,p
        
#---------------------------------------------------------------------------------------------------------------------
def float_to_hexstring(Number):
    """this function converts a float number to an hex number in string format"""
    #32 bit output
    result=''
    
    if Number>=0:
        sign='0'
    else:
        sign='1'
        Number= -Number
    
    i=np.int(Number)    #integer part
    f=Number-i       #decimal part

    #integer to bin conversion            
    while len(result)<64:
        
        result+=str(i%2)
        i=np.int(i/2)
    
        if i==0:
            break
    #decimal to bin conversion
    result=result[::-1]
    #global r
    #r=result
    if result.find('1')==-1:
        exp=0
    else:
        exp=len(result)-result.find('1')+126
    
    while len(result)<64:
        f*=2
        result+=str(np.int(f))
        f-=np.int(f)
    
    #global rf
    #rf=result        
    
    if exp==0:
        exp = result.find('1')
        if exp==-1:
            exp=0
        else:
            exp=-exp+127
        
    #exp evaluation with normalization
    if (result=='0'*len(result)) and exp==0:
        return '0000'
    
    
    exp=bin(exp)[2:]
    exp='0'*(8-len(exp))+exp
    
    
    #global a,b,c,d
    #a=sign
    #b=exp
    result=result[result.find('1')+1:]
    #c=result
    
    string=sign+exp+result
    string=string[:32]
    string=string+'0'*(32-len(string))
    #d=string
    
    
    return hex(int(string,2))[2:]
        
#----------------------------------------------------------------------------------------------------------------------------------
def time_difference(Reference,Multiplier=1):
    """given a time as a reference, this function will evaluate how much time 
    is passed since the reference time.
    
    It is possible to pass a Multiplier (def 1), the final time is 
    multiplied by it"""
    
    tmp=time.gmtime((time.time()-Reference)*Multiplier )
    return '%02ih %02im %02is' % (tmp.tm_hour, tmp.tm_min, tmp.tm_sec)

#----------------------------------------------------------------------------------------------------------------------------------    
def shift90(Sig,Sig_freq,Sampling_freq,Truncated=True):
    '''function Sig_out,Sig_out_90 = shift90(Sig,Sig_freq (MHz),Sampling_freq (GHz),Truncated=True):
    
    This function will shift a signal of 90 degrees, as example a cosin will become a sin.
    
    - Sig is the original signal
    - Sig_freq (MHz) is the signal frequency and Sampling_freq (GHz) is the sampling frequency: they are used to evaluate the shift in samples, if it is not an integer a warning will be displayed
    - Truncated (def:True): is used to truncate the output signals so that their length will be equal (the truncation is a quarter of the period)
                            if it is False, a zero padding will be added at the end of the outputs to preserve the original length
    
    NOTE: to have an integer number the Sig_freq (MHz) should be a multiple of 250*Sampling_freq (GHz)
    
    '''
    
    #global shift
    shift=np.float(Sampling_freq)*250/Sig_freq
    
    if shift-np.int(shift) != 0:
        print('Error: the shift in samples is not an integer number: '+str(Sig_freq))
        raise Exception('SHIFT-NOT-INTEGER')
    
    shift=np.int(shift)
        
    if Truncated==True:
        return Sig[:len(Sig)-shift],Sig[shift:]
    else:
        tmp=np.zeros(shift)
        return np.hstack((Sig[:len(Sig)-shift],tmp)), np.hstack((Sig[shift:],tmp))

#----------------------------------------------------------------------------------------------------------------------------------

    
    
    
#--------------------------------------------------------------------------------------------------------------
def IQ2AMPH(I,Q):
    """This function converts I and Q params in phase and amplitude"""
    
    try:
        if len(I) != len(Q):
            print('ERROR: I and Q arrays must have the same length')
            raise Exception("SIZEERR")
            
        c=np.array(I,dtype=np.complex)+1j*np.array(Q,dtype=np.complex)
        
    
    except TypeError:
        c = np.complex(I,Q)
        
    

    
    

    return np.abs(c),np.angle(c)

#--------------------------------------------------------------------------------------------------------------------------

def S2PRead(input_name,plots_name='',save_plots=True,save_data=True,Simmetric=True,return_all=False):
    """
    This function reads a S2P files saved on the VNA and return the S 
    parameters, plotting and saving them.
    
    arguments:
    
    - input_name is the name of the file
    - plots_name is the name of the plots if the option to save them is True, 
    the name of the plots will be followed by '- Sxx'
    - save_plots can be 1 or True, 0 or False, it is used to save the plots
    - Simmetric: if this value is True than S21 is ignored (it is equal to S12)
    - return_all the function will return all 8 parameters 
    (S parameter + calibration par)
    """
    
    plt.rc('figure',figsize=(16,10))
    
    
    d=open(input_name,'r')
    lines=d.readlines()
    d.close()
    del d
    
    for i,j in enumerate(lines):
        if j[0]=='#':
            first=i+1
            break
    
    points = len(lines)-first
    freq,S11,S11c,S12,S12c,S21,S21c,S22,S22c= np.ndarray(points), np.ndarray(points), np.ndarray(points), np.ndarray(points),np.ndarray(points),np.ndarray(points),np.ndarray(points),np.ndarray(points),np.ndarray(points)
    
    for j,i in enumerate(np.arange(first,len(lines))):
        tmp=lines[i]
        tmp=tmp.split('\t')
        
        freq[j] = np.double(tmp[0])
        S11[j] = np.double(tmp[1])
        S11c[j] = np.double(tmp[2])
        S12[j] = np.double(tmp[3])
        S12c[j] = np.double(tmp[4])
        S21[j] = np.double(tmp[5])
        S21c[j] = np.double(tmp[6])
        S22[j] = np.double(tmp[7])
        S22c[j] = np.double(tmp[8])
    
    border_line=freq.copy()
    border_line[:]=-20
    
    plt.figure(),plt.plot(freq/1e6,S11),plt.plot(freq/1e6,border_line,'r'),plt.xlabel('Frequency (MHz)'),plt.ylabel('S11 (dB)'),
    if save_plots:
        plt.savefig(plots_name+' - S11')
    
    plt.figure(),plt.plot(freq/1e6,S12),plt.xlabel('Frequency (MHz)'),plt.ylabel('S12 (dB)')
    if save_plots:
        plt.savefig(plots_name+' - S12')
    
    plt.figure(),plt.plot(freq/1e6,S22),plt.xlabel('Frequency (MHz)'),plt.ylabel('S22 (dB)'),plt.plot(freq/1e6,border_line,'r')
    if save_plots:
        plt.savefig(plots_name+' - S22')

    if Simmetric != True:        
        plt.figure(4),plt.plot(freq/1e6,S21),plt.xlabel('Frequency (MHz)'),plt.ylabel('S21 (dB)')
        if save_plots==True:
            plt.savefig(plots_name+' - S21')
    
    if save_data==True:
        np.savez(plots_name+' - S11',x=freq,y=S11)
        np.savez(plots_name+' - S12',x=freq,y=S12)
        if Simmetric!=True:
            np.savez(plots_name+' - S21',x=freq,y=S21)
        np.savez(plots_name+' - S22',x=freq,y=S22)
    
    if return_all == True:
        return freq,S11,S11c,S12,S12c,S21c,S21c,S22,S22c
    else:
        return freq,S11,S12,S21,S22
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def filter_data(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only 
        smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def write_waves_csv(filename,wavename,data,type='ANALOG_16'):
    ''' funtion write_waves_csv(filename,wavename,data,[type='ANALOG_16'])
    
    This function creates a csv file that can be loaded in the AWG or DIO 
    memory 
    
    types are:
    - ANALOG_16
    - ANALOG_32    
    - ANALOG_16_DUAL
    - ANALOG_32_DUAL
    - IQ
    - IQPOLAR
    - DIGITAL
    '''
    
    types= ['ANALOG_16', 'ANALOG_32', 'ANALOG_16_DUAL', 'ANALOG_32_DUAL', 'IQ', 'IQPOLAR', 'DIGITAL']
    
    index=types.index(type.upper())
    
    with open(filename+'.csv','w') as f:
        
        f.writelines('waveformName,'+wavename+',\n')
        f.writelines('waveformPoints,'+str(int(len(data)))+',\n' )
        f.writelines('waveformType,WAVE_'+types[index]+',\n')
        if index==6:
            for d in data:
                f.writelines(str(int(d))+',\n')
        elif index==0 or index==2:
            for d in data:
                f.writelines(str(np.float16(d))+',\n')
        else:
            for d in data:
                f.writelines(str(np.float32(d))+',\n')
        

def progressive_plot_2d(x,y,style='b',clear=True,plt_title=None,labels=['','']):
    """This function can be used to make a progressive plot, everytime the 
    x and y axis are updated, the function must be called again"""
    
    from IPython import display
    
    if clear:
        plt.clf()
        
    if x is None:
        plt.plot(y,style)
    else:
        plt.plot(x,y,style)
        
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if plt_title is not None:
        plt.title(plt_title)
    display.clear_output(wait=True)
    display.display(plt.gcf())

def progressive_plot_3d(x,y,z,Levels=10,plt_title=None,labels=['','']):
    """This function can be used to make a progressive plot, everytime the x 
    and y axis are updated, the function must be called again
    
    This function uses contourf
    """
    from IPython import display
    
    plt.contourf(x,y,z,Levels)
    
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if plt_title is not None:
        plt.title(plt_title)
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
            
class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)
        
def combine_re_im_cplx(re, im):         
    """Combines 2d data modolue one containing real other imaginary data"""
    if (re.x == im.x).all():            #Check if they belong to same span
        
        datacplx = dm.data_cplx()
        datacplx.load_var(re.x, re.y, im.y)
        datacplx.par = re.par
        
        return datacplx
    else:
        print("X values of datamodules have to be the same.")
        
        
def combine_re_im_dat_file_cplx(loc_re, loc_im):
    """As .csv filess should be exported this function should not be used any
    more. It remains to ensure downwards compatibility.
    This function combines two .dat files each containg frequency in the
    first column and real respectivly complex part of the S parameter in the
    second column.
    Expected arguments are the location of those dat files
    Returns complex datamodule"""    
    
    dm_cplx = dm.data_cplx()
    data_re = dm.data_2d()
    data_im = dm.data_2d()
    
    dat_re = import_dat_file(loc_re,' ',0,'a')
    dat_im = import_dat_file(loc_im,' ',0,'a')
    data_re.load_var(dat_re[:,0],dat_re[:,1])
    data_im.load_var(dat_im[:,0],dat_im[:,1])

    dm_cplx = combine_re_im_cplx(data_re, data_im)
    
    return dm_cplx
    
def import_csv_to_cplx_dm(file_name, print_header = False):
    """Creates complex data module from .csv file. First column MUST BE freq.
    second corresponding real and third imaginary data (as expected from HFSS).
    Furthermore it checks the corresponding table headers (and gives warnings), 
    and stores the headers of the table in comments of datamodule."""

    dm_cplx =  dm.data_cplx()
    csv_data = pd.read_csv(file_name)
    
    if (len(csv_data.columns) is not 3):
        print('Give csv file with expected three columns.')
        return
        
    header = list(csv_data.columns.values)
    
    if (header[0][:4] != 'Freq'):   #Checks if columns are in correct order.
        print('Warning! - 1st column seems to not contain frequency data.')
        
    if (header[1][:2] != 're'):
        print('Warning! - 2nd column seems to not contain real part of data.')
        
    if (header[2][:2] != 'im'):
        print('Warning! - 3rd column seems to not contain real part of data.')
        
    matrix_data = csv_data.as_matrix()
    
    freq_data = matrix_data[:,0]    #To store data in datamodule
    re_data = matrix_data[:,1]
    im_data = matrix_data[:,2]
    
    dm_cplx.load_var(freq_data, re_data, im_data)
        #Stores header in the comments
    dm_cplx.comments = header[0] +'||'+ header[1] +'||'+ header[2]
    
    if print_header is True:
        print('File info:' + dm_cplx.comments)
    
    return dm_cplx
    
def multi_dm_cplx_readout(data):
    """Functions gets array of complex data modules and returns array of:
    Qi, Qi_err, Qc, Qc_err, Ql, Ql_err, fr, fr_err"""
    
    Qi = []
    Qi_err = []
    Qc = []
    Qc_err = []
    Ql = []
    Ql_err = []
    fr = []
    fr_err = []
    
    for i in np.arange(len(data)):
        Qi.append(data[i].fitresults.loc['Qint','Values'])
        Qc.append(data[i].fitresults.loc['Qc','Values'])
        Ql.append(data[i].fitresults.loc['Ql','Values'])
        fr.append(data[i].fitresults.loc['fr','Values'])
        
        Qi_err.append(data[i].fitresults.loc['Qint','Errors'])
        Qc_err.append(data[i].fitresults.loc['Qc','Errors'])
        Ql_err.append(data[i].fitresults.loc['Ql','Errors'])
        fr_err.append(data[i].fitresults.loc['fr','Errors'])
        
    return Qi, Qi_err, Qc, Qc_err, Ql, Ql_err, fr, fr_err


def multi_dm_cplx_readout_new(data):
    """Functions gets array of complex data modules and returns array of:
    Qi, Qi_err, Qc, Qc_err, Ql, Ql_err, fr, fr_err"""
    
    Qi = []
    Qi_err = []
    Qc = []
    Qc_err = []
    Ql = []
    Ql_err = []
    fr = []
    fr_err = []
    
    for i in np.arange(len(data)):
        Qi.append(data[i].fitresults.Value['Qint'])
        Qc.append(data[i].fitresults.Value['Qc'])
        Ql.append(data[i].fitresults.Value['QL'])
        fr.append(data[i].fitresults.Value['fr (GHz)'])
        
        Qi_err.append(data[i].fitresults.Error['Qint'])
        Qc_err.append(data[i].fitresults.Error['Qc'])
        Ql_err.append(data[i].fitresults.Error['QL'])
        fr_err.append(data[i].fitresults.Error['fr (GHz)'])
        
    return Qi, Qi_err, Qc, Qc_err, Ql, Ql_err, fr, fr_err

    
def segments_vna(centerfrq, span_tot, span_detail , npoints_wings, npoints_detail):
    """Returns array of three dictionaries to hand over to VNA.
        parameters: centerfrq (GHz), span_tot (GHz) --> begin meas centerfrq-span/2
        span_detailed (GHz) span which is measured in detail (with npoints_detail)
        npoints_wings: number of points used on each of the wings"""   
    
    seg1 = {'start': centerfrq - span_tot/2,'stop':centerfrq - span_detail/2 ,'npoints':npoints_wings}
    seg2 = {'start': centerfrq - span_detail/2,'stop':centerfrq + span_detail/2 ,'npoints':npoints_detail}
    seg3 = {'start': centerfrq + span_detail/2,'stop':centerfrq + span_tot/2 ,'npoints':npoints_wings}
    
    return [seg1,seg2,seg3]
    
    
    
    
    
        


def font_list():
    """ returns the font list names that can be used in maplotlib.pyplot"""
    import matplotlib.font_manager as fm
    return [f.name for f in fm.fontManager.ttflist]

def load_datamodules_in_folder(path='.',word_search=None,parameters=[],return_filelist=False, sort = True):
    """This function opens all the datamodules in a folder and returns them in 
    a list
    
    if word_search is a string, only the files that contains it will be loaded.
    
    It is possible to pass a list that contains parameters name, new arrays 
    containing the specified parameters will be returned. The first parameter 
    is the most important, if it will not be found in the datamodule, a warning
    will occur and the file will not be loaded.
    
    if return_filelist is True, the complete loaded filelist will be added to 
    the output
    
    the parameters and file will be sorted in function of the first parameter 
    as default. Set sort to False to avoid it.
    NOTE: for nested dictionaries, use a . to separate the keywords, example:
        parameters['Pulse.Frequency']
    
    Examples:
        
    dmlist = load_datamodules_in_folder('test')
    
    dmlist, powers = load_datamodules_in_folder('test',['excitation_power'])
    powers = powers[0]
    
    dmlist, powers = load_datamodules_in_folder('test',
    ['excitation_power','ex_frequency'])
    freqs = powers[1]
    powers = powers[0]
    
    """
    
    import glob
    import os
    
    if type(parameters) != list and type(parameters)!= tuple:
        print('parameters must be a list or a tuple of strings')
        raise Exception('PARTYPE')
    
        
    path = os.path.split(path)[0]
    if path is '':
        path = '.'
    
    #return path
    filelist= glob.glob(path+'/*.dm')
    #return filelist
    
    dmlist = []
    fl = []
    for f in filelist:
        
        
        if word_search is not None:
            if f.find(word_search) > -1:
                dmlist.append(dm.load_datamodule(f)) #If the word has been found, the file is kept, otherwise it is discarded
                fl.append(f) # the filename is kept also
        else: #every file is taken
            dmlist.append(dm.load_datamodule(f))
            fl.append(f)
    
    if len(parameters) == 0: # No parameters is required, so only the data modules are passed
        if return_filelist is True:
            return dmlist,fl # the filelist is added to the output IF required
        else:
            return dmlist
    else:
        parslist=[]
        for i in range(len(parameters)): # in the general case the parameters list is a MxN matrix, where M is the number of parameters specified by the user and N is the number of opened data modules
            parslist.append([])
    
    i,l = 0,len(dmlist)   
    
    while(i<l):
        for j in range(len(parameters)):
            try:
                #it can happen that the datamodule par is a nested dictionary, the following code should deal with it
                partmp = parameters[j]
                partmp=partmp.split('.')
                tmp = dmlist[i].par
                for k in partmp:
                    tmp=tmp[k]
                
                parslist[j].append(tmp) #If the specified parameter has been found, it will be added
                
            except:
                print('The file doesn\'t contain the specified parameter: '+str(fl[i])) 
                dmlist.pop(i) #otherwise the datamodule that doesn't contain it will be discarded
                fl.pop(i)
                for tmp in range(j):
                    parslist[tmp].pop(i)
                l-=1
                i-=1
                break
        
        i+=1
                
    
    
        
    
    dmlist = np.array(dmlist) #arrays can be easily sorted
    parslist = np.array(parslist)
    
    if sort is True:
        arguments = np.argsort(parslist[0])
        dmlist = dmlist[arguments] #now dmlist is ordered in function of the specified parameter
        
        for j in range(len(parslist)):
            parslist[j] = parslist[j][arguments] #now every parameter list is ordered also
        
        
    if return_filelist is True:
        return dmlist,parslist,fl
    else:
        return dmlist,parslist


def dbmtovp(dbm,precision=3):
    '''Assuming a 50 Ohm impedence, this function will return the peak voltage 
    (in V) corresponding to the assigned power (dBm).
    It is possible to assign the rounding precision (def=3)'''
    return np.round(np.sqrt( (10**(dbm/10) / 10)),precision)
    
def vptodbm(vp,precision=1):
    '''Assuming a 50 Ohm impedence, this function will return the power in dBm 
    corresponding to the assigned peak voltage (in V).
    It is possible to assign the rounding precision (def=1)
    '''
    P = vp**2/100
    
    return np.round(10.0*np.log10(P*1e3),precision)
    

def temperature_measurement(data,order,cryoID,Sensor='T_MC_RuOx',add_time = True):
    """
    This function adds temperature infos to the data module, together with the 
    time of the last update of the temperature.
    
    Moreover the information of the current time is added at the data module as well.
    
    args:
        - data: is the data module 
        - order: 
            'b', 0 or 'before' to insert the temperature before the measurement, in temp_start and temp_start_time
            'a', 1 or 'after' to insert the temperature after the measurement, in temp_stop and temp_stop_time
        - CryoID is the cryostat in use (case insensitive)
        - add_time: True (def) or anything else to disable, it is used to add the time information to the data module
        
    NOTE: temp_start_time and temp_stop_time containts the time of the last fridge update, not the measurement time.
    
    The time_start and time_stop parameters, will contain the measurement timing informations 
    """
    
    from SensorReader import SensorReader
    
    if type(order) is str:
        order = order.lower()
        
    sr = SensorReader(cryoID,Sensor)
    sr.update()
    
    if order == 0 or order == 'b' or order == 'before':
        data.temp_start = sr.base_temp()
        data.temp_start_time = sr.last_update()
        
        if add_time is True:
            data.time_start = time.time()
            
    elif order == 0 or order == 'b' or order == 'before':
        data.temp_stop = sr.base_temp()
        data.temp_stop_time = sr.last_update()
        
        if add_time is True:
            data.time_stop = time.time()
    else:
        print('Wrong order iserted')
        
    




    
    
    

    
def guess_cos_pars(x,y):
    '''
    Given a cosine data (x,y), this function will try to guess the initial fit 
    parameters: ['amplitude', 'period', 0., 'offset'].
    
    NOTE: phase has not been implemented and it shouldn't be necessary for the
    fit.
    
    '''
    
    offset = np.sum(y)/len(y) 
    
    yf = np.fft.fft(y-offset)/len(y)
    yf = yf[:int(len(yf)/2)]
    pg = [0.,0.,0.,0.]
    
    index = np.abs(yf).argmax()
    
    pg[0] = np.abs(yf)[index]*2
    pg[1] = x[-1]/np.float(index)
    #pg[2] = np.angle(yf)[index]
    pg[3] = offset
    return pg

def check_minmax_x_y(datas):
    '''given a set of dm_surfaces, it will check the min,max and minimum steps 
    along all the x-axis and y-axis.
    
    args:
    - datas must be an iterable of dm_surfaces
    
    the function will return an iterable: 
        min_x,max_x,minstep_x,min_y,maxy_,minstep_y
    '''
    
    minsx,maxsx= [],[]
    minsy,maxsy= [],[]
    minsxs,minsys = [],[]
    #val = []
    for d in datas:
        minsx.append(d.x.min())
        maxsx.append(d.x.max())
        minsxs.append(np.diff(d.x).min())
        minsy.append(d.y.min())
        maxsy.append(d.y.max())
        minsys.append(np.diff(d.y).min())
        
    
    return np.round_((np.min(minsx),np.max(maxsx),np.min(minsxs),np.min(minsy),np.max(maxsy),np.min(minsys)),10)

def combine_maps(datas,v):
    '''
    Given several dm_surface elements, this function will combine them in a 
    unique map, the empty area will be filled with the value v.
    
    args:
    - datas must be an iterable of dm_surfaces
    - v must be a float number that will fill the rest of the map
    
    returns a dm_surface
    
    
    '''

    mm = check_minmax_x_y(datas)
    
    x2 = np.round_(np.arange(mm[0],mm[1]+mm[2],mm[2]),9)
    
    
    if x2[-1]<mm[1]:
        x2 = np.hstack((x2,mm[1]))
        
    y2 = np.round_(np.arange(mm[3],mm[4]+mm[5],mm[5]),9)
    
    if y2[-1]<mm[4]:
        y2 = np.hstack((y2,mm[4]))
        
    z2 = np.ones((len(y2),len(x2)))*v
    
    try:
        for d in datas:
        
            for i,valy in enumerate(d.y):
                
                ind_y = np.where(y2>=valy)[0][0]
                for j,valx in enumerate(d.x):
                    
                    ind_x = np.where(x2>=valx)[0][0]
                    z2[ind_y,ind_x] = d.z[j,i]

    except IndexError:
        print('IndexError occurred: returning mm,current_map,current_valuex,curr_valuey\n')
        return mm,d,valx,valy
    
    z2 = z2.transpose()
    """ to implement better    
    def check_and_erase_y(z,y,value):
        indexes=[]
        for i,d in enumerate(z):
            if np.all(d==value):
                indexes.append(i)
        
        tmp1=z.tolist()
        tmp2=y.tolist()
        for i in indexes[::-1]:
            tmp1.pop(i)
            tmp2.pop(i)
        z = np.array(tmp1)
        y = np.array(tmp2)
        
    def check_and_erase_x(z,x,value):
        tmp1 = z.transpose()
        tmp2 = x.tolist()
        indexes=[]
        for i,d in enumerate(tmp1):
            if np.all(d==value):
                indexes.append(i)
                
        tmp1 = tmp1.tolist()
        for i in indexes[::-1]:
            tmp1.pop(i)
            tmp2.pop(i)
        
        z = np.array(tmp1).transpose()
        x = np.array(tmp2)
    
    
    
    check_and_erase_x(z2,x2,v)
    check_and_erase_y(z2,y2,v)
    """
    return dm.data_grid((x2,y2,z2))



def sendEmail(fromtxt, tolist, subj, msgtxt):
    """
        Function to send email.

        Parameters:
        fromtxt:    the string cointaining who is the Sender
        tolist:     a list of email addresses, it has to be a list even for
                    one address
        subj:       Subject string
        msgtext:    Message to send
    """
    import smtplib
    
    message = 'Subject: %s\n\n%s' % (subj, msgtxt)
    server = smtplib.SMTP('smtp.uibk.ac.at', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login("x2241036", "Ugu9Pap3")
    return server.sendmail(fromtxt, tolist, message)


def dicttolist(dictionary):
    """This function return the keys and values of a dictionary as lists"""
    
    if type(dictionary)!=dict:
        print('dictionary must be a python dictionary\n')
        raise Exception(TypeError)
        
    k = [x for x in dictionary.keys()]
    v = [x for x in dictionary.values()]
    return k,v

def listtodict(keys_list, values_list):
    """Given a list of names and a list of values, this function will merge them
    in a dictionary"""
    
    if len(keys_list)!= len(values_list):
        print('Error: keys_list and values_list must have the same length!\n')
        raise Exception(ValueError)
        
    return {keys_list[i]:values_list[i] for i in range(len(keys_list))}

def dictionarytofile(dictionary,filename):
    """This function can be used to write a dictionary to a text file"""
    if type(dictionary)!=dict:
        print('Error: dictionary must be a python dictionary\n')
        raise Exception(TypeError)
        
    with open(filename,'w') as d:
        d.writelines(repr(dictionary))

def filetodictionary(filename):
    """This function can load a text file as a dictionary, the text file should 
    contain a string that python can interpret as a dictionary. 
    
    Example:
     {'a': 5, 'b': 6, 'd': 7, 'e': {'e1': 5, 'e2': 6},'f':'test'}
        """
    """
    with open(filename,'r') as d:
        text = d.readlines()
    
    
    
    exec('a='+text[0])
    possibles = globals().copy()
    possibles.update(locals())
    return possibles.get('a')
    """
    with open(filename,'r') as d:
        tmp = eval(d.read())
    
    return tmp


def IQ_imbalance(g, phi):
    phi = np.radians(phi)
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]