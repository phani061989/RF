# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:31:53 2014

@author: Oscar Gargiulo

Utilities used to create pulses
v3.1.1: 
    - bugfix in sequence object
    
v3.1.0:
    - fixed the shape of the gaussian
    
v3.0.1:
    - inserted offset
    
v3.0.0:
    - inserted a python dictionary
    
v2.0.2:
    - rounded output to 12 digits

v2.0.1:
    - Solved bug in modulation, a delay or wait time >0 was giving problem to the function

v2.0.0:
    - changed the library for new QUBEXP code, more object oriented

v1.1.0:
    -changed some parameters for compatibility with ANAPICO 4 channels

v1.0.1:
    - added a variable called par, that contains additional parameters used for the creation of pulses with the instruments
v1.0.2:
    - added a function that returns the parameters so that they can be saved, more info in the function help (return_par_infos)
"""


#print('Pulse library v2.0.3')


def load_pulse_from_file(filename):
    import pickle
    
    with open(filename,'rb') as f:
        a = pickle.load(f)
        
    
    
    return a

class Pulse(object):
    
    import numpy as np
    import numexpr as ne
    import matplotlib.pyplot as pp
    
    version= '3.1.1'
    
    def __init__(self,Frequency=0.0,Phase=0.0,Shape='s',Width=0.0,Delay=0.0,Wait=0.0,Amplitude=1.0,Offset=0.,Sigma=0.0,SF=1.0,ID=''):
            """Class Init parameters:

            chosen_name=Pulse(    
    
            - Frequency (Ghz)        def 0
            - Phase (deg)       def 0
            - Shape ('s';'g')   def 's'
            - Length (ns)       def 0
            - Delay (ns)        def 0
            - Width (ns)        def 0
            - Amplitude         def 1
            - Offset            def 0
            - Sigma (ns)        def 0 
            - SF (Gigasample/s) def 1
            - ID (text)         def ''
            )
            
            the arguments have a default value (showed), so it is possible to 
            create the class without arguments.
            
            Examples:
            
            - p=Pulse()
            - p=Pulse(Length=20)
            - p=Pulse(8,30,Length=20)
            """
            self.pars_dict={}
            
            self.offset(Offset)
            self.delay(Delay)
            self.width(Width)
            self.wait(Wait)
            self.sampling(SF)
            self.frequency(Frequency)
            self.phase(Phase)
            self.shape(Shape)
            self.sigma(Sigma)
            self.amplitude(Amplitude)
            self.id(ID)
            
            
            
        

         
    
#------------------------------------------------------------------------------ check functions --------------------
    def __check_Type(self,Type):
        
        Type_list = ['t','s']
        
        try:
            Type_list.index(Type)
        except ValueError:
            print('Wrong Type inserted: t or s\n')
            raise Exception('TYPEERR')

    def __check_Shape(self,Shape):
        Shape_list = ['s','g']
    
        try:
            Shape_list.index(Shape)
        except ValueError:
            print('Wrong Shape inserted: s or g\n')
            raise Exception('SHAPEERR')
            
    def __check_phase_Unit(self,Unit):
        Unit_list = ['d','r','deg','rad']
    
        try:
            Unit_list.index(Unit)
        except ValueError:
            print('Wrong Unit inserted: deg or rad\n')
            raise Exception('PHUNITERR')
    
    
        
        
            
 
#---------------------------------------------------------------- sampling------------------------------------------        
    def sampling(self,Sampling=None):
        '''
        
        This function is used to set/get (def) the Sampling Frequency property of the pulse:
        
        - Sampling is in Gigasamples/s
        
        NOTE: wait,delay and Width are stored in time (ns), changing the sampling will change
        the number of samples of the total pulse.
        
        '''
        if Sampling is None:
            return self.pars_dict['Sampling frequency']
        
        if Sampling<0:
            print('Error: sampling should be positive')
            raise Exception('value')
            
        self.pars_dict['Sampling frequency'] = Sampling
        
        
#-------------------------------------------------------------------------    delay ----------------------        
    def delay(self,Delay=None,Type='t'):
        '''
        
        This function is used to set/get (def) the delay property of the pulse:
        - Delay is in ns or samples
        - Type can be 't' for delay in ns  or 's' for delay in samples
        
        '''
        Type = Type.lower()
        self.__check_Type(Type)
        
        if Delay is None:
            if Type == 't':
                return self.pars_dict['Delay']
            else:
                return int(self.pars_dict['Delay']*self.sampling())
            
        
        if Delay <0 :
            print('Delay cannot be negative\n')
            raise Exception('DELAYERR')
        
        
        
        if Type=='s':
            self.pars_dict['Delay'] = Delay/self.sampling()
        else:
            self.pars_dict['Delay'] = Delay
                
    
#-------------------------------------------------------------------------    Width ----------------------        
    def width(self,Width=None,Type='t'):
        '''
        
        This function is used to set/get (def) the Width property of the pulse:
        - Width is in ns or samples
        - Type can be 't' for delay in ns  or 's' for delay in samples
        
        '''
        Type = Type.lower()
        self.__check_Type(Type)
        
        if Width is None:
            if Type == 't':
                return self.pars_dict['Width']
                
            else:
                return int(self.pars_dict['Width']*self.sampling())
                
            
                
            
        
        if Width <0 :
            print('Delay cannot be negative\n')
            raise Exception('DELAYERR')
        
        
        
        if Type=='s':
            self.pars_dict['Width'] = Width/self.sampling()
            
        else:
            self.pars_dict['Width'] = Width

#-------------------------------------------------------------------------    wait ----------------------        
    def wait(self,Wait=None,Type='t'):
        '''
        
        This function is used to set/get (def) the Wait property of the pulse:
        - Wait is in ns or samples
        - Type can be 't' for delay in ns  or 's' for delay in samples
        
        '''
        Type = Type.lower()
        self.__check_Type(Type)
        
        if Wait is None:
            if Type == 't':
                return self.pars_dict['Wait']
            else:
                return int(self.pars_dict['Wait']*self.sampling())
            
        
        if Wait <0 :
            print('Delay cannot be negative\n')
            raise Exception('DELAYERR')
        
        
        
        if Type=='s':
            self.pars_dict['Wait'] = Wait/self.sampling()
        else:
            self.pars_dict['Wait'] = Wait


#------------------------------------------------------------------------- total Width ----------------

    def total_width(self,Type = 't'):
        """
        This funciton returns the total pulse Width in time (ns - def) or samples.
        """
        
        Type = Type.lower()
        self.__check_Type(Type)
        
        tmp = self.wait()+self.width()+self.delay()
        
        if Type == 't':
            return tmp
        else:
            return int(tmp*self.sampling() )

#-------------------------------------------------------------------------    Sigma ----------------------        
    def sigma(self,Sigma=None):
        ''' Function sigma()
        
        This function is used to set/get (def) the sigma property of the pulse, used only if the shape of the pulse is gaussian:
        - Sigma is in ns.  Setting Sigma to 0 or to pulse Width/6 is the same.
        
        
        '''
        
        if Sigma is None:
            return self.pars_dict['Sigma']
        
        if Sigma <0:
            print('ERROR: Sigma cannot be negative\n')
            raise Exception('NEGNUM')
            
        if Sigma == 0:
            self.pars_dict['Sigma'] = self.width()/6
        else:
            self.pars_dict['Sigma'] = Sigma


#-------------------------------------------------------------------------    frequency ----------------------        

                            
    def frequency(self,Frequency=None):
        '''
        
        This function is used to set/get (def) the frequency property of the pulse:
        - Frequency (Ghz)
        
        '''

        if Frequency is None:
            return self.pars_dict['Fequency']


        if Frequency <0:
            print('ERROR: Frequency cannot be negative\n')
            raise Exception('NEGNUM')
            
        
        self.pars_dict['Fequency'] = Frequency

#-------------------------------------------------------------------------  phase ----------------------                
        
            
    def phase(self,Phase = None, Unit = 'deg'):
        '''
        
        This function is used to set/get (def) the phase property of the pulse,
        
        - Phase: the phase in Unit
        - Unit: 'd' or 'deg' for degrees, 'r' or 'rad' for radians
        
        '''
        Unit = Unit.lower()
        self.__check_phase_Unit(Unit)
        
        if Phase is None:
            if Unit == 'd' or Unit == 'deg':
                return self.np.degrees( self.__phase)
            else:
                return self.__phase
        
        if Unit == 'd' or Unit == 'deg':
            self.__phase = self.np.radians( Phase)
        else:
            self.__phase = Phase
    
#-------------------------------------------------------------------------  shape ----------------------                
        
            
    def shape(self,Shape = None):
        '''
        
        This function is used to set/get (def) the Shape property of the pulse,
        
        - Shape: it can be 's' for square pulse or 'g' for gaussian pulse
        
        NOTE: if Shape is set to 'g', the default sigma will be Width/6.
        
        
        '''
        
        if Shape is None:
            return self.pars_dict['Shape']
        
        Shape = Shape.lower()
        self.__check_Shape(Shape)
        
        
        if Shape == 'g':
            self.sigma(0)
        
        self.pars_dict['Shape'] = Shape
        

#---------------------------------------------------------------------------------------- amplitude -----------------------------        
    


        
    def amplitude(self,Amplitude=None):
        ''' 
        
        This function is used to set the Amplitude property of the pulse:
        
        - Amplitude should be between 0 and 1 (Normalized)
        
        '''
        if Amplitude is None:
            return self.pars_dict['Amplitude']
        
        if Amplitude < 0 or Amplitude > 1:
            print('ERROR: Amplitude must be [0,1]\n')
            raise Exception('NORMERR')
        
        self.pars_dict['Amplitude'] = Amplitude
        
    def offset(self,Offset=None):
        ''' 
        
        This function is used to set the Offset property of the pulse:
        
        - Offset should be between ]-1,1[ (Normalized)
        
        '''
        if Offset is None:
            return self.pars_dict['Offset']
        
        if Offset>=1 or Offset <= -1:
            print('Offset too large: ]-1,1[')
            raise ValueError
        
        self.pars_dict['Offset'] = Offset
#---------------------------------------------------------------------------------------- id -----------------------------        
    


        
    def id(self,ID=None):
        ''' 
        
        This function is used to get/set the ID property of the pulse:
        
        - ID must be a text
        
        '''
        if ID is None:
            return self.pars_dict['ID']
        
        if type(ID)!= str:
            print('ERROR: ID must be a string\n')
            raise Exception(TypeError)
        
        self.pars_dict['ID'] = ID
#-------------------------------------------------------------------------------------- generation functions  ---------#                    

    def time_axis(self):
        """
        This function is used to get the time axis for the pulse (ns). 

        """
        
        return self.np.round_(self.np.linspace(0,self.total_width(),self.total_width(Type='s' ) ), 9)
    
    def frequency_axis(self,Mode=''):
        """
        This function is used to get the fourier transform frequency axis for
        the pulse (GHz). 
        If Mode=='m' the mirror frequencies will be rendered negatives 

        """
        
        if Mode=='m' or Mode=='M':
            return self.np.round_(self.np.linspace(-self.sampling()/2,self.sampling()/2,self.total_width(Type='s')) ,9)
        else:
            return self.np.round_(self.np.linspace(0,self.sampling(),self.total_width(Type='s')) ,9)
    

    def FT(self,Mode=''):
        '''
        This function return the fourier transform of the signal. 
        Mode=='m' will return the simmetric fourier transform.'''
   
        if Mode=='m' or Mode=='M':
           return self.np.fft.fftshift(self.np.fft.fft(self.generate()))
        else:
           return self.np.fft.fft(self.generate())

    def plot(self,Mode=0):
        """
        - Mode: 
            - 1 - will plot the signal and the Fourier Transform
            - 2 - will plot the signal without interpolation (just the dots)
            - 3 - same as 2 but with the fft
    
        any other value (default is 0) will just plot the signal


        """
        
        #preparation of the time axis
        taxis = self.time_axis()
        signal = self.generate()
        
        self.pp.figure()
        if Mode==2 or Mode==3:
            self.pp.scatter(taxis,signal)
        else:
            self.pp.plot(taxis,signal)
        
        self.pp.title('Signal plot')
        self.pp.xlabel('Time (ns)')
        self.pp.ylabel('Amplitude')

        #axis adjust
        if min(signal)<0 :
            ymin=min(signal)*1.05
        elif min(signal)==0 :
            ymin=-0.1
        else:
            ymin=min(signal)*0.95
        
        if max(signal)<0 :
            ymax=max(signal)*0.95
        elif max(signal)==0 :
            ymax=-0.1
        else:
            ymax=max(signal)*1.05
    
        self.pp.axis([-signal.size/self.pars_dict['Sampling frequency']*0.05, signal.size/self.pars_dict['Sampling frequency']*1.05, ymin,ymax]);
    
        if Mode==1 or Mode==3:
            self.pp.figure()
            z=abs(self.np.fft.fftshift(self.np.fft.fft(signal)))/len(signal)*2
            x=self.frequency_axis('m')
            
            self.pp.plot(x,z)
            self.pp.title('Signal Spectrum plot')
            self.pp.xlabel('Frequency (Hz)');
            self.pp.ylabel('Amplitude');
            self.pp.axis([min(x)*1.05, max(x)*1.05, 0, max(z)*1.05]);


    def generate(self):
        '''function generate()
        
            This function will return a numpy array containing the pulse 
            data points
        '''
        
    
        if self.shape() == 'g':
            y = self.__gaussian_pulse()
        else:
            y = self.__square_pulse()

        if self.frequency() == 0:
            y= self.np.round(y,12)
        else:
            y= self.np.round(self.__modulation(y),12)
        
        if self.offset()!=0:
            #print('OFFSETTING')
            return self.__set_offset(y,self.offset())
        else:
            return y
#-----------------------------------------------------------------------------#        


        
    def __square_pulse(self):

        y=self.np.ndarray(self.total_width(Type='s'))
        a,b = self.delay(Type='s'), self.width(Type='s')
        
        
        y[0:a]=0
        y[a:a+b] = self.amplitude()
        if self.wait()>0:
            y[a+b:]=0
        
        
        
        return y
    
    


#-----------------------------------------------------------------------------#        
    
    def __gaussian_pulse(self):
        
        #center of the pulse
        m = self.width()/2.0
    
        sig= self.sigma()
        
        #Amplitude = self.amplitude()
        #x=arange(-m*L,0,L/self._GS);    
        x = self.np.linspace(0,self.width(),self.width(Type='s'))
        x = self.ne.evaluate('exp(-0.5*((x-m)/sig)**2)')
        x=x/self.np.max(x)*self.amplitude()
        
        y = self.np.ndarray(self.total_width(Type='s'))
        a,b = self.delay(Type='s'), self.width(Type='s')
        
        
        y[0:a] = 0
        y[a:a+b] = x
        if self.wait()>0:
            y[a+b:] = 0
    
        
        
        return y
        

#-----------------------------------------------------------------------------#        
        

    def __modulation(self,Pulse):
        
        
        W = 2*self.np.pi*self.frequency() / self.sampling()
        Phase = self.phase(None,'rad')
        
        x= self.np.arange(0,Pulse.size,1)
        #sin=self.np.sin
        x= self.ne.evaluate("sin(W*x+Phase)*Pulse" )
        
        
            
        return x
        
#-----------------------------------------------------------------------------#        
    def __set_offset(self,p,offset):
        old_offset = self.np.average(p)
    
        p0 = p-old_offset #no offset now
    
        if self.np.max(p0)+offset>1 or self.np.min(p0)+offset<-1:
            p1 = self.ne.evaluate("p0*(1-abs(offset))+offset")
        else:
            p1 = p0+offset
    
        return p1
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#                        

    def copy(self):
        '''

            This function is used to get a copy of the Pulse        
        
        '''
        
        a= Pulse(self.frequency(),self.phase(),self.shape(),self.width(),self.delay(),self.wait(),self.amplitude(),self.offset(),self.sigma(),self.sampling(),self.id() )
        
        return a
                    
                
            
        

#------------------------------------------------------------------------------------------------------------------------------------------------------------------#                        

    def print_properties(self):
        '''

        This function is used to print alle the Pulse properties
        
        '''
        text = '\nFrequency: {} GHz\n'.format(self.frequency())
        
        text += 'Phase: {} deg\n'.format(self.phase() )
        text += 'Pulse shape: {}\n'.format(self.shape())
        text += 'Delay: {} ns\n'.format(self.delay())
        text += 'Width: {} ns\n'.format(self.width()) 
        text += 'Wait: {} ns\n'.format(self.wait())
        text += 'Amplitude: {}\n'.format(self.amplitude())
        text += 'Offset: {}\n'.format(self.offset())
        text += 'Sigma: {} ns\n'.format(self.sigma())
        text += 'Sampling frequency: {} Gsamples/s\n'.format(self.sampling())
        text += 'ID: {}\n'.format(self.id())
                    
                
        print(text)
        

    def save(self,Filename,Overwrite=False):
        import pickle
        import os
        
        a = self.copy()
        a.par = self.par.copy()
       
        path = os.path.split(Filename)
        
        if not os.path.exists(path[0]) and path[0]:
            os.makedirs(path[0])
        
        file_name = path[1]
        
        if file_name[-4:].lower() != '.pls':
            file_name += '.pls'
            
        file_name = os.path.normpath(os.path.join(path[0]+file_name))
        
        if Overwrite is False:
            if os.path.isfile(file_name):
                print('File already exists\n')
                raise Exception('FILEEXISTS')
            
        with open(file_name,'wb') as f:
            pickle.dump(a,f,-1)
        
        
class Sequence(object):
    import numpy as np
    
    def __init__(self,pulses_list=[]):
        """Class used for concatenating pulses:
            
           - pulses_list must be a list of Pulse objects, def is empty
            """
            
        
        for p in pulses_list:
            if type(p)!= Pulse:
                print('Wrong Pulse type inserted\n')
                raise Exception(TypeError)
            
        self.pl = pulses_list
        
    
    def sequence_id(self):
        tmp=[]
        for p in self.pl:
            tmp.append(p.id())
        return tmp

    def sequence(self,num=None):
        """returns the sequence or the specified pulse"""
        if num is None:
            return self.pl
        else:
            return self.pl[num]
    
    def generate_sequence(self,auto_phase=True):
        """generate the sequence, optionally correcting the phase.
        
        - auto_phase (def True), activate automatical phase generation
        
        if auto_phase is on, the phase of the pulses with index >0 is considered
        relative to the first pulse, so if the phase of these pulses is 0, the
        phase is continuous"""
        
        
        p = self.pl[0]
        freq = p.frequency()
        phase_tmp = 360.*freq*p.total_width()+p.phase() #this is the phase at the end of the pulse
        
        tmp = p.generate()
        if auto_phase:
            for p in self.pl[1:]:
                freq = p.frequency()
                ptmp = p.copy()
                ptmp.phase(phase_tmp+ptmp.phase())
                
                tmp = self.np.hstack((tmp,ptmp.generate()))
                phase_tmp += 360.*freq*p.total_width()
            
            return tmp
        else:
            for p in self.pl[1:]:
                
                tmp = self.np.hstack((tmp,p.generate()))
                
            
            return tmp
        
    def generate_sequence_for_AWG(self,auto_phase=True,auto_zero=True):
        """Return a Nx2 Matrix, where N is the number of pulses in the sequence
        
        The first column element is a number corresponding to the pulse delay,
        the second column contains the pulse array.
        
        - auto_phase (def True), activate automatical phase generation
        
        NOTES:
            - Wait time are added to the next pulse delay, the last wait time 
            is ignored
            - if auto_phase is on, the phase of the pulses with index >0 is considered
              relative to the first pulse, so if the phase of these pulses is 0, the
              phase is continuous
             - auto_zero (def True): the last point of the pulse is set to zero,
               the AWG keeps the last value costant
        """
        p = self.pl[0]
        freq = p.frequency()
        phase_tmp = 360.*freq*p.total_width()+p.phase() #this is the phase at the end of the pulse
        
        delay_list = [p.delay()]
        
        ptmp = p.copy()
        ptmp.delay(0)
        ptmp.wait(0)
        pulse = ptmp.generate()
        plist = [pulse]
        ids = [p.id()+'-N0']
        
        prev_wait= p.wait()
        if auto_phase:
            for i,p in enumerate( self.pl[1:]):
                freq = p.frequency()
                #adding pulse after erasing the delay and wait time
                ptmp = p.copy()
                ids.append(ptmp.id()+'-N{}'.format(i+1))
                ptmp.phase(phase_tmp+360*freq*p.delay()+ptmp.phase())
                ptmp.delay(0)
                ptmp.wait(0)
                pulse = ptmp.generate()
                plist.append(pulse)
                
                delay_list.append(p.delay()+prev_wait)
                #recalculating phase and storing wait time
                phase_tmp += 360.*freq*p.total_width()
                prev_wait = p.wait()
            
            
        else:
            for i,p in enumerate(self.pl[1:]):
                
                #adding pulse after erasing the delay and wait time
                ptmp = p.copy()
                ids(ptmp.id()+'-N{}'.format(i+1))
                ptmp.delay(0)
                ptmp.wait(0)
                
                pulse = ptmp.generate()
                if auto_zero:
                    pulse[-1]=0
                plist.append(pulse)
                
                delay_list.append(p.delay()+prev_wait)
                #recalculating phase and storing wait time
                
                prev_wait = p.wait()
            
        if auto_zero:
            last_piece = self.np.zeros(10)
        else:
            last_piece = []
        
        plist[-1]= self.np.hstack((plist[-1],self.np.zeros(self.np.int(prev_wait*p.sampling())),last_piece ))
        
        return [ids,delay_list,plist]
                
    def total_width(self):
        """
        This funciton returns the total sequence length in ns.
        """
        tmp=0
        for p in self.pl:
            tmp += p.total_width()
        
        return tmp
        
    def time_axis(self):
        return self.np.arange(self.total_width())

    def plot(self,auto_phase=True):
        import matplotlib.pyplot as plt
        plt.plot(self.time_axis(),self.generate_sequence(auto_phase))
        plt.xlabel('Time (ns)')
        plt.ylabel('Normalized amplitude')
        
    def copy(self):
        '''

            This function is used to get a copy of the Sequence        
        
        '''
        new_list=[]
        for p in self.pl:
            new_list.append(p.copy())
        a= Sequence(new_list)
        
        return a