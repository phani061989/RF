# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 12:27:31 2015

@author: Oscar


Library used to control the R&S spectrum analyzer
"""

print('SPECAN v1.0.0')

class Specan(object):
    version='1.0.0'
    import visa
    #import time
    import numpy as np
    #import scipy as scipy
    #from scipy import optimize,signal
    #import matplotlib.pylab as plt

    def __init__(self,IP='192.168.0.115'):
        rm = self.visa.ResourceManager()
        self._sa = rm.open_resource('TCPIP::{}::INSTR'.format(IP) )
        
        self._sa.timeout=2**20
        print(self.Identify())

    def __trace_check(self,Trace):
        Trace=int(Trace)
        
        if Trace<1 or Trace>6:
            return False
        
        else:
            return True
        
    # Close connection
    def Close(self):
        self._sa.close()
    
    def Identify(self):
        return self._sa.ask('*IDN?')


    def Ref(self,Type='EXT'):
        """ function Ref([Type='EXT']
        
        This function will set the frequency reference as external 10 MHz or internal
        
        Type can be:
            - "EXT" (def): for a 10MHz external reference
            - 'INT': for no external reference
        """
        
        if Type.upper()=='EXT':
            
            self._sa.write('SOUR:EXT:ROSC EXT')
        elif Type.upper()=='INT':
            self._sa.write('SOUR:EXT:ROSC INT')
        else:
            print('ERROR: Wrong Type inserted')
            raise Exception('TypeError')
        
    #Data acquisition
    def Read(self,Freq='A',Trace=1):
        '''Function read([Freq='A'],[Trace=1])
        
        This function is used to get the data array
        
        - Trace: is the number of the trace 1 (def) to 6
        - Freq: def is 'A', the function will return the tuple (x,y) where x is the freq axis and y is the spectrum value
                it is possible to set Freq (GHz) to a specified value or a set of values and get only the spectrum amplitude 
                corresponding to the specified frequency/s.
                
        Examples:
        
        x,y = sa.read()
        y= sa.read(9)
        y= sa.read((1.,3, 5))
        y=sa.read([1.,3,5])
        '''
        #Trace check
        Trace=int(Trace)
        if Trace<1 or Trace>6:
            print('Error: Trace must be an integer between 1 and 6')
            raise('ValueError')
        
        start= self.np.float(self._sa.ask('FREQ:START?'))
        stop= self.np.float(self._sa.ask('FREQ:STOP?'))
        
        #Freq choice
        if type(Freq)==str:
            if Freq.upper()=='A':
                x= self.np.linspace(start,stop,self.Points())        
                y= self._sa.ask('TRAC? TRACE'+str(Trace))
                return x, self.np.array(y.split(','),dtype=self.np.float64)
            else:
                print('Error: Freq can be a number or \'A\'')
                raise Exception('FreqError')
        else:
            
            y = self._sa.ask('TRAC? TRACE'+str(Trace))
            y = self.np.array(y.split(','),dtype=self.np.float64)
            
            if type(Freq)==tuple or type(Freq)==list or type(Freq)== self.np.ndarray:
                result=self.np.ndarray(len(Freq))
                
                for i,f in enumerate(Freq):
                    index= f*1e9/(stop-start)*y.size
                    result[i]= y[int(round(index,9))]
                
                return result
            
            else:
                index= (Freq*1e9-start)/(stop-start)*y.size
                return y[int(round(index))]
    
    #Trace commands
    def Span(self,Span=None):
        ''' function Span([Span=None]):
        
        This function is used to set the Span width (MHz), if it is None, it will be queried
        '''
        if Span is None:
            return self.np.round(self.np.float(self._sa.ask("FREQ:SPAN?"))/1e6,9)
        else:
            self._sa.write("FREQ:SPAN "+str(Span)+"MHz")

    def Center(self,Center=None):
        ''' function Center(Center=None):
        
        This function is used to set the Center frequency of the Span (GHz)
        '''
        
        if Center is None:
            return self.np.round(self.np.float(self._sa.ask("FREQ:CENT?"))/1e9,9)
        else:
            self._sa.write("FREQ:CENT "+str(Center)+" GHz")
        
    def Points(self,Points=None):
        ''' function Points([Points=None]):
        
        This function is used to set or get the number of points in the Span, it can be:
        
        - "MIN": 101 Points
        - "MAX": 32001 Points
        - an integer between 101 and 32001
        - If it is None (def) the number of Points used will be queried
        '''
        if Points is None:
            return self.np.int(self._sa.ask("SWE:POIN?"))
        elif type(Points)==str:
            if Points.upper()=="MAX":
                Points=32001
            elif Points.upper()=="MIN":
                Points=101
            else:
                print("ERROR: Wrong Points inserted")
                raise Exception("PointsError")
        else:
            Points=int(Points)
            if Points<101 or Points>32001:
                print('Wrong number of Points inserted: muse be an integer between 101 and 32001')
                raise Exception('PointsError')
            else:
                self._sa.write("SWE:POIN "+str(Points))
        
    def Auto(self,Mode='ALL'):
        '''function Auto([Mode='ALL']):
        
        This function will set automatically the ideal insturment configuration for the parameters specified:
        
        - Mode can be:
            - 'ALL' (def): All parameters will be automatically set up
            - 'FREQ': the highest level in the frequency span will be set as center frequency
            - 'LEV': Automatically adjust the level
        '''
        
        if Mode.upper() == 'ALL':
            self._sa.write('ADJ:ALL')
        elif Mode.upper()=='FREQ':
            self._sa.write('ADJ:FREQ')
        elif Mode.upper()=='LEV':
            self._sa.write('ADJ:LEV')
        else:
            print('ERROR: Wrong Mode inserted')
            raise Exception('ModeError')
            
    def Averages(self,State,Count=0,Trace=1):
        ''' function Averages(State,[Count=0],[Trace=1]):
        
        This function will turn on/off the averages on the specified Trace (1 def).
        
        - State can be 'ON' or 1, 'OFF' or 0
        - Trace can be 1 (def) to 6
        '''
        
        if type(State)==str:
            if State.upper()=='ON':
                State=1
            elif State.upper()=='OFF':
                State=0
            else:
                print('ERROR: Wrong State inserted')
                raise Exception('StateError')
        
        
        if self.__trace_check(Trace) == False:
            print('Wrong Trace inserted')
            raise Exception('TraceError')
        
        if State==1:
            
            
            Count=int(Count)
            if Count<0 or Count>32767:
                print('Error: Count must be an integer between 0 and 32767')
                raise Exception('CountError')
            else:
                self._sa.write('SWE:CONT: OFF')
                self._sa.write('AVER:COUN '+str(Count))
                self._sa.write('AVER:STAT'+str(Trace)+' ON')
            
        elif State==0:
            self._sa.write('AVER:STAT'+str(Trace)+' OFF')
        else:
            print('Wrong State inserted')
            raise Exception('StateError')

    def Count(self):
        '''function Count()
        
        This function return the current averages number'''
        
        return self.np.float(self._sa.ask('SWE:COUN:CURR?'))
        
    def Continuous(self):
        '''
        function Continuous(Active=1):

        This function turn on the continuous acquisition mode
        '''
    
        self._sa.write('INIT:CONT ON')
    
    def Single(self):
        '''
        function Continuous(Active=1):

        This function turn on the single  acquisition mode        
        '''

        self._sa.write('INIT:CONT OFF')
    
    def Run(self):
        """function Run():
        
        This function start a new acquisition when the instrument is in single mode        
        """
        
        self._sa.write("INIT")
    
    def BWRes(self,Res="AUTO"):
        """function BWRes([Res="AUTO']):
        
        This function set the Bandwidth resolution (MHz):
        
        """
        
        if type(Res)==str:
            if Res.upper()=="AUTO":
                self._sa.write("BAND:AUTO ON")
            else:
                print("ERROR: Wrong Res inserted")
                raise Exception("BWRESEXC")
        else:
            self._sa.write("BAND:AUTO OFF")
            self._sa.write("BAND "+str(Res)+" MHz")