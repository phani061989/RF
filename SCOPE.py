# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:04:53 2014

@author: Oscar Gargiulo


Library used to control the Tektronix DSA Scope

v1.2.1 - OSC:
    - migrated to visa

v1.2.0 - OSC:
    - adapted to Instruments class
    - missing mode query

v1.1.2:
    - adapted to the new data_table module

v1.1.1:
    - added the function get_channel_scale

version: 1.1.0

- The function read will returns a data module (2d), instead of arrays

"""



import DataModule as dm
class Scope(object):
    """Class Init parameters:


    chosen_name=Scope(    
    
    - SCOPE_IP_ADDRESS = "192.168.0.99",
    - DEF_PC_FOLDER = "", NOTE: it is not used
    - DEF_SCOPE_FOLDER = "C:\\scopesaves"
    
    )
    
    the arguments have a default value (showed), so it is possible to create the class without arguments.
    
    Examples:
    
    - sc=Scope()
    - sc=Scope(DEF_PC_FOLDER='c:\\example')
    
    NOTE: Don't include a \\ at the end of a path
    
    
    """
    import visa
    import numpy as np
    import time
    #import os
    version = '1.2.1'
#------------------------------------------------------------------------------    
    def __init__(self,SCOPE_IP_ADDRESS='192.168.0.99',DEF_PC_FOLDER="",DEF_SCOPE_FOLDER="C:\\scopesaves"):
            self._inst = None
            self.__SF_MIN = 5e-3 #Gigasamples/s
            self.__SF_MAX = 100 #Gigasamples/s
            self.__CHLIST=['CH1','CH2','CH3','CH4','MATH1','MATH2','MATH3','MATH4','REF1','REF2','REF3','REF4']   
            self.__MODELIST=['SAM','PEAK','HIR','AVE','WFMDB','ENV','FASTACQ','FASTFRAME']
            
            
            self.SCOPE_IP_ADDRESS = SCOPE_IP_ADDRESS
            self.DEF_PC_FOLDER = DEF_PC_FOLDER
            self.DEF_SCOPE_FOLDER = DEF_SCOPE_FOLDER
            
            self.connect()

    def com(self, command, arg="?"):
        """Function to communicate with the device. Gives the current status
        if no arg is given"""
        if arg == "?":
            resp =  self._inst.query("{}?".format(command))
            try:
                return float(resp)
            except:
                return resp.strip('\n')
                
        else:
            self._inst.write("{} {}".format(command, arg))
            return 0
#------------------------------------------------------------------------------ Utilities functions
    
    class __BASEEXC(Exception):
        pass
    
    class SCEXC(__BASEEXC):
        def __init__(self,expression,message):
            self.expression = expression
            self.message = message
    
    def connect(self):
        """This function is used to connect python to the scope.\nThe function will print the instrument ID.\n
    
    """

        try:
            rm = self.visa.ResourceManager()
            self._inst = rm.open_resource('TCPIP::{}::INSTR'.format(self.SCOPE_IP_ADDRESS) )
        except:
            print("Connection Failed")
            raise
        
        
        print(self.com("*IDN"))
        
        

    def disconnect(self):
           
        self._inst.close()
        self._inst=None

    def __check_channel(self,Channel):
        if type(Channel)==str:
            try:
                self.__CHLIST.index(Channel.upper())
                return Channel.upper()
            except ValueError:
                raise self.SCEXC('CHNAME','Wrong channel ID inserted: {}'.format(Channel))
        else:
            Channel = int(Channel-1)
            if Channel<0 or Channel >3:
                raise self.SCEXC('CHNUM','Wrong channel number inserted: {} / [1,4]'.format(Channel))
            
            return self.__CHLIST[Channel]
            
                
    
#----------------------------------------------------------------------------------------------------------------------------- 
#-------------------------------------------------------------------------- Parameters FUNCTIONS -----------------------------
#-----------------------------------------------------------------------------------------------------------------------------
    
    def sampling_frequency(self,SF='?'):
        '''
        This function gets or sets the scope sampling frequency in GS/s.
        '''
        
        command = 'hor:mode:sampler'
        
        if SF == '?':
            return self.com(command,SF)/1e9
        else:
            if SF<self.__SF_MIN or SF>self.__SF_MAX:
                raise self.SCEXC('WRONGSF','Error: SF is out of the range ['+str(self.__SF_MIN)+', '+str(self.__SF_MAX)+']')
            else:
                self.com(command,SF*1e9)
                print('The sampling frequency has been set to +'+str(self.sampling_frequency())+' GS/s')
            
#----------------------------------------------------------------------------------------------------------------------------------    
    

    def averages(self,Averages='?',Channel=1):
        ''' 
        This function gets or sets the number of averages that the specified 
        math channel will do:
        
            - Number is the math channel number, it can be 1 (def), 2, 3, 4
        '''
        
        if type(Channel)!=str:
            Channel = 'MATH'+str(int(Channel))
        
        Channel = self.__check_channel(Channel)
        
        command = Channel+':NUMAVG'
        
        if type(Averages)!=str:
            try:
                Averages=int(Averages)
            except:
                print('Averages must be an integer number')
                raise Exception(ValueError)
            
            
        tmp = self.com(command,Averages)
        if tmp is not None:
            return int(tmp)

    
#----------------------------------------------------------------------------------------------------------------------------------    

    def activate_channel(self,State='?',Channel=1):
        ''' function set_channel_onoff(Channel='ch1',State='on'):
        
        This function is used to turn on/off a scope channel, or all of them
        - Channel can be 'chN','mathN','refN' with N=1,2,3,4 or 'a' / 'all'
        - State can be 'on' or 'off'
        '''
        State_list = ['OFF','ON','?']
        
        if type(State) == str:
            try:
                State_list.index(State.upper())
            except ValueError:
                raise self.SCEXC('STATEERR','Wrong state inserted: [ON,OFF]')
        else:
            State = int(State)
            try:
                if State==2:
                    raise Exception(ValueError)
                State = State_list[State]
            except ValueError:
                raise self.SCEXC('STATEERR','Wrong state inserted: [0,1]')
                
        if Channel=='a' or Channel=='all' or Channel=='A' or Channel=='ALL':
                for i in self.__CHLIST:
                    self.com('SEL:'+i,State)
        else:
            Channel = self.__check_channel(Channel)
                
            tmp = self.com('SEL:'+Channel,State)
            if tmp is not None:
                return State_list[int(tmp)]
                
            
#----------------------------------------------------------------------------------------------------------------------------------    

    def offset(self,Offset='?',Channel='ch1'):
        ''' function set_channel_offset(Offset,Channel='ch1'):
        
        This function is used to set the offset of a scope channel, or all of them
        - Channel can be 'chN','mathN','refN' with N=1,2,3,4 or 'a' / 'all'
        
        '''
        Channel = self.__check_channel(Channel)
        
            
        
        if Channel=='a' or Channel=='all':
            for i in self.__CHLIST:
                self.com(i+':offset',Offset)
        else:
            
            tmp = self.com(Channel+':offset',Offset)
            if tmp is not None:
                return int(tmp)
        
#----------------------------------------------------------------------------------------------------------------------------------    

    def scale(self,Scale='?',Channel='ch1'):
        ''' function set_channel_offset(Scale (mV),Channel='ch1'):
        
        This function is used to set the offset of a scope channel, or all of them
        - Channel can be 'chN','mathN','refN' with N=1,2,3,4 or 'a' / 'all'
        - Scale is in mV/div
        '''

        Channel = self.__check_channel(Channel)        
         
        if type(Scale)!=str:   
            Scale *= 1e-3
            
        if Channel=='a' or Channel=='all':
            for i in self.__CHLIST:
                self.com(i+':scale', Scale)
        else:
            
            tmp = self.com(Channel+':scale', Scale)
            if tmp is not None:
                return int(tmp*1e3)
    
 
    
#----------------------------------------------------------------------------------------------------------------------------------    

    def acquisition_duration(self,Time='?'):
        ''' function set_acquisition_duration(Time (ns)):
        
        This function is used to set the acquisition time
        - Time is in ns
        '''
        
        command = 'HOR:MODE:SCAlE'
        
        if type(Time)!=str:
            Time = Time*1e-10
            
        tmp = self.com(command,Time)
        if tmp is not None:
            return int(tmp*1e10)
        
    

#----------------------------------------------------------------------------------------------------------------------------------    

    def horizontal_position(self,Time='?'):
        ''' function set_horizontal_position(Time (ns)):
        
        This function is used to get/set the horizontal delay time
        - Time is in ns
        '''
 
        command = 'HOR:POS'
        
        if type(Time)!=str:
            Time = Time*1e-9
            
        tmp = self.com(command,Time)
        if tmp is not None:
            return tmp*1e9
       
        

#----------------------------------------------------------------------------------------------------------------------------------    

    def set_mode(self,Mode='SAM',*args):
        ''' function set_mode(Mode='SAM'):
        
        This function is used to set the acquisition mode, default is Sample
        - Mode can be:
            'SAM' default
            
            'AVE': Is used for averages (def 10000), use set_mode('AVE',MAX_NUMBER_OF_AVERAGES)
            
            'FASTACQ': fast acquisition mode enabled
            
            'FASTFRAME': all triggered waves are registered one after another, it is possible (and you should!) specify the maximum number of acquired waves
            
        '''
        
        if type(Mode)!=str:
            print('Wrong Mode inserted 1')
            raise Exception('Mode')
        
        try:
            index=self.__MODELIST.index(Mode.upper())
            self.com('HOR:FASTFRAME:STATE','OFF')
            self.com('FASTAcq:STATE', 'OFF')
            self.com('ACQ:STOPAfter', 'RUNSTop')
            
            if index==0:
                self.com('ACQ:MOD',self.__MODELIST[index])
            elif index==3:
                if len(args)==0:
                    self.com('ACQ:MOD',self.__MODELIST[index])
                    self.com('ACQ:NUMAV', 10000)
                else:
                    if len(args)==0:
                        tmp=1000
                    else:
                        tmp=repr(float(args[0]))
                        
                    self.com('ACQ:MOD',self.__MODELIST[index])
                    self.com('ACQ:NUMAV',tmp)
            elif index==6:
                self.com('FASTAcq:STATE', 'ON')
            elif index==7:
                self.com('ACQ:MOD',self.__MODELIST[0])
                self.com('HOR:FASTFRAME:STATE', 'ON')
                if len(args)==0:
                    tmp=1
                else:
                    tmp=args[0]
                
                self.com('HOR:FAST:COUN',tmp)
                    
        
            else:
                print('Mode not yet implemented')
                raise Exception('Implementation')
        except ValueError:
            print("Wrong number inserted")
            raise
        except:
                print('Wrong Mode inserted 2')
                raise Exception('Mode')

                    
    

        #{SAMple|PEAKdetect|HIRes|AVErage|WFMDB|ENVelope}

#----------------------------------------------------------------------------------------------------------------------------------    

    def get_acquisitions(self):
        ''' get_acquisitions():
        
        This function is used to return the number of waves acquired by the scope
        '''
        
        return int(self.com("ACQ:NUMACQ"))

#----------------------------------------------------------------------------------------------------------------------------------    

    def trigger(self,Level='?',Channel='AUX'):
        ''' set_trigger(Channel='AUX',Level=''):
        
        This function is used to set the scope trigger.
        
            - Channel can be:
                'CH1' to 'CH4' or 'AUX' (default)
            - Level is the voltage level used for the trigger (in Volt), if Level is None (default) it will be set to 'auto'
        '''
        if Level == '?':
            return self.__get_trigger()

        if type(Channel)!=str:
            Channel = int(Channel)
            if Channel<1 or Channel > 4:
                raise self.SCEXC('TRIGCHERR','Wrong channel number inserted: {} / [1,4]'.format(Channel))
        elif Channel.upper()!='AUX' and Channel.upper()!='CH1' and Channel.upper()!='CH2' and Channel.upper()!='CH3' and Channel.upper()!='CH4' :
            raise self.SCEXC('TRIGCHERR','The specified channel is wrong: {} / [ch1,ch2,ch3,ch4,aux]'.format(Channel))
            
        self.com('TRIG:A:TYP', 'EDGE')
        self.com('TRIG:A:EDGE:SOU',Channel)
        
        if Level == 'AUTO' or Level == 'auto':
            self.com('TRIG:A:MOD', 'AUTO')
        else:
            self.com('TRIG:A:MOD', 'NORM')            
            self.com('TRIG:A:LEV',Level)

#----------------------------------------------------------------------------------------------------------------------------------    

    def __get_trigger(self):
        '''(Channel,Level) = get_trigger():
        
        This function is used to get the trigger configuration        
        '''
        if self.com('TRIG:A:MOD')=="AUTO":
            return (self.com('TRIG:A:EDGE:SOU'), "AUTO")
        else:
            return ( self.com('TRIG:A:LEV'),self.com('TRIG:A:EDGE:SOU'),)
    
        
#----------------------------------------------------------------------------------------------------------------------------------    
    def run(self,State='RUN'):
        ''' function Run([State='RUN'])
        
            this function can start or stop the acquisition.
            
            State can be:
            - 'RUN' or 1 (default) 
            - 'STOP' or 0'''
            
        if type(State) == str:
            if State.upper()=='RUN'  :              
                self.com('ACQuire:STATE', 'RUN')
            elif State.upper() == 'STOP':
                self.com('ACQuire:STATE', 'STOP')
            else:
                print('Wrong state inserted')
                return
        elif State==1:
            self.com('ACQuire:STATE', 'RUN')
        elif State==0:
            self.com('ACQuire:STATE', 'STOP')
        else:
                print('Wrong state inserted')
                return
#----------------------------------------------------------------------------------------------------------------------------------
                
    def state(self):
        '''function state():
        
        this function is used to get the scope trigger state.
        
        Examples:
        'SAVE': the scope is stopped and it is possible to read data (in some acquisition mode it is possible to save data also in 'READY' mode)
        'READY': the scope is ready for the acquisition
        '''
        
        return self.com('TRIG:STATE')
#----------------------------------------------------------------------------------------------------------------------------------
    def scope_setup(self,Filename,Operation='Load'):
        '''function scope_setup(self,Filename,[Operation='Load'])
        
This function is used to save the scope setup to filename or to load (default) it.
The folder user will be the default setup folder on the scope.

- Operation: 

    - 'save' for saving
    - 'load' for loading
                
        '''
        
        if Operation.upper()=='SAVE':
            self.com("SAVE:SETUP \""+Filename+".SET\"",'')
        elif Operation.upper()=='LOAD':
            self.com("RECALL:SETUP \""+Filename+".SET\"",'')            
        else:
            print("ERROR: Operation not valid")
            return
    
#----------------------------------------------------------------------------------------------------------------------------------    
    def clear(self):
        ''' function clear()
        
        Clear the screen and all information acquired'''
        
        self.com('clear all','') 

#----------------------------------------------------------------------------------------------------------------------------- 
#-------------------------------------------------------------------------- MEASURMENTS FUNCTIONS -----------------------------
#-----------------------------------------------------------------------------------------------------------------------------



    def read(self,Channel_name,Framestart=1,Framestop='L'):
        '''function [x,]y = read(Channel_name,[Framestart=1],[Framestop='L'],[Taxis=False]):\n
    This function will return measured signal registered on the specified channel. \n
    
    - Channel_name should be one of the following:\n
    
    - 'ch1' 'ch2' 'ch3' 'ch4'
    - 'math1' 'math2' 'math3' 'math4'
    
    - Framestart (def 1): specify which is the first frame to be transferred (it should be used in FastFrame mode)
    - Framestop (def 'L'): specify which is the last frame to be transferred (it should be used in FastFrame mode)    
    Framestart and Framestop can be setted to 'L' to specify the last one acquired
    
    Taxis (def False): if it is True, the time axis will be automatically generated, the output will be: (time,signal)
    
    '''
       
        lastframe=self.com("hor:fast:coun")
        
        if type(Framestart)==str:
            if Framestart!='l' and Framestart=='L' and Framestart!='a' and Framestart!='A':
                print('Error: Framestart should be a number, or \'L\' or \'A\'')
                return
        else:
            if Framestart>int(lastframe):
                print("Error: max frame is "+lastframe)
                return
        

        if type(Framestop)==str:
            if Framestop!='l' and Framestop!='L' and Framestop!='a' and Framestop!='A':
                print('Error: Framestop should be a number, or \'L\' or \'A\'')
                return
        else:
            if Framestop>int(lastframe):
                print("Error: max frame is "+lastframe)
                return
        
        
        if type(Framestart)!=str and type(Framestop)!=str:
            if Framestart>Framestop:
                print('Error: Framestart should be less or equal to Framestop')
                return
        
        #check number of samples
        samples=int(self.com("horizontal:mode:recordlength"))
        
        sampling_period= self.com("WFMOutpre:XINCR")
    
        Channel_name=Channel_name.upper()
        self.com("Data:source",Channel_name)    
        
        if self.com("data:source") != Channel_name:
            print("Channel_name is not correct")
            raise ValueError
    
        #Position
        position= self.np.double(self.com(Channel_name+":Position"))
        position*= self.np.double(self.com(Channel_name+":scale"))                
    
        self.com("Data:start", 1)
        self.com("Data:stop", samples)
        self.com("Data:width", 1)
        self.com("Data:enc", "ascii")
        
        if self.com('HOR:FASTFRAME:STATE'):
        
            
            
            if Framestart=='a'     or Framestart=='A' or Framestop=='a' or Framestop=='A':
                y=[]
                for i in range(int(lastframe)):
                    self.com("dat:framestart", i)
                    self.com("dat:framestop", i)
    
                    yt=self.com("Curve")
                    vscale= self.com("WFMOutpre:YMULT")
                    yt= self.np.array(self.np.double(yt.split(',')))
                    yt*=vscale
                    yt-= self.np.double(position)
                    
                    y.append(yt)
            else:
                if Framestart=='l' or Framestart=="L":
                    Framestart=lastframe
        
                if Framestop=='l' or Framestop=="L":
                    Framestop=lastframe
                
                y=[]                
                for i in range(int(Framestop)-int(Framestart)+1):
                    self.com("dat:framestart", Framestart+i)
                    self.com("dat:framestop", Framestart+i)

                
                    yt=self.com("Curve")
                    vscale= self.com("WFMOutpre:YMULT")
                    yt=self.np.array(self.np.double(yt.split(',')))
                    yt*=vscale
                    yt-= self.np.double(position)
                    
                    y.append(yt)
                
    
        
        else:    
            y=self.com("Curve")
            vscale= self.com("WFMOutpre:YMULT")
            y= self.np.array(self.np.double(y.split(',')))
    
            if  self.com('fasta'):
                z=[]
                for i in range(1000):
                    z.append(y[252*i:252+252*i])
                z=self.np.array(z).transpose()
    
                #each row of z must be converted
                y=self.np.ndarray(1000)
                for j in range(1000):
                    a=z[:,j]
                    y[j]=(self.np.argmax(a)-127)*vscale-position
                    
    
                
            
                
            else:
                y*=vscale
                y-= position
        
        
        #Time axis generation
        
        xres= self.com("horizontal:main:scale")
        x_start= self.com("horizontal:main:position")/10*xres
        
        if  self.com('fasta'):
                x= self.np.linspace(x_start,round(10*xres,10),1000)
        else:
                x= self.np.linspace(x_start,y.size*sampling_period*1e9,y.size)
        
        data = dm.data_table((x,y),('Time (ns)','Amplitude (V)'))
        
        return data

#------------------------------------------------------------------------------
        
    def save_curve(self,Filename,Channel_name):
        """function save_curve(Filename,Channel_name):
        
    This function is used to save the waveform on the channel_name on a memory\n
    The filename should include the whole path, the file extension will be added automatically, the saving format is Matlab dat file.\n
    
    Example: save_curve('z:\\curves\\temp','ch2') """
        
        
        try:
            Channel_name=Channel_name.upper()    
            state=int(self.com("Select:"+Channel_name))
        except:
            print("Error: probably the channel name is wrong")
            return
            
        if state==0:
            print("Warning: the selected channel is not active")
        
        
        self.com("SAVE:WAVEFORM:FILEFORMAT", "MATLAB")
        self.com("SAVE:WAVEFORM "+Channel_name+",\""+Filename+".dat\"",'')

#------------------------------------------------------------------------------
        
    def single_acquisition(self,State='?'):
        '''This function gets/sets the single acquisition (if it is off, the scope will
        acquire continuously).
        '''
    
        state_list = ['RUNSTOP','SEQUENCE']
        self.com("ACQuire:STATE", "OFF")
        
        if type(State)!= str:
            try:
                State = state_list[int(State)]
            except IndexError:
                self.SCEXC('NUMERR','Wrong State inserted : {} / [0,1]\n'.format(int(State)))
        
        tmp = self.com("ACQUIRE:STOPAFTER",State)
        if tmp is not None:
            if state_list.index(tmp)==0:
                return 'OFF'
            elif state_list.index(tmp)==1:
                return 'ON'
            else:
                self.SCEXC('UNKNOWN','An unknown error occurred!')
        

        
        
#------------------------------------------------------------------------------        
        
    def op_wait(self,timeout=5):
        start=self.self.time.time()

        while self.com('BUSY'):
            self.time.sleep(1)
            if self.time.time()-start > timeout:
                print('Scope not ready')                    
                return False       
        
        return True
#------------------------------------------------------------------------------        
    def reset(self,timeout=5):
        #import time
        
        
        self.run(0)
        self.op_wait(timeout)   
        self.com('*WAI','')
        self.run(1)
        
        start=self.time.time()
        while True:
                        
                if self.com("TRIG:STATE")=='READY':
                    return True          
                
                self.run(1)
                
                if (self.time.time()-start)>timeout:
                    print("Scope time out")
                    return False
                

        
       
            
        
            
        
#------------------------------------------------------------------------------
    def force_trigger(self):
        '''function force_trigger()
        
This function is used to send a trigger to the scope        
        '''
        
        self.com("trigger", "force")



        
