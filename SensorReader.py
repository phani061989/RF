"""
__author__ = 'oscar'

Utility used to  get data from the Triton system control, like the cryostat base temperature.

v2.2.0 - OSC:
    - read_table is deprecated, changed in read_csv
    
v2.1.0:
    - the file_path where the file with temperatures is can now be specified

v2.0.0
- adapted to new logfiles format

v1.1.0
- Implemented multi-cryo support

"""

print('Sensor reader v2.2.0')

import pandas as pnd

class SensorReader(object):
    
    def __init__(self,CryoID,BaseSensor='T_MC_RuOx',file_path='/home/measurement/DataLogging/Fridges/'):
        
        
        CryoID = str(CryoID)
        
        self.path = file_path+'/'+CryoID+'.last'
        
        self.DEFBASESENS= BaseSensor
        
        self.__time_format = '%Y-%m-%d %H:%M:%S'

        
        
        self.data = pnd.read_csv(self.path,index_col=0,sep=',',names=['Values'])
        tmp = self.data.iloc[-1].name
        self.data.index.values[-1]='Last Update'
        self.data.Values.values[-1] = tmp
    
        

    def update(self):
        """function update()

        updates the column of the values"""        
        
        timeout=0
        
        while(timeout<10):
            try:
                data = pnd.read_csv(self.path,index_col=0,sep=',',names=['Values'])
                self.data['Values']=data['Values'].values
                self.data.Values.values[-1] = data.index.values[-1]
                        
#                print('Updated from file:',self.data.iloc[-1].values[0])
                break
            except ValueError:
                timeout+=1
        if timeout == 10:
            print('failed to update')


    def sensor_list(self):
        """ function sensor_list()
        
        this function return the index column, one can use this names to access the sensor with the function:
        
        <name>.data.loc[index]
        
        NOTE: index is a string

        Example:

        a = SensorReader()
        a.data.loc['WIN']

        In this way we will get the Descripion, Name and Value of the sensor of the Pulse Tube Water In        
        
        """
        return self.data.index


    def base_temp(self):
        '''function base_temp()
        
        returns the temp of the Mixing Chamber from the sensor RuOx as a number in mK'''
        
        self.update()
        tmp = self.data.loc[self.DEFBASESENS].values[0]
        return float(tmp[:-1])*1e3
    
    def last_update(self):
        '''function last_update()
        
        This function will return the last update of the temperature sensor'''
        
        return self.data.iloc[-1].values[0]