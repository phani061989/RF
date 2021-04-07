# -*- coding: utf-8 -*- #######################################################
"""
Created on Tue Sep 23 10:03:21 2014

@author: Oscar Gargiulo, Christian Schneider

This library is created to control the R&S  Digital attenuator

v1.0.2 - OSC:
    - migrated to visa
"""
VERSION = '1.0.2'
print('DIGATT v{}'.format(VERSION))
###############################################################################
import visa


class Digatt(object):
    """ Instrument class for controlling of a digital attenuator via vxi11."""

    def __init__(self, ip='192.168.0.131'):
        self.__ATT_MIN = 0  # dB
        self.__ATT_MAX = 115  # dB
        self.__MAX_FREQ = 18  # GHz
        self.__ATT_STEP = 5  # dB
        self.__ATT_LIST = ['INT', 'EXT1', 'EXT2', 'EXT3', 'EXT4']
        self.ip = ip
        
        # Initialize connection
        rm = visa.ResourceManager()
        self._inst = rm.open_resource('TCPIP::{}::INSTR'.format(ip) )
        print(self.identify())
    # Downward compatibility to v1.0.0
        
    def identify(self):
        return self.com('*IDN')

    def close(self):
        '''close connection to the instrument'''
        self._inst.close()
        
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
    
    def get_Att_idx(self, Type):
        try:
            if type(Type) == str:
                Num = self.__ATT_LIST.index(Type.upper())
                return Num
            else:
                Num = abs(int(Type))
                if Num > 4:
                    print('Wrong number inserted. Use internal instead.')
                    return 0
                else:
                    return Num
        except:
            print('Wrong type inserted. Use internal instead.')
            return 0

    def deviation(self, Type='INT'):
        """ Returns deviation from set attenuation value """
        Num = self.get_Att_idx(Type)
        return self.com('ATT' + repr(Num+1) + ':DEV')
    
    def list_attenuators(self):
        return self.com('ATT:ENUM').strip('"')
        
    def attenuation(self, Att=None, Type='INT'):
        '''function attenuation([Att=0],[Type='INT']):

        Type can be internal (DEF) or 'EXT1'-4

        Examples:

        da.attenuation(10) will set 10db attenuation to the internal attenuator
        da.attenuation(50,3) will set 50dB of attenuation to the third external
        attenuator
        da.attenuation(50,'EXT3') same command as before

        NOTE: the 0 level is actually around -3 db
        '''

        # Get attenuator number
        Num = self.get_Att_idx(Type)

        # Check for query
        if Att is None:
            return self.com('ATT'+repr(Num+1)+':ATT')

        # Set specified value
        else:
            Att = abs(Att)
            if Att > self.__ATT_MAX:
                print('ERROR: max attenuation is '+repr(self.__ATT_MAX))
                return 0
            self.com('ATT'+repr(Num+1)+':ATT',Att)

    def set_offset(self, Offset=0, Type="INT"):
        """ function set_offset([Offset=0],[Type='INT'])
        This function is used to set an offset to the instrument, useful for
        calibration.
            - Offset is set to zero as default
            - Type can be internal (DEF) or EXT1-4
        """
        try:
            if type(Type) == str:
                Num = self.__ATT_LIST.index(Type.upper())
            else:
                Num = abs(int(Type))
                if Num > 4:
                    print('Wrong number inserted')
                    return 0
        except:
            print('Wrong type inserted')
            return 0
        self.com('ATT'+repr(Num+1)+':UCAL:OFF', abs(Offset))
