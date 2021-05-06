'''
Created on Sep 18, 2014

@author: Seyed Iman Mirzaei
'''

#import numpy as np
import numpy as np
import scipy.optimize as opt
print('JJObject v1.0.0')

class junction():
    version='1.0.0'
    def __init__(self, Tc, T, Area, RnUnit,
                 CjUnit, Cground):
        
    #def __init__(self, Tc=1.23, T=20e-3, Area=4e-13, RnUnit=1300e-12,
    #            CjUnit=50e-3, Cground=30e-18):
        
        # Defining physical constants
        self.ec_ = 1.602176565e-19  # elementary charge (C)
        self.c_ = 299792458  # speed of light (m.s^-1)
        self.eps0_ = 8.854187817e-12  # electric constant (F.m^-1)
        self.mu0_ = 4*np.pi*1e-7  # magnetic constant (N.A^-2)
        self.h_ = 6.62606957e-34  # planck constant (J.s)
        self.hbar_ = self.h_/(2*np.pi)  # reduced planck constant (J.s)
        self.kb_ = 1.3806488e-23  # Boltzman coefficient (J.K^-1)
        self.Phi0_ = self.h_/(2*self.ec_)  # magnetic flux quanta (Wb)
        
        # Defining junction constants from parameters passed to the object
        self.Tc = Tc  # critical temperature in kelvin
        self.T = T  # junction temperature in kelvin
        self.Area = Area  # junction area (m^2)
        self.RnUnit = RnUnit  # normal state resistance of the junction per unit area (ohm)
        self.CjUnit = CjUnit  # junction capacitance per unit area (Farads)
        self.Cground = Cground
        
        # Calculating junction properties
        self.update()
        
        
    
    
    def update(self, Tc=None, T=None, Area=None, RnUnit=None, CjUnit=None, Cground=None):
        if not Tc is None :
            self.Tc = Tc
        if not T is None :
            self.T = T
        if not Area is None :
            self.Area = Area
        if not RnUnit is None :
            self.RnUnit = RnUnit
        if not CjUnit is None :
            self.CjUnit = CjUnit
        if not Cground is None :
            self.Cground = Cground
        
            
        self.Delta0 = self.calcDelta0(self.Tc)
        self.Delta = self.calcDelta(self.T, self.Tc, self.Delta0)
        self.Rn = self.calcRn(self.Area, self.RnUnit)
        self.Cj = self.calcCj(self.Area, self.CjUnit)
        self.Ic = self.calcIc(self.Rn,self.Delta,self.T)
        self.Ej = self.calcEj(self.Ic)
        self.Ec = self.calcEc(self.Cj)
        # in the following case, the junction current is assumed to be a thousand 
        # times smaller than the critical current
        self.Lj = self.calcLj_lowCurr(self.Ic, self.Ic*1e-3)  
        self.F0 = self.calcF0(self.Lj, self.Cj)
        

    def calcDelta0(self,Tc):
        ''' This function calculates the zero kelvin gap value based on BCS theory'''
        return 1.764*self.kb_*Tc  # gap value in Joules
    
    def calcDelta(self,T,Tc,d0):
        ''' This function calculates the finite-temperature gap based on a simplified equation that is more valid around 0 and Tc'''
        return d0*np.tanh((np.pi/1.764)*np.sqrt(0.95333*((Tc/T)-1)))  # gap value in Joules

    def calcRn(self,Area,RnUnit):
        return RnUnit/Area 
    
    def calcCj(self,Area,CjUnit):
        return CjUnit*Area

    def calcIc(self,Rn,Delta,T):
        '''This function calculates the junction critical current based on Ambegaokar-Baratoff relation [Ambegaokar1963]
        '''
        return (np.pi*Delta/(2*self.ec_*Rn))*np.tanh(Delta/(2*self.kb_*T))
    
    def calcEj(self,Ic):
        return self.h_*Ic/(4*self.ec_*np.pi)
    
    def calcEc(self,Cj):
        return self.ec_**2/(2*Cj)
    
    def calcLj_lowCurr(self,Ic,I):
        return self.Phi0_/(2*np.pi*Ic*np.sqrt(1-(I/Ic)**2))
    
    def calcF0(self,Lj,Cj):
        return 1/(2*np.pi*np.sqrt(Lj*Cj))
    
    def junctionInfo(self,PrintInfo=True):
        self.update()
        txt = ('\n\nJunction Information: \n' +
              '\n=================================================================' +
              '\n* F0 = ' + '%0.2f' % (self.F0/1e9) + ' GHz   Plasma frequency ' +
              '\n* Delta('+str(self.T)+'K) = ' + '%0.2f' % (self.Delta/(self.h_*1e9)) + ' GHz   Gap value'
              '\n* ic(' + str(self.T) + 'K) = ' + '%0.3f' % (self.Ic*1e-6/self.Area) + ' uA/um^2   Critical current density'
              '\n* Ic(' + str(self.T) + 'K) = ' + '%0.2f' % (self.Ic*1e9) + ' nA  Critical current'
              '\n*----------------------------------------------------------------'
              '\n* Cj = ' + '%0.3f' % (self.Cj*1e15) + ' fF'
              '\n* Lj = ' + '%0.3f' % (self.Lj*1e9) + ' nH'
              '\n* Rjn = ' + '%0.3f' % (self.Rn/1e3) + ' kOhm'
              '\n*----------------------------------------------------------------'
              '\n* Ej = ' + '%0.3f' % (self.Ej/(self.h_*1e9)) + ' GHz'
              '\n* Ec = ' + '%0.3f' % (self.Ec/(self.h_*1e9)) + ' GHz'
              '\n* Ej/Ec = ' + '%0.3f' % (self.Ej/self.Ec) +
              '\n================================================================= \n\n')
              
        if PrintInfo is True:
            print(txt)
        
        return txt
    


    
class array():
    def __init__(self, junction, Cs, N):
        self.h_ = 6.62606957e-34  # planck constant (J.s)
        self.kb_ = 1.3806488e-23  # Boltzman coefficient (J.K^-1)
        
        self.j = junction
        self.Cs = Cs        
        self.N = N
        
        self.update()
        
            
    def update(self, junction = None, N = None, Cs = None):
        
        if not junction is None :
            self.j = junction
        if not N is None :
            self.N = N 
        if not Cs is None :
            self.Cs = Cs 
            
        self.slipRate = self.calcSlipRate(self.j.Ej,self.j.Ec,self.N)
        self.unloadedModes = self.calcModesUnloaded(self.j.F0, np.arange(1,np.floor(self.N/2)), self.N, self.j.Cj, self.j.Cground)
        self.loadedModes = self.calcModesLoaded(self.j.F0, np.arange(1,np.floor(self.N/2)), self.N, self.j.Cj, self.j.Cground,self.Cs,self.j.Ec,self.j.Ej)
        self.thermalPop = self.calcThermalPop(self.slipRate, self.j.T)
    
    
    def calcSlipRate(self,Ej,Ec,N):
        return ((1/self.h_)*N*16
                *np.sqrt((Ej*Ec/np.pi))
                *(Ej/(2*Ec))**0.25
                *np.exp(-np.sqrt(8*Ej/Ec))) 
        
    def calcModesUnloaded(self,f0,n,N,Cj,C0):
        return f0*np.sqrt((1-np.cos(np.pi*n/N))/(1-np.cos(np.pi*n/N)+C0/(2*Cj)))
    
    def calcModesLoaded(self,f0,n_vec,N,C_j,C_0,C_s,E_c,E_j):
        #eq_even = lambda omega_l: -1020.0*np.sqrt(2)*C_s*omega_l*np.sqrt(E_c/(E_j*(-np.cos(np.pi*n/N) + 1)*(C_0/(2*C_j) - np.cos(np.pi*n/N) + 1))) - np.tan(np.pi*n*omega_l/(2*f0*np.sqrt((-np.cos(np.pi*n/N) + 1)/(C_0/(2*C_j) - np.cos(np.pi*n/N) + 1))))
        #eq_odd = lambda omega_l: -np.tan(np.pi*n*omega_l/(2*f0*np.sqrt((-np.cos(np.pi*n/N) + 1)/(C_0/(2*C_j) - np.cos(np.pi*n/N) + 1)))) + 0.000490196078431373*np.sqrt(2)/(C_s*omega_l*np.sqrt(E_c/(E_j*(-np.cos(np.pi*n/N) + 1)*(C_0/(2*C_j) - np.cos(np.pi*n/N) + 1))))
        
        sol = np.array([])
        eps = 1e2

        n_vec = np.int_(n_vec)
        for n in n_vec:
    
            if n%2==0: # even numbers
                #eq = lambda w_l_num: eq_even_num(nn,N_num,C0_num,Cj_num,Cs_num,w0_num,Ec_num,Ej_num,w_l_num)
                eq = lambda omega_l: -1020.0*np.sqrt(2)*C_s*omega_l*np.sqrt(E_c/(E_j*(-np.cos(np.pi*n/N) + 1)*(C_0/(2*C_j) - np.cos(np.pi*n/N) + 1))) - np.tan(np.pi*n*omega_l/(2*f0*np.sqrt((-np.cos(np.pi*n/N) + 1)/(C_0/(2*C_j) - np.cos(np.pi*n/N) + 1))))
                sol = np.append(sol,opt.ridder(eq,self.unloadedModes[n-1]*(1-1/n)+eps,self.unloadedModes[n-1]-eps))
            else:
                #eq = lambda w_l_num: eq_odd_num(nn,N_num,C0_num,Cj_num,Cs_num,w0_num,Ec_num,Ej_num,w_l_num)
                eq = lambda omega_l: -np.tan(np.pi*n*omega_l/(2*f0*np.sqrt((-np.cos(np.pi*n/N) + 1)/(C_0/(2*C_j) - np.cos(np.pi*n/N) + 1)))) + 0.000490196078431373*np.sqrt(2)/(C_s*omega_l*np.sqrt(E_c/(E_j*(-np.cos(np.pi*n/N) + 1)*(C_0/(2*C_j) - np.cos(np.pi*n/N) + 1))))
                if n==1:
                    sol = np.append(sol,opt.ridder(eq,eps,self.unloadedModes[n-1]-eps))
                else:
                    sol = np.append(sol,opt.ridder(eq,self.unloadedModes[n-1]*(1-2/n)+eps,self.unloadedModes[n-1]-eps))

        return sol
    
    
    def calcThermalPop(self,slipRate,T):
        return np.exp(-2*slipRate*self.h_/(self.kb_*T))/(1 + np.exp(-2*slipRate*self.h_/(self.kb_*T)))*100
    
    def arrayInfo(self,PrintInfo=True):
        self.update()
        txt =('\n\nArray Information: \n' +
              '\n=================================================================' +
              '\n* Cs = ' + '%0.3f' % (self.Cs*1e15) + 'fF' +
              '\n* Cj/C0 = ' + '%0.3f' % (self.j.Cj/self.j.Cground) + 
              '\n* Lj_array = ' + '%0.3f' % (1e9*self.N*self.j.Lj) + ' nH' + 
              '\n* Ej_array = ' + '%0.3f' % (1e-9*self.j.Ej/(self.h_*self.N)) + ' GHz' +
              '\n* First unloaded mode = ' + '%0.3f' % (1e-9*self.unloadedModes[0]) + ' GHz    (K=1)' + 
              '\n* First loaded mode = ' + '%0.3f' % (1e-9*self.loadedModes[0]) + ' GHz    (K=1)' +
              '\n* Phase slipRate = ' + '%0.2e' % (self.slipRate) + ' Hz' +
              '\n* Qubit thermal population = ' + '%0.2f' % (self.thermalPop) + ' %' +
              '\n================================================================= \n\n' 
              )
        if PrintInfo is True:
            print(txt)
            
        return txt
    
    
    
        
