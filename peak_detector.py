# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:32:56 2017

@author: Oscar
"""

import DataModule as dm
import numpy as np
import matplotlib.pyplot as plt
from UtilitiesLib import cut_pulse_auto,filter_data,progressive_plot_2d
import peakutils
#import time

#data = dm.load_datamodule('/Oscar/2017-11-06-scan-between-3.9-4.5GHz-1.6V-diff-hose.dm')

#data.plot(engine='p')

noise_level= 10

#test_x_axis = np.linspace(8,8.4,10001)

def measure_test(x):
    y = dm.lorentzian_fit(x,*[8.13,10e-3,0,1])
    #y+=np.cos(2*np.pi*2*x)*noise_level*3
    try:
        y += (2*np.random.rand(len(y))-1)*noise_level/100
    except TypeError:
        y += (2*np.random.rand()-1)*noise_level/100
    return y
"""
def thresholding_algo(y, lag, threshold, influence,plot=True):
    signals = np.zeros(len(y))
    #filteredY = np.array(y)
    avgFilter = [np.mean(y[0:lag])]*len(y)
    stdFilter = [np.std(y[0:lag])]*len(y)
    #avgFilter[lag - 1] = 
    #stdFilter[lag - 1] = 
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            #filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = (1-influence)*avgFilter[i-1] + influence* np.mean(y[(i-lag):i])
            stdFilter[i] = (1-influence)*stdFilter[i-1] + influence* np.std(y[(i-lag):i])
        else:
            signals[i] = 0
            #filteredY[i] = y[i]
            avgFilter[i] = np.mean(y[(i-lag):i])
            stdFilter[i] = np.std(y[(i-lag):i])
    
    
    avgFilter = np.array(avgFilter)
    stdFilter = np.array(stdFilter)
    
    if plot is True:
        dm.nice_plot()
        x = np.arange(0,len(y),1)
        plt.plot(x,y,'-o')    
        #plt.plot(fY,'-k')    
        plt.plot(x-1,avgFilter+stdFilter*threshold,'r')
        plt.plot(x-1,avgFilter-stdFilter*threshold,'r')
        plt.plot(x-1,avgFilter,'c')    
        plt.figure()
        dm.nice_plot()
        plt.plot(signals,'o')

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))
"""  
"""  
def peak_detect(x,y,window_size,n_std=3,order=2,plot=True):
    window_size = int(window_size)
    dy = filter_data( deriv(y),window_size,order)
    dy=dy[window_size:-window_size]
    new_x =x[window_size:-window_size]
    
    avg = np.mean(dy[:window_size])
    std = np.std(dy[:window_size])*n_std
    
    width = new_x[dy.argmax()]-new_x[dy.argmin()]
    if width >0:
        direction = -1
    else:
        direction = 1
    
    center = (new_x[dy.argmax()]+new_x[dy.argmin()])/2
    #index = int(abs(dy.argmax()+dy.argmin())/2)
    if plot is True:
        plt.figure()
        plt.plot(x,y)
        plt.figure()
        plt.plot(new_x,dy)
        plt.vlines(center,dy.min(),dy.max(),'r',linewidth=4)
        plt.hlines(avg,new_x[0],new_x[-1],'c')
        plt.hlines(std+avg,new_x[0],new_x[-1],'k')
        plt.hlines(-std+avg,new_x[0],new_x[-1],'k')
    
    if dy.max() > avg+std or dy.min() < avg-std:
        return (center,abs(width),direction)
    else:
        return (None,None,0)
"""

def peak_detect(x,y,window_size=0,threshold=None,plot=True):
    """
    
    This function will try to find peaks in the data:
    
    Params:
        - x and y are the data axis
        - windows_size is used for filtering, if it is 0 (def), no filtering 
          will be used
        - threshold is normalized ]0,1], if it is None it is automatically
          evaluated (def)
        - plot will plot the results if True (def)
        
    returns:
        a list cointaining the number of peaks, each element is a tuple of:
            (center,width, amplitude)
        
        positive amplitude means upward peak
        

    NOTES: 
        1)the begin of the data must be a bit far of the peak, the peak 
        needs "wings"
        2) recognition of a mix of upward and downward peaks is not implemented
    
    
    """
    
    try:
        iter(x)
    except TypeError:
        print('x must be an iterable')
    
    try:
        iter(y)
    except TypeError:
        print('y must be an iterable')

    
        
    baseline = y[0]
    
    #find direction beacuse the alg works only with upwards peaks
    temp1 = y.max()-baseline
    temp2 = y.min()-baseline

    if window_size == 0:    
        temp = y.copy()
    elif window_size >0:
        if window_size %2 ==0:
            window_size+=1
        temp = filter_data(y,window_size,2)
    else:
        print('Windows size cannot be negative\n')
        raise Exception(ValueError)
        
    if np.max([abs(temp1),abs(temp2)]) == abs(temp2): #downward, must be reverted
        temp *= -1
    del temp1,temp2
    
    #Let's find peak centers indexes in the y array/list
    
    if threshold is None:
        threshold = abs(np.std(temp)*3)/abs(temp.max()-temp.min())
        
    #return threshold    
    indexes=peakutils.indexes(temp,threshold) #indexes is a list that contains the indexes
    
    if len(indexes)==0: #no peaks found
        return []
    
    
    #retrieving amplitudes:
    amplitudes = y[indexes]-baseline

    #now let's exteem the widths
    left_sides = np.ndarray(len(indexes))

    count=-1
    for i,ind in enumerate(indexes):
        half_amp = baseline+amplitudes[i]*0.5
        while(1):
            temp = y[ind+count]
            if (amplitudes[i] <0 and temp >half_amp) or (amplitudes[i] >0 and temp <half_amp):
                left_sides[i]= x[ind+count]
                break
            count-=1
    
    right_sides = np.ndarray(len(indexes))
    count=1
    for i,ind in enumerate(indexes):
        while(1):
            temp = y[ind+count]
            if (amplitudes[i] <0 and temp >half_amp) or (amplitudes[i] >0 and temp <half_amp):
                right_sides[i]= x[ind+count]
                break
            count+=1
        
    widths = right_sides-left_sides
    
    if plot is True:
        plt.figure(figsize=(16,9))
        plt.plot(x,y,'-b')
        plt.plot(x[indexes],y[indexes],'ro')
    
    final = []
    for i in range(len(indexes)):
        final.append( (x[indexes[i]],widths[i],amplitudes[i]) ) #center,width,amplitude
    return final

    

def deriv(a):
    # First derivative of vector using 2-point central difference.
    #  T. C. O'Haver, 1988.
    a = np.array(a)
    n=len(a)-1
    d=np.zeros(len(a))
    d[0]=a[1]-a[0]
    d[n]=a[n]-a[n-1]
    for j in range(1,n):
        d[j]= (a[j+1]-a[j-1]) / 2
    
    return d


def data_sort(x,y):
    indexes = np.argsort(x)
    x=np.array(x)
    y=np.array(y)
    x,y = x[indexes],y[indexes]        
    return x.tolist(),y.tolist()
    

def slope_points_distribution(width,amp,points):
    if points %2 == 0:
        points+=1
        
    cos_theta = np.cos(np.arctan(np.abs(amp/width)*2))
    
    # now I have y = m*x and I want to distribute points along y first:
        
    y1 = np.linspace(-amp,amp,points )
    #y2 = np.linspace(y1[-1],0,int(points/2) )
    
    #then project on x and make it symmetric, adding one point in the middle (the peak should be here)
    x1 = y1*cos_theta
    
    
    
    return x1

def peak_scan_auto(start,step,stop,function,avg_window=6,n_std=1.5,points_per_peak = 21,auto_break=True):
    '''
    given a y=function(x), this algorithm changes the value of x according to 
    start (GHz),step (MHz),stop (GHz) and tries to detect a peak. If a peak is found the function 
    will evaluate his width and it will add points_per_peak points to it. 
    
    If auto_break is True, the measurement will stop after finding and refining
    a peak.
    
    notes on avg_windows:
        - avg_window*step shouldn't be much larger than the peak
        - avg_window must be odd because of the filter function
        - the measurement will be at least 2*avg_window long
    
    '''
    
    if points_per_peak < 2:
        print('Too few points_per_peak specified (>1)')
        raise Exception('FewPoints')
    if points_per_peak %2 == 0:
        points_per_peak+=1
    
    step = step/1e3 # from MHz to GHz
    #peak_types = ['p','n','b']
    avg_window = int(avg_window)
    if avg_window%2 ==0:
        avg_window +=1
    
    def scan(x,y,x_now):
        y.append(function(x_now))
        x.append(x_now)
        
        
        p = peak_detect(x,y,avg_window,n_std,plot=False)
        if p[-1]==0:
            return False,p
        
        else:
            return True,p
    
    def refine_scan(x,y,p,step):
        x_now = p[0]+step
        y.append(function(x_now))
        x.append(x_now)
        
        x,y = data_sort(x,y)
        
        p = peak_detect(x,y,avg_window,n_std,plot=False)
        
        return x,y,p
    
    def measure_N_points(x,y,x_now,N):
        for count in range(N):
            y.append(function(x_now))
            x.append(x_now)
            x_now+=step
        
        return x,y,x_now
    
    def interactive_plot(x,y,style='-o'):
        progressive_plot_2d(x,y,style)
        plt.title(len(y))
    
    #Code starts here
    try:
        
        x=[]
        y=[]
        p=0
        #Measure first a number of points = avg_window
        x,y,x_now=measure_N_points(x,y,start,avg_window*2)
        
        
        loop_step=1
        added_points = False
        while(x_now < stop):
            if loop_step == 1:
                detected,p = scan(x,y,x_now)
                
                if detected is True and added_points is False:
                    print('detected')
                    x,y,x_now = measure_N_points(x,y,x_now,9)
                    added_points,prev_peak = True,p[0]
                    
                if added_points is True and detected is True:
                    if abs(p[0]-prev_peak) < step:
                        loop_step=2
                    else:
                        added_points=False
                if detected is False:
                    added_points=False
                    x_now += step
                        
        
            if loop_step == 2:
          
                #new_step = np.min([p[1]*4,step/2])
                steps = slope_points_distribution(p[1]*2,np.max(y)-np.min(y),points_per_peak)
                count=0
                while(count<points_per_peak):
                    x,y,p=refine_scan(x,y,p,steps[count])
                    print('refining')
                    if p[-1] != 0:
                        if abs(p[0]-prev_peak) < step:
                            loop_step=1
                            break
                        #new_step/=2
                        count+=1
                        loop_step=3
                    else:
                        loop_step=1
                        break
            
            interactive_plot(x,y,'o')
            
            
            if loop_step==3 and auto_break is True:
                break
                  
        #print(loop_step)
        
    except KeyboardInterrupt:
        return np.array(x),np.array(y),np.round_(p,9)
    
    return np.array( x),np.array( y),np.round_(p,9)


def try_lorentzian_fit(data,p,plot=True,plot_params=True):
    if p[2]>0:
        data.fit(dm.lorentzian_fit,[p[0],p[1],data.y[0],p[2]],labels=dm.lorentzian_fit_pars_labels,plot_init=False,plot=plot,plot_params=plot_params)
    else:
        data.fit(dm.lorentzian_fit,[p[0],p[1],data.y[0],p[2]],labels=dm.lorentzian_fit_pars_labels,plot_init=False,plot=plot,plot_params=plot_params)


"""
measured_x,measured_y,peak = peak_scan_auto(8.,4,9,6)
data = dm.data_line(measured_x,measured_y)
if peak[2]>0:
    data.fit(dm.lorentzian_fit,[peak[0],peak[1],data.y.min(),data.y.max()-data.y.min()],labels=dm.lorentzian_fit_pars_labels)
else:
    data.fit(dm.lorentzian_fit,[peak[0],peak[1],data.y.max(),data.y.min()-data.y.max()],labels=dm.lorentzian_fit_pars_labels)
data.plot(engine='p',style='o')
"""
#plt.plot(measured_x,measured_y,'-o')
"""
"""
#test_x = np.arange(8.0,8.3,0.5e-3)
#test_y = measure_test(test_x)

#p0 = peak_detect(test_x,test_y,21)

#ps = slope_points_distribution(2*p0[1],test_y.max()-test_y.min(),20)

#test_p_x = p0[0]+ps

#test_p_y = measure_test(test_p_x)

