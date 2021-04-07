# -*- coding: utf-8 -*-
"""Data fitting plugin for the datamodule.

Author: Christian Schneider <c.schneider@uibk.ac.at>
Date: 13.08.2019
"""

# Essentials
import numpy as np
from scipy.signal import find_peaks
from lmfit import Model
from IPython.display import display, Math, Latex
import pandas as pd

# Plotting
import bokeh.plotting as bp


def lorentz_function(x, x0, gamma, offset, amplitude):
    """Lorentz function
    
    See section about physics lorentzian curve on wikipedia:
    https://www.wikiwand.com/en/Cauchy_distribution
    
    gamma is FWHM in our implementation.
    
    .. math::
        y(x) = amplitude \cdot (\Gamma/2)^2/((x-x0)^2 + (\Gamma/2)^2 ) + 
        \mathrm{offset}

    """
    return amplitude * (gamma / 2) ** 2 / (
                (x - x0) ** 2 + (gamma / 2) ** 2) + offset

def exp_func(x, tau, amp, vert_offset):
    return amp * np.exp(-(x) / tau) + vert_offset


def T2_func(x, tau, f, phase, amp, vert_offset):
    return (amp * np.cos(2 * np.pi * f * (x) + phase) *
            np.exp(-(x) / tau)) + vert_offset

class fit_plugin(object):

    def __init__(self, data):
        """

        Parameters
        -----------
        data : datamodule
            Datamodule with stored data
        """
        self._data = data

    def _fit_routine(self, gmodel, params, plot=True, print_results=True,
                    plot_init=False):
        """Fit and plot/print results"""

        try:
            result = gmodel.fit(self._data.y, params, x=self._data.x)

        except Exception as e:
            p = bp.figure(plot_width=800, plot_height=400)

            # add both a line and circles on the same plot
            p.circle(self._data.x, self._data.y, size=4)
            p.line(self._data.x, gmodel.eval(params, x=self._data.x),
                   color='green', line_width=2)
            bp.show(p)
            raise Exception('Fit did not converge. Try tuning guess parameters:', e)

        # Plot result ##########################################################
        if plot:
            p = bp.figure(plot_width=800, plot_height=400)

            # add both a line and circles on the same plot
            p.circle(self._data.x, self._data.y, size=4)
            p.line(self._data.x, result.best_fit, color='firebrick',
                   line_width=2)
            if plot_init:
                p.line(self._data.x, result.init_fit, color='green',
                       line_width=2)
            bp.show(p)

        # Parameters to give back ##############################################
        r_sq = 1 - result.residual.var() / np.var(self._data.y)

        # Print results
        if print_results:
            display(result)
            display(Math(r'R^2 = {:.3f}'.format(r_sq)))

        # Save fitresults to mfit module
        self._data._fit_executed = 'mfit'
        self.model = gmodel
        self.result = result
        self.r_squared = r_sq

        # Save fitresult to datamodule
        f_dict = {
            'Value': [result.params[k].value for k in result.params.keys()],
            'Error': [result.params[k].stderr for k in result.params.keys()]}
        self._data.fitresults = pd.DataFrame(data=f_dict,
                                            index=list(result.params.keys()),
                                            columns=['Value', 'Error'])

    def custom(self, function, par_dict, bounds=None,
               plot=True, print_results=True, plot_init=False):
        """Fit for arbitray functions. You have to define the function yourself
        and add it inside here, furthermore you have to give a dictionary of 
        initial values.

        """
        # Create LM Fit model ##################################################
        gmodel = Model(function)

        # Init pars to guess values ############################################
        params = gmodel.make_params(**par_dict)

        # Create bounds if given ###############################################
        if bounds is not None:
            for k in bounds:
                params[k].min = bounds[k][0]
                params[k].max = bounds[k][1]

        # Fitting ##############################################################
        self._fit_routine(gmodel, params, plot, print_results, plot_init)


    # Qubit fit functions ######################################################
    def T1(self, tau0=None, amp0=None, vert_offset0=None,
           plot=True, print_results=True, plot_init=False):
        """Fit exponential decay to data

        Parameters
        -----------
        tau0 : int, float
            Decay time in units of x axis
        amp0 : float
            Amplitude in units of y axis
        vert_offset0 : float
            Y Offset in y units
        plot : bool
            Plot fit after successful fitting
        print_results : bool
            Print results after successful fitting
        plot_init : bool
            Plot initial guess for parameters
        """
        # Guess values #########################################################
        # Vertical offset
        if vert_offset0 is None:
            vert_offset_guess = np.nanmean(self._data.y)
        else:
            vert_offset_guess = vert_offset0

        # Amplitude
        if amp0 is None:
            amp_guess = self._data.y[0] - self._data.y[-1]
        else:
            amp_guess = amp0

        # Decay time
        if tau0 is None:
            tau_guess = (np.abs(self._data.x[-1] - self._data.x[0])) / 10
        else:
            tau_guess = tau0

        # Create LM Fit model ##################################################
        gmodel = Model(exp_func)

        # Init pars to guess values ############################################
        params = gmodel.make_params(tau=tau_guess,
                                    amp=amp_guess,
                                    vert_offset=vert_offset_guess)

        # Fitting ##############################################################
        self._fit_routine(gmodel, params, plot, print_results, plot_init)

    def T2(self, tau0=None, f0=None, phase0=None, amp0=None, vert_offset0=None,
           plot=True, print_results=True, plot_init=False):
        """Fit T2 decay to data

        Function used for fitting
        math:`amp \cos{2\pi f (x) + \phase} \exp{-(
        x)/\tau) + vert_offset`


        """
        # Guess values #########################################################
        # Vertical offset
        # Guess: mean value
        if vert_offset0 is None:
            vert_offset_guess = np.nanmean(self._data.y)
        else:
            vert_offset_guess = vert_offset0

        # Decay time
        # Guess: A tenth of x range
        if tau0 is None:
            tau_guess = (np.abs(self._data.x[-1] - self._data.x[0])) / 10
        else:
            tau_guess = tau0

        # -- Do fft for guessing frequency, amplitude and phase ----------------
        freqs = np.fft.fftfreq(self._data.x.size,
                               self._data.x[1] - self._data.x[0])
        fft = np.fft.fft(self._data.y)

        # Remove DC
        freqs = freqs[1:]
        fft = fft[1:]

        # Find frequencies (biggest peaks in spectrum)
        srtd = np.argsort(np.abs(fft))
        # ----------------------------------------------------------------------

        # Amplitude
        # Guess from fft
        if amp0 is None:
            amp_guess = np.abs(fft)[srtd[-1]] / fft.size + np.abs(fft)[
                srtd[-2]] / fft.size
            amp_guess *= 5  # Guessing that cos decaying over a fifth of xrange
        else:
            amp_guess = amp0

        # Phase
        # Guess from fft
        if phase0 is None:
            phase_guess = np.abs(
                np.angle(fft[srtd[-1]]) - np.angle(fft[srtd[-2]]))
        else:
            phase_guess = phase0

        # Frequency
        # Guess from fft
        if f0 is None:
            f_guess = np.abs(freqs[srtd[-1]])
        else:
            f_guess = f0

        # Create LM Fit model ##################################################
        gmodel = Model(T2_func)

        # Init pars to guess values ############################################
        params = gmodel.make_params(tau=tau_guess,
                                    f=f_guess,
                                    phase=phase_guess,
                                    amp=amp_guess,
                                    vert_offset=vert_offset_guess)
        # Set boundaries
        params['tau'].min = 0
        params['f'].min = 0
        params['amp'].min = 0

        # Fitting ##############################################################
        self._fit_routine(gmodel, params, plot, print_results, plot_init)

    def lorentzian(self, x0=None, gamma=None, amplitude=None, offset=None,
                   plot=True, print_results=True, plot_init=False):
        """Fit Lorentz function to data
        
        See section about physics lorentzian curve on wikipedia:
        https://www.wikiwand.com/en/Cauchy_distribution
        
        gamma is FWHM in our implementation.
        
        .. math::
            y(x) = amplitude \cdot (\Gamma/2)^2/((x-x0)^2 + (\Gamma/2)^2 ) + 
            \mathrm{offset}
        
        """
        # Guess values #########################################################

        # x0
        if x0 is None:
            # Guess using find_peaks
            peaks, _ = find_peaks(np.abs(self._data.y),
                                  distance=len(self._data.y))
            x0_guess = self._data.x[peaks[0]]
        else:
            x0_guess = x0

        # Ampltiude
        if amplitude is None:
            # Guess using find_peaks
            peaks, _ = find_peaks(np.abs(self._data.y),
                                  distance=len(self._data.y))
            amp_guess = self._data.y[peaks[0]]
        else:
            amp_guess = amplitude

        # Width
        if gamma is None:
            # Guess a fraction of range
            gamma_guess = np.abs(self._data.x[-1] - self._data.x[0]) / 100

        else:
            gamma_guess = gamma

        if offset is None:
            # Guess mean value
            offset_guess = np.nanmean(self._data.y)
        else:
            offset_guess = offset

        # Create LM Fit model ##################################################
        gmodel = Model(lorentz_function, nan_policy='omit')

        # Init pars to guess values ############################################
        params = gmodel.make_params(x0=x0_guess,
                                    amplitude=amp_guess,
                                    gamma=gamma_guess,
                                    offset=offset_guess)
        # Set boundaries
        params['x0'].min = 0
        params['gamma'].min = 0

        # Fitting ##############################################################
        self._fit_routine(gmodel, params, plot, print_results, plot_init)


    def lorentzian_multiple(self, n=2, x0=None, gamma=None, amplitude=None,
                            offset=None, plot=True, print_results=True, 
                            plot_init=False):
        """Fit multiple Lorentz function to data
        
        If guess values are given, give them in an array, eg. x0=[3.5, 4].

        See section about physics lorentzian curve on wikipedia:
        https://www.wikiwand.com/en/Cauchy_distribution
        
        gamma is FWHM in our implementation.
        
        .. math::
            y(x) = amplitude \cdot (\Gamma/2)^2/((x-x0)^2 + (\Gamma/2)^2 ) + 
            \mathrm{offset}
        
        """
        # Guess values #########################################################

        # x0
        if x0 is None:
            peaks, tmp = find_peaks(np.abs(self._data.y), 
                        height=np.nanmean(self._data.y)+0.05*np.nanmean(self._data.y),
                        distance=10)

            heighest = np.argsort(tmp['peak_heights'])
            heighest = heighest[::-1]

            peaks = peaks[heighest]
            x0_guess = [self._data.x[peaks[i]] for i in range(n)]
        else:
            x0_guess = x0

        # Ampltiude
        if amplitude is None:
            # Guess using find_peaks
            peaks, tmp = find_peaks(np.abs(self._data.y), 
                        height=np.nanmean(self._data.y)+0.05*np.nanmean(self._data.y),
                        distance=10)

            heighest = np.argsort(tmp['peak_heights'])
            heighest = heighest[::-1]
            peaks = peaks[heighest]
            amp_guess = [self._data.y[peaks[i]] for i in range(n)]
        else:
            amp_guess = amplitude

        # Width
        if gamma is None:
            # Guess a fraction of range
            gamma_guess = np.abs(self._data.x[-1] - self._data.x[0]) / 100 * np.ones(n)

        else:
            gamma_guess = gamma

        if offset is None:
            # Guess mean value
            offset_guess = np.nanmean(self._data.y)
        else:
            offset_guess = offset


        # Init pars to guess values ############################################
        gmodel = Model(lorentz_function, nan_policy='omit',
                       prefix='l{}_'.format(0))
        pars = gmodel.make_params(x0=x0_guess[0],
                                amplitude=amp_guess[0],
                                gamma=gamma_guess[0],
                                offset=offset_guess)
        # Set boundaries
        pars['l0_x0'].min = 0
        pars['l0_gamma'].min = 0

        for i in range(1,len(x0_guess)):
            gmodel_tmp = Model(lorentz_function, nan_policy='omit',
                               prefix='l{}_'.format(i))
            pars.update(gmodel_tmp.make_params(x0=x0_guess[i],
                                          amplitude=amp_guess[i],
                                          gamma=gamma_guess[i],
                                          offset=offset_guess))
            pars['l{}_x0'.format(i)].min = 0
            pars['l{}_gamma'.format(i)].min = 0
            gmodel = gmodel + gmodel_tmp
        
        # Fitting ##############################################################
        self._fit_routine(gmodel, pars, plot, print_results, plot_init)