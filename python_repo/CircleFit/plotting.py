# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:33:29 2016
plotting_utilities_module_version = '1.0.1'
@author: David Zoepfl, Christian Schneider
"""
import numpy as np
from scipy.linalg import  svd
import pandas as pd
from IPython.display import display, Markdown
try:
    import matplotlib.pyplot as plt
except NotImplementedError:
    pass
from bokeh.plotting import figure, show, gridplot, ColumnDataSource
from bokeh.models import HoverTool
from collections import OrderedDict
from DataModule.plot_style import cc
from .fit_toolbox import lorentzian_abs, tan_phase, notch_model
from .fit_toolbox import reflection_model, reflection_model_mag

# Bokeh settings
TOOLS = 'box_zoom,pan,wheel_zoom,reset,save'
TOOLTIPS = [('Frequency', '@freq{1.11111111} GHz'), ("Re", "@re"),
            ("Im", "@im"),
            ('Mag', '@mag dB'), ('Phase', '@phase°')]

def _get_tools():
    tools = ('box_zoom', 'pan', 'wheel_zoom', 'reset', 'save',
             'crosshair')
    return tools


def plot_rawdata(module, engine='b', fig=None, **kwargs):
    """Plot complex data in three subplots:

    | Im(data) over Re(data) | Mag(data) over freq | Ang(data) over freq |
    """

    if engine[0].lower() == 'b':
        # Real/Imaginary
        fig1 = plot_ReIm(module,engine=engine,fit=False, **kwargs)
        # Mag over freq
        fig2 = plot_MagFreq(module, engine=engine, fit=False, **kwargs)
        # Phase over freq
        fig3 = plot_PhaseFreq(module, engine=engine, fit=False, **kwargs)

        fig = gridplot([fig1, fig2, fig3], ncols=3, plot_width=300,
                       plot_height=300)
        show(fig)

    elif engine[0].lower() == 'p':
        if fig is None:
            plt.figure(dpi=200, figsize=(12, 3))
            plt.subplots_adjust(wspace=0.33)
        # Real/Imaginary
        plt.subplot(131)
        plot_ReIm(module, engine=engine, fit=False, **kwargs)
        # Mag over freq
        plt.subplot(132)
        plot_MagFreq(module, engine=engine, fit=False, **kwargs)
        # Phase over freq
        plt.subplot(133)
        plot_PhaseFreq(module, engine=engine, fit=False, **kwargs)


def plot_cfit(module, engine='bokeh', fig=None, **kwargs):
    """Plot data and fit.
    """
    if engine[0].lower() == 'b':
        # Real/Imaginary
        fig1 = plot_ReIm(module, engine=engine,fit=True, **kwargs)
        # Mag over freq
        fig2 = plot_MagFreq(module, engine=engine, fit=True, **kwargs)
        # Phase over freq
        fig3 = plot_PhaseFreq(module, engine=engine, fit=True, **kwargs)
        # Normalized circle
        fig4 = plot_NormCircle(module, engine=engine, fit=True)

        fig = gridplot([fig2, fig3, fig1, fig4], ncols=2, plot_height=350,
                       plot_width=350)
        show(fig)

    elif engine[0].lower() == 'p':
        if fig is None:
            plt.figure(dpi=150, figsize=(8.5, 8.5))
            plt.subplots_adjust(wspace=0.35, hspace=0.35)
        # Magnitude
        plt.subplot(221)
        plt.title('Magnitude')
        plot_MagFreq(module, engine=engine, **kwargs)
        # Phase over freq
        plt.subplot(222)
        plt.title('Phase')
        plot_PhaseFreq(module, engine=engine, **kwargs)
        # Re/Im
        plt.subplot(223)
        plt.title('Re/Im')
        plot_ReIm(module, engine=engine, **kwargs)
        # Normalized circle
        plt.subplot(224)
        plt.title('Normalized Circle')
        plot_NormCircle(module, engine=engine, **kwargs)


def plot_ReIm(module, engine='b', fit=True, **kwargs):
    """Plots imaginary over real part of S parameters

    Parameters
    -----------
    module : data_cplx
        Complex datamodule
    engine : 'b', 'p'
        Engine for plotting. Chose between bokeh and pyplot
    fit : bool
        Plot fit if available
    """

    # Easy access variable names
    freq = module.x
    if module.circuit is not None and fit:
        # Plot data without electrical delay corrected
        data = module.value_raw
    else:
        data = module.value

    # Bokeh
    if engine[0].lower() == 'b':

        # Data
        source_data = ColumnDataSource(
            data=dict(
                freq=freq,
                re=data.real,
                im=data.imag,
                mag=20 * np.log10(np.abs(data)),
                phase=np.unwrap(np.angle(data), 0.5)* 180/np.pi
        ))
        # Figure
        fig = figure(tools=TOOLS)
        fig.xaxis.axis_label = 'Re'
        fig.yaxis.axis_label = 'Im'
        c1 = fig.circle('re', 'im', source=source_data, size=3)
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c1])
        fig.add_tools(hover)
        fig.select(dict(type=HoverTool)).tooltips = OrderedDict(TOOLTIPS)
        # Fit
        if fit and module.circuit is not None:
            # Correct electrical delay and calculate mag and phase values
            fit_tmp = (module.circuit.value_calc * np.exp(2j * np.pi * freq *
                       module.circuit.delay))
            mag_fit = 20 * np.log10(np.abs(fit_tmp))  # Power Mag in dB
            phase_fit = np.unwrap(np.angle(fit_tmp), 0.5) * 180 / np.pi  # Phase
            source_fit = ColumnDataSource(
                data=dict(
                    freq=freq,
                    re=module.circuit.value_calc.real,
                    im=module.circuit.value_calc.imag,
                    phase=phase_fit,
                    mag=mag_fit,
            ))
            fig.line('re', 'im', source=source_fit, line_width=2,
                     color='firebrick')
        # Return figure
        return fig

    elif engine[0].lower() == 'p':
        # Plot
        plt.plot(data.real, data.imag, '.')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.grid()
        plt.locator_params(axis='x', nbins=4)   # Reduce x ticks
        plt.locator_params(axis='y', nbins=4)   # Reduce x ticks
        # Fit
        if fit and module.circuit is not None:
            # Correct electrical delay and calculate mag and phase values
            fit_tmp = (module.circuit.value_calc * np.exp(2j * np.pi * freq *
                       module.circuit.delay))
            mag_fit = 20 * np.log10(np.abs(fit_tmp))  # Power Mag in dB
            phase_fit = np.unwrap(np.angle(fit_tmp), 0.5) * 180 / np.pi  # Phase
            plt.plot(module.circuit.value_calc.real,
                     module.circuit.value_calc.imag,
                     color=cc['r'],
                     linewidth=1.3)
        return None


def plot_MagFreq(module, engine='b', fit=True, **kwargs):
    """Plot Magnitude over frequency"""

    # Easy access variable names
    freq = module.x
    data = module.value

    # Bokeh
    if engine[0].lower() == 'b':
        source_data = ColumnDataSource(
            data=dict(
                freq=freq,
                re=data.real,
                im=data.imag,
                mag=20 * np.log10(np.abs(data)),
                phase=np.unwrap(np.angle(data), 0.5)* 180/np.pi,
            ))

        # Figure
        fig = figure(tools=TOOLS)
        fig.xaxis.axis_label = 'Frequency (GHz)'
        fig.yaxis.axis_label = 'Magnitude (dB)'
        c2 = fig.line('freq', 'mag', source=source_data, line_width=2)
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c2])
        fig.add_tools(hover)
        fig.select(dict(type=HoverTool)).tooltips = OrderedDict(TOOLTIPS)
        # Fit
        if fit and module.circuit is not None:
            # Correct electrical delay and calculate mag and phase values
            fit_tmp = (module.circuit.value_calc * np.exp(2j * np.pi * freq *
                                                          module.circuit.delay))
            mag_fit = 20 * np.log10(np.abs(fit_tmp))  # Power Mag in dB
            phase_fit = np.unwrap(np.angle(fit_tmp), 0.5) * 180 / np.pi  # Phase
            source_fit = ColumnDataSource(
                data=dict(
                    freq=freq,
                    re=module.circuit.value_calc.real,
                    im=module.circuit.value_calc.imag,
                    phase=phase_fit,
                    mag=mag_fit,
                ))
            fig.line('freq', 'mag', source=source_fit, line_width=2,
                     line_dash=[5, 5], color='firebrick')
        # Return figure
        return fig

    # Pyplot
    elif engine[0].lower() == 'p':
        plt.plot(freq, 20 * np.log10(np.abs(data)))
        plt.ylabel('Magnitude (dB)')
        plt.xlabel('Frequency (GHz)')
        plt.grid()
        plt.xlim([freq[0], freq[-1]])
        #plt.locator_params(axis='x', nbins=4)  # Reduce x ticks
        if fit and module.circuit is not None:
            # Correct electrical delay and calculate mag and phase values
            fit_tmp = (module.circuit.value_calc * np.exp(2j * np.pi * freq *
                       module.circuit.delay))
            mag_fit = 20 * np.log10(np.abs(fit_tmp))  # Power Mag in dB
            phase_fit = np.unwrap(np.angle(fit_tmp), 0.5) * 180 / np.pi  # Phase
            plt.plot(freq,
                     mag_fit,'--',
                     color=cc['r'],
                     linewidth=2)
        return None


def plot_PhaseFreq(module, engine='b', fit=True, **kwargs):
    """Plot Magnitude over frequency"""

    # Easy access variable names
    freq = module.x
    data = module.value

    # Bokeh
    if engine[0].lower() == 'b':
        source_data = ColumnDataSource(
            data=dict(
                freq=freq,
                re=data.real,
                im=data.imag,
                mag=20 * np.log10(np.abs(data)),
                phase=np.unwrap(np.angle(data), 0.5)* 180/np.pi,
            ))

        # Figure
        fig = figure(tools=TOOLS)
        fig.xaxis.axis_label = 'Frequency (GHz)'
        fig.yaxis.axis_label = 'Phase (deg)'
        c3 = fig.line('freq', 'phase', source=source_data, line_width=2)
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c3])
        fig.add_tools(hover)
        fig.select(dict(type=HoverTool)).tooltips = OrderedDict(TOOLTIPS)
        # Fit
        if fit and module.circuit is not None:
            # Correct electrical delay and calculate mag and phase values
            fit_tmp = (module.circuit.value_calc * np.exp(2j * np.pi * freq *
                                                          module.circuit.delay))
            mag_fit = 20 * np.log10(np.abs(fit_tmp))  # Power Mag in dB
            phase_fit = np.unwrap(np.angle(fit_tmp), 0.5) * 180 / np.pi  # Phase
            source_fit = ColumnDataSource(
                data=dict(
                    freq=freq,
                    re=module.circuit.value_calc.real,
                    im=module.circuit.value_calc.imag,
                    phase=phase_fit,
                    mag=mag_fit,
                ))
            fig.line('freq', 'phase', source=source_fit, line_width=2,
                     line_dash=[5, 5], color='firebrick')
        # Return figure
        return fig

    # Pyplot
    elif engine[0].lower() == 'p':
        plt.plot(freq, np.unwrap(np.angle(data), 0.5) * 180/np.pi)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Phase (deg)')
        plt.grid()
        plt.xlim([freq[0], freq[-1]])
        if fit and module.circuit is not None:
            # Correct electrical delay and calculate mag and phase values
            fit_tmp = (module.circuit.value_calc * np.exp(2j * np.pi * freq *
                       module.circuit.delay))
            mag_fit = 20 * np.log10(np.abs(fit_tmp))  # Power Mag in dB
            phase_fit = np.unwrap(np.angle(fit_tmp), 0.5) * 180 / np.pi  # Phase
            plt.plot(freq,
                     phase_fit,'--',
                     color=cc['r'],
                     linewidth=2)
        return None


def plot_NormCircle(module, engine='b', fit=True, **kwargs):
    """Plots normalized circle

    Parameters
    -----------
    module : data_cplx
        Complex datamodule
    engine : 'b', 'p'
        Engine for plotting. Chose between bokeh and pyplot
    fit : bool
        Plot fit if available
    """

    # Easy access variable names
    freq = module.x
    data = module.value

    # Plot parameters
    if module.circuit.type == 'Notch':
        x_low = -0.1
        x_high = 1.9
        y_low = -1
        y_high = 1
    else:  # Reflection
        x_low = -1.2
        x_high = 1.2
        y_low = -1.2
        y_high = 1.2

    # Bokeh
    if engine[0].lower() == 'b':

        # Normalized data
        source_norm = ColumnDataSource(
            data=dict(
                freq=freq,
                re=module.circuit.circle_norm.real,
                im=module.circuit.circle_norm.imag,
                mag=20 * np.log10(np.abs(module.circuit.circle_norm)),
                phase=np.unwrap(np.angle(module.circuit.circle_norm)),
            ))
        # Figure
        fig = figure(tools=TOOLS, lod_threshold=10, lod_factor=100,
                     match_aspect=True, x_range=(x_low, x_high),
                     y_range=(y_low, y_high))
        fig.xaxis.axis_label = 'Re'
        fig.yaxis.axis_label = 'Im'
        c4 = fig.circle('re', 'im', source=source_norm, size=3)
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c4])
        fig.add_tools(hover)
        fig.select(dict(type=HoverTool)).tooltips = OrderedDict(TOOLTIPS)
        # Fit
        if fit:

            # Choose model to fit
            fit_freq = np.linspace(freq[0], freq[-1], 8001)
            if module.circuit.type == 'Notch':
                if module.circuit.finalfit == 'full':
                    d = notch_model(fit_freq, *module.circuit._cir_fit_pars[0])
                else:
                    Ql, absQc, fr = module.circuit._cir_fit_pars[0]
                    phi0 = module.circuit.phi0
                    d = notch_model(fit_freq, Ql, absQc, fr, phi0)

            else:
                if module.circuit.finalfit == 'full':
                    d = reflection_model(fit_freq,
                                         *module.circuit._cir_fit_pars[0])
                else:
                    Ql, absQc, fr, phi0, _ = module.circuit._cir_fit_pars[0]
                    d = reflection_model(fit_freq, Ql, absQc, fr, phi0)

            # Plot fit
            fig.line(d.real, d.imag,
                     legend='Circle Fit', color='firebrick')
        # Return figure
        return fig

    elif engine[0].lower() == 'p':
        # Plot
        plt.plot(module.circuit.circle_norm.real,
                 module.circuit.circle_norm.imag,
                 '.', zorder=0, color=cc['b'])
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.axis('equal')
        plt.grid()
        plt.locator_params(axis='x', nbins=4)   # Reduce x ticks
        plt.locator_params(axis='y', nbins=4)   # Reduce x ticks
        # Fit
        if fit:
            # Mark offresonant point
            if module.circuit.type == 'Notch':
                offres = 1
            else:
                offres = -1
            plt.plot(offres, 0, "o", color=cc['r'])

            # Fit
            xc1 = module.circuit.fitresults_full_model.Value.xc
            yc1 = module.circuit.fitresults_full_model.Value.yc
            r = module.circuit.fitresults_full_model.Value.r
            plt.gca().add_artist(plt.Circle((xc1, yc1), r, color=cc['r'],
                                            fill=False, linewidth=2))
        return None


def plot_delay(circuit):
    display(Markdown('## 1. Electric Delay'))
    if circuit._delayOffset is None:
        print('Delay set manually. Offset value unknown. Therefore the line '
              'could be running just parallel')
        fig = figure(width=800, height=400, title='Electric Delay fit')
        phase = np.unwrap(np.angle(circuit.value_raw)) * 180 / np.pi
        fig.circle(circuit.freq, phase, legend='Data')
        # Plot fit range
        idx = int(circuit.fit_range*len(circuit.freq))
        fig.circle(circuit.freq[:idx], phase[:idx], color='green')
        fig.circle(circuit.freq[-idx:-1], phase[-idx:-1], color='green')
        # Plot fit
        fig.line(circuit.freq,
                 circuit.delay * circuit.freq * 360,
                 color='firebrick', line_dash=[5, 5], legend='Linear Fit',
                 line_width=2)
        show(fig)
    else:
        fig = figure(width=800, height=400, title='Electric Delay fit')
        phase = np.unwrap(np.angle(circuit.value_raw)) * 180 / np.pi
        fig.circle(circuit.freq, phase, legend='Data')
        # Plot fit range
        idx = int(circuit.fit_range*len(circuit.freq))
        fig.circle(circuit.freq[:idx], phase[:idx], color='green')
        fig.circle(circuit.freq[-idx:-1], phase[-idx:-1], color='green')
        # Plot fit
        fig.line(circuit.freq, circuit._delayOffset -
                 circuit.delay * circuit.freq * 360,
                 color='firebrick', line_dash=[5, 5], legend='Linear Fit',
                 line_width=2)
        show(fig)


def plot_linear_slope(circuit):
    display(Markdown('## 2. Linear slope in magnitude signal\n'))
    try:
        fig = figure(width=800, height=400, title='Linear fit in dB ' +
                                                  'magnitude')
        fig.circle(circuit.freq, 20 * np.log10(np.abs(circuit.value_raw)),
                   legend='Data')
         # Plot fit range
        idx = int(circuit.fit_range*len(circuit.freq))
        fig.circle(circuit.freq[:idx],  20 * np.log10(np.abs(
                   circuit.value_raw[:idx])), color='green')
        fig.circle(circuit.freq[-idx:-1],  20 * np.log10(np.abs(
                   circuit.value_raw[-idx:-1])), color='green')
        # Plot fit
        fig.line(circuit.freq, circuit._bg_pars[-1] +
                 circuit._bg_pars[-2] * circuit.freq, color='firebrick',
                 line_dash=[5, 5], legend='Lorentzian Fit', line_width=2)
        show(fig)
    except AttributeError:
        print('No background subtraction\n')


def plot_lorentzian(circuit):
    display(Markdown('## 3. Lorentzian fit\n'))
    fig = figure(width=800, height=400, title='Lorentzian fit')
    fig.circle(circuit.freq, np.abs(circuit.value),
               legend='Data')
    fig.line(circuit.freq, lorentzian_abs(circuit.freq,
                                          *circuit._lor_pars[0]),
             legend='Lorentzian Fit', color='firebrick', line_dash=[5, 5],
             line_width=2)
    show(fig)
    errs = np.sqrt(np.diag(circuit._lor_pars[1]))
    df = pd.DataFrame({'Value': circuit._lor_pars[0], 'Error': errs},
                      index=['Offset', 'Amplitude', 'Ql', 'fr'],
                      columns=['Value', 'Error'])
    display(df)


def plot_circle_fit_I(circuit):
    display(Markdown('## 4. Circle fit to obtain offrespoint, a and ' +
                     'alpha\n'))
    fig = figure(width=400, height=400, title='Circle Fit I')
    fig.circle(circuit.value.real, circuit.value.imag, legend='Data',
               size=3*circuit._weights)
    x_o = circuit.fitresults_full_model.Value.x_offres
    y_o = circuit.fitresults_full_model.Value.y_offres
    offrespoint = np.complex(x_o, y_o)
    xc, yc, r = circuit._circle_pars1
    p = np.linspace(0, 2 * np.pi, 100)
    circle = np.complex(xc, yc) + r * np.exp(1j * p)
    fig.circle(offrespoint.real, offrespoint.imag, color='firebrick',
               size=10, legend='Offrespoint')
    fig.line(circle.real, circle.imag, color='firebrick', line_width=2,
             legend='Circle Fit')
    show(fig)


def plot_phase_fit(circuit):
    xc, yc, r = circuit._circle_pars1
    p = np.linspace(0, 2 * np.pi, 100)
    display(Markdown('## 5. Phase fit for theta0'))
    data = (circuit.value - np.complex(xc, yc))
    fig = figure(width=800, height=400, title='Lorentzian fit')
    fig.circle(circuit.freq, np.unwrap(np.angle(data)),
               legend='Data')
    fig.line(circuit.freq, tan_phase(circuit.freq,
                                     *circuit._theta_pars[0]),
             legend='Tan Fit', color='firebrick', line_dash=[5, 5],
             line_width=2)
    show(fig)
    errs = np.sqrt(np.diag(circuit._theta_pars[1]))
    df = pd.DataFrame({'Value': circuit._theta_pars[0], 'Error': errs},
                      index=['Theta0', 'Ql', 'fr', 'compression'],
                      columns=['Value', 'Error'])
    display(df)


def plot_final_circle(circuit):
    display(Markdown('## 6. Final Circle fit'))
    fig = figure(width=400, height=400, title='Circle fit (final)' +
                                              'for Ql, absQc and fr')
    fig.circle(circuit.circle_norm.real, circuit.circle_norm.imag,
               legend='Data', size=3*circuit._weights)
    if circuit.type == 'Notch':
        d = notch_model(circuit.freq, *circuit._cir_fit_pars[0])
        fig.line(d.real, d.imag,
                 legend='Cirlce Fit', color='firebrick',
                 line_width=2)
    else:
        d = reflection_model(circuit.freq, *circuit._cir_fit_pars[0])
        fig.circle(d.real, d.imag,
                 legend='Circle Fit', color='firebrick',
                 )
    show(fig)

    # Get errors. Note that you cannot trust these if the values are close to
    # the bounds. See https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739
    # Do Moore-Penrose inverse discarding zero singular values.
    res = circuit._cir_fit_pars[1]
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    s_sq = cost / (circuit.freq.size - 4)  # 4 fit parameters
    pcov = pcov * s_sq

    errs = np.sqrt(np.diag(pcov))
    df = pd.DataFrame({'Value': circuit._cir_fit_pars[0], 'Error': errs},
                      index=['Ql', 'absQc', 'fr', 'Phi0'],
                      columns=['Value', 'Error'])
    display(df)
    print(res)

def plot_steps(circuit):
    """
        Plots step by step what is done during the whole fitting
        procedure.
    """
    # 1. Delay
    plot_delay(circuit)

    # 2. Linear slope in magnitude
    plot_linear_slope(circuit)

    # 3. Lorentzian
    plot_lorentzian(circuit)

    # 4. Circle Fit I
    plot_circle_fit_I(circuit)

    # 5. Phase fit
    plot_phase_fit(circuit)

    # 6. Final circle fit
    plot_final_circle(circuit)

    # 7. Plot residuals
    plot_residuals(circuit)


def plot_residuals(circuit):
    display(Markdown('## Residuals for final circuit fit'))
    fig = figure(width=800, height=400, title='Residuals')
    fig.circle(circuit.freq, np.abs(circuit.value_raw - circuit.value_calc))
    show(fig)


def plot_weights(circuit):
    fig = figure(width=800, height=400, title='Weights')
    fig.circle(circuit.freq, circuit._weights)
    show(fig)
