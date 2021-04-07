# -*- coding: utf-8 -*-
"""Basic DataModule functions.

This contains functions typically used for the datamodule, like loading and
saving.
"""

import glob
import re
import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr

from .downwards_compatibility.data_line import data_line,data_2d
from .downwards_compatibility.data_surface import data_surface,data_3d

from .data_complex import data_complex
from .data_grid import data_grid
from .data_table import data_table
from .data_IQ import data_IQ

try:
    import matplotlib.pyplot as plt
    import matplotlib
except NotImplementedError:
    pass
from .version import __version__
import bokeh.plotting as bp
from .plot_style import color_scheme
import holoviews as hv

current_types = (data_grid, data_complex, data_table)


# Function Library ############################################################
def load_datamodule(filename, upgrade=True):
    """ Load a datamodule

    Parameters
    ----------
    filename : str
        Filepath
    upgrade : bool
        Upgrade datamodule to newest version

    Returns
    -------
    DataModule
        Returns a datamodule
    """
    if filename[-3:] == '.dm':
        # Old datamodule storage (pickled)
        try:
            with open(filename, 'rb') as f:
                a = pickle.load(f)
            if upgrade is True:
                return upgrade_dm(a)
            else:
                return a

        ## This sections is for downwards compatibility ######################
        # Used for very old datamodules. Recommened to update this with a
        # to .h5 script
        except (AttributeError, NameError, ImportError) as e:
            # This is just for compatibility with old datamodule files
            import sys
            import DataModule.downwards_compatibility.data_line as data_l
            import DataModule.downwards_compatibility.data_surface as data_s
            import DataModule.downwards_compatibility.data_complex as data_c
            import DataModule.data_table as data_t
            sys.modules['DataModule.data_line'] = data_l
            sys.modules['DataModule.data_surface'] = data_s
            sys.modules['DataModule'].data_3d = data_s.data_3d
            tmp = sys.modules['DataModule.data_complex']
            sys.modules['DataModule.data_complex'] = data_c
            sys.modules['DataModule'].data_cplx = data_c.data_cplx
            sys.modules['DataModule.data_xy'] = data_t
            sys.modules['DataModule.data_xy'].data_xy = data_table

            with open(filename, 'rb') as f:
                a = pickle.load(f)

            # Old data complex class
            if isinstance(a, data_c.data_complex):
                f = a.x
                v = a.value
                b = data_complex(f, v)
                try:
                    b.idx_min = a.idx_min
                    b.idx_max = a.idx_max
                except AttributeError:
                    b.idx_min = 0
                    b.idx_max = None

            elif isinstance(a, (data_line, data_surface)):
                b = upgrade_dm(a)

            elif isinstance(a, (data_table)):
                a.name_x = a.df.keys()[0]
                a.name_y = a.df.keys()[1]
                b = upgrade_dm(a)

            # Restore new classes
            sys.modules['DataModule.data_complex'] = tmp
            del sys.modules['DataModule.data_line']
            del sys.modules['DataModule.data_surface']
            del sys.modules['DataModule.data_xy']
            # Save new datamodules
            print('\rUpgrade to datamodule V3. Saved new datamodule',
                  end=' ', flush=True)
            b.par = a.par
            b.save(filename, useDate=False)
            return b

        except NameError:
            import sys
            with open(filename, 'rb') as f:
                a = pickle.load(f)

            f = a.x
            v = a.value
            sys.modules['DataModule.data_complex'] = tmp
            data_new = data_complex(f, v)
            try:
                data_new.idx_min = a.idx_min
                data_new.idx_max = a.idx_max
            except AttributeError:
                data_new.idx_min = 0
                data_new.idx_max = None
            data_new.save(filename, useDate=False)
            return data_new

    elif filename[-5:] == 'dm.h5':
        # Load with pandas HDF reader
        with pd.HDFStore(filename, 'r') as f:
            # Determine first if complex data or data_table
            if '/dm_complex' in f.keys():
                tmp = data_complex()
                dtype = 'dm_complex'
            elif '/dm_table' in f.keys():
                tmp = data_table()
                dtype = 'dm_table'
            # Get metadata
            attrs = eval(f.get_storer(dtype).attrs['dm_metadata'])
            # Load dataframe
            tmp.df = f.get(dtype)
            # Set attributes of dm object
            for key, value in attrs.items():
                setattr(tmp, key, value)

    elif filename[-5:] == 'dm.nc':
        # Load with xarray netCDF reader
        with xr.open_dataset(filename) as ds:
            # Create empty data_grid
            tmp = data_grid([[0], [0], [[0]]])

            # Save xarray DataArray
            tmp.df = ds[list(ds.keys())[0]]

            # Set attributes of dm object
            for key, value in eval(tmp.df.attrs['dm_metadata']).items():
                setattr(tmp, key, value)

    # Return datamodule
    return tmp

def load_csv(fname, **kwargs):
    """Load

    Parameters
    -----------
    fname : str
        Filename

    Other Parameters
    ----------------
    ** kwargs: `~pandas.read_csv` properties

    Returns
    --------
    data_table : `~DataModule.data_table.data_table`
    """
    log = pd.read_csv(fname, **kwargs)
    log.rename(columns=lambda x: x.strip(), inplace=True)  # Remove whitespace
    tmp_dm = data_table([log[l] for l in log], [l for l in log])
    return tmp_dm


def plot_multiple(dm_list, label_list, figure=None, colormap=None,
                  engine="h", logx=False, logy=False, **kwargs):
    """Plot multiple Datamodules in a single bokeh/pyplot figure.

    Parameters
    -----------
    dm_list : list, array
        List of datamodules: [dm1, dm2, ...]
    label_list : list, array
        List of labels for the plot: ["Data of dm1", "Data of dm2"]
    figure : object
        Optional: Figure to plot in
    colormap : List of colors
        Optional: Bokeh colorpalette. You have to import it first from
        bokeh.palettes or specify itself. Format: ["#FFFFFF", "#01245", ...]
    engine : str
        Optional: Specify if Plot should be in bokeh or pyplot
    kwargs : keywords
        Optional: Keywords for bokeh or pyplot
    """
    # Import default colormap
    if colormap is None:
        colormap = color_scheme

    # Plot
    # Plot in pyplot
    if engine.lower()[0] == "p":
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
        for d, l, c in zip(dm_list, label_list, colormap):
            d.plot(figure=fig, legend=l, color=c, engine="p", **kwargs)
        plt.legend(label_list)
        if logx and logy:
            plt.loglog()
        elif logx:
            plt.semilogx()
        elif logy:
            plt.semilogy()

    elif engine.lower()[0] == 'h':
        # Plot with holoviews
        # Create Legend (bugfix for holoviews)
        color_points = hv.NdOverlay({label_list[i]: hv.Points([[dm_list[i].x[0],
                                                                dm_list[i].y[
                                                                    0]]],
                                                              label=str(
                                                                  label_list[
                                                                      i]))
                                    .opts(style=dict(color=colormap[i]))
                                     for i in range(len(dm_list))})
        plot = dm_list[0].plot_hv(color=colormap[0])
        for d, c in zip(dm_list[1:], colormap[1:]):
            plot *= d.plot_hv(color=c)
        plot *= color_points
        return plot

    else:
        # Plot with bokeh
        if not 'muted_alpha' in kwargs:
            # Set to invisible when not other specified 
            kwargs['muted_alpha'] = 0.0
        if figure is None:
            kws = {}
            if logx:
                kws['x_axis_type'] = 'log'
            if logy:
                kws['y_axis_type'] = 'log'
            tools = ['box_zoom', 'pan', 'wheel_zoom', 'reset',
                     'save', 'hover']
            fig = bp.figure(plot_width=800, plot_height=400,
                            toolbar_location='above',
                            sizing_mode='scale_width',
                            tools=tools,
                            **kws)
        else:
            fig = figure
        for d, l, c in zip(dm_list, label_list, colormap):
            d.plot(fig=fig, legend=l, style='-', color=c, engine=engine,
                   **kwargs)
        fig.legend.click_policy = "mute"
        fig.legend.location = "top_left"
        bp.show(fig)


def load_folder(folderpath, word_search=None, dm_pars=None,
                regexp='[\w+\-.@<>:]+(?:.dm.h5|.dm)$', sort=True, sort_par=None):
    """Loads all datamodules from a specified folder path.

    Accepts a keyword to filter files or a regular expression to obtain
    parameters from filename.

    Sorts the datamodules by the first given parameter. Either in dm_pars
    or in the regular expression. If you wish to not sort it, pass `sort` False
    as argument.

    Examples
    ---------
    >>> dm.load_folder('./fluxsweep/', word_search='NewRun')
    >>> dm.load_folder('./fluxsweep/',\
 regexp='[\w+\-.]+I_(?P<current>[\d\-]+)[\w+\-.]')

    Parameters
    ----------
    folderpath : str
        Path to the folder
    word_search : str, optional
        Only files containing this string will be added
    dm_pars : list(str), optional
        Extract this parameters out of the datamodules
        The first parameter is the sorting parameter
    regexp : str, optional
        Parameter to specify how the file name should look like. If given a
        parameter with (?P<par>) it will added to the dm_pars.
        https://docs.python.org/3.5/library/re.html
    sort : bool
        Sort the datamodules according to first parameter

    Returns
    --------
    tuple(list(DataModule), dict)
        Returns a tuple. First element is a sorted list of datamodules, the
        second argument is a dictionary with the sorted parameters
    """
    if not dm_pars:
        dm_pars = []

    # User mistake handling
    if not isinstance(dm_pars, list) and not isinstance(dm_pars, tuple):
        print('dm_pars must be a list or a tuple of strings')
        raise Exception('PARTYPE')

    # Read files
    fpath = os.path.abspath(folderpath)
    filelist = [os.path.basename(x) for x in glob.glob(fpath + '/*.dm*')]
    dm_list = []
    pars = {}
    # Check if Keyword is given
    if word_search:
        tmp = []
        for f in filelist:
            if f.find(word_search) != -1:
                tmp.append(f)
        filelist = tmp
    # Check for regular expression
    for f in filelist:
        m = re.match(re.compile(regexp), f)

        if m:
            # Check for parameters in filepath
            for p, p_val in m.groupdict().items():
                try:
                    pars[p].append(p_val)
                except KeyError:
                    dm_pars.append(p)
                    pars[p] = [p_val]

            # Append loaded datamodule
            dm_list.append(load_datamodule(fpath + '/' + f))
            # Append parameters from parameter list
            for p in dm_pars:
                # Check if not already added
                if p not in m.groupdict():
                    try:
                        par = dm_list[-1].par[p]
                    except TypeError:
                        # Probably a datamodule with segments
                        # --> Take first
                        par = dm_list[-1].par[0][p]
                    except KeyError:
                        print('File {} has no parameter {}'.format(f, p))
                        raise Exception('Parameter not found.')
                    # Try to append parameter to dict. If no key is found,
                    # create one
                    try:
                        pars[p].append(par)
                    except:
                        pars[p] = [par]
    # Sort if a parameter is given/found by the first parameter in the list
    # Extract parameters from datamodule
    dm_list = np.array(dm_list)
    if sort and len(dm_pars) > 0:
        if sort_par is None:
            order_par = dm_pars[0]

        else:
            order_par = sort_par
        pars[order_par] = np.array(pars[order_par], dtype=float)  # for sorting
        ordered_idxs = np.argsort(pars[order_par])
        # Order datamodules
        dm_list = dm_list[ordered_idxs]
        # Order parameters
        for p in dm_pars:
            pars[p] = np.array(pars[p])
            pars[p] = pars[p][ordered_idxs]

    return dm_list, pars


def load_folder_grid(folder, par_name, dtype='dB', interpolate=True, delay=None):
    """Loads a folder and creates a data_grid from it.
    If interpolate is set to True, it will interpolate all the data with the
    finest measurement resolution. This is required since xarray has problems
    with combining arrays with different x axis.

    For complex datamodules, you can specify if you want to look at magnitude
    (dtype='dB') or phase (dtype='phase').

    Supports unequal x and y axes.

    Parameters
    ------------
    folder : str
        Folder
    par_name : str
        Name of the data_module parameter used for x axis of data_grid
    dtype : 'dB', 'phase'
        Use Magnitude (dB) or Phase (phase) for complex datamodules.
    interpolate : bool
        If set to true, interpolate data and create uniform coordinate grid
    delay: None, float
        Delay for electrical delay correction. If None it will try to fit it
    """
    tmp = load_folder(folder, dm_pars=[par_name])

    if interpolate is True:
        # Obtain minimum y value, maximum y value and resolution
        y_min = 1e99
        y_max = -1e99
        res = 1e99
        for d in tmp[0]:
            tmp_y_min = np.min(d.x)
            tmp_y_max = np.max(d.x)
            tmp_res = findMinDiff(d.x)
            if tmp_y_min < y_min:
                y_min = tmp_y_min
            if y_max < tmp_y_max:
                y_max = tmp_y_max
            if tmp_res < res:
                res = tmp_res
        yvals = np.arange(y_min, y_max, res)
        zvals = []
        xvals = []

        # If complex data
        if isinstance(tmp[0][0], data_complex):
            for idx, d in enumerate(tmp[0]):
                xvals.append(tmp[1][par_name][idx])
                if dtype == 'phase':
                    d.correct_delay(delay)  # Correct for electrical delay
                    zvals.append(np.interp(yvals, d.x, d.phase.y))
                    name = d.phase.name_y
                else:
                    zvals.append(np.interp(yvals, d.x, d.dB.y))
                    name = d.dB.name_y

        # Data table --> Just first column
        elif isinstance(tmp[0][0], data_table):
            name = tmp[0][0].name_y
            for idx, d in enumerate(tmp[0]):
                xvals.append(tmp[1][par_name][idx])
                zvals.append(np.interp(yvals, d.x, d.y))

        # Create data_grid
        data = data_grid([xvals, yvals, zvals],
                         [par_name,
                          tmp[0][0].name_x,
                          name])

    elif interpolate is False:
        if isinstance(tmp[0][0], data_complex):
            if dtype == 'phase':
                data = data_grid([[tmp[1][par_name][0]],
                                  tmp[0][0].x,
                                  [tmp[0][0].phase.y]])
            else:
                data = data_grid([[tmp[1][par_name][0]],
                                  tmp[0][0].x,
                                  [tmp[0][0].dB.y]])
            for idx, d in enumerate(tmp[0][1:]):
                if dtype == 'phase':
                    tmp_data = data_grid([[tmp[1][par_name][idx + 1]],
                                          d.x,
                                          [d.phase.y]])
                else:
                    tmp_data = data_grid([[tmp[1][par_name][idx + 1]],
                                          d.x,
                                          [d.dB.y]])
                data.df = data.df.combine_first(tmp_data.df)

        elif isinstance(tmp[0][0], data_table):
            data = data_grid([[tmp[1][par_name][0]],
                              tmp[0][0].x,
                              [tmp[0][0].y]])
            for idx, d in enumerate(tmp[0][1:]):
                tmp_data = data_grid([[tmp[1][par_name][idx + 1]], d.x, [d.y]])
                data.df = data.df.combine_first(tmp_data.df)

    return data


def data_stack_x(dm_list):
    """Stack datamodules along the x axis.

    It returns a new data module that contains the same parameters of the
    first argument.

    Automatically sorts data along x and averages if there are multiple values
    for single x.

    Examples
    --------
        >>> total = data_stack_x([data1,data2,data3,...])
    """
    # Get data type
    data_type = dm_list[0]

    # List datamodules (ensure that it is list and not np array, etc...)
    datas = list(dm_list)

    # Create empty arrays
    x_new = np.array([])
    y_new = np.array([])
    add_cols = {}  # Additional cols (more than 2 in data_table)

    # Stack along x
    for count in range(len(datas)):
        # Go through the other arrays and append
        a = datas.pop(0)

        # Display error if datatypes are not the same for each element
        if not isinstance(a, type(data_type)):
            raise Exception('TYPE Mismatch: The data types are different.' +
                            ' Are you sure you are stacking datamodules of ' +
                            'same type?')

        # Stack x values
        x_new = np.hstack((x_new, a.x))

        # Data table #########################################################
        if isinstance(data_type, data_table):
            # First stack y axis
            y_new = np.hstack((y_new, a.y))

            # Check if there are additional rows
            if len(a.df.keys()) > 2:
                for k in a.df.keys():
                    if k != a.name_x and k != a.name_y:
                        try:
                            # Stack
                            add_cols[k] = np.hstack((add_cols[k], a.df[k]))
                        except KeyError:
                            # Create if first element
                            add_cols[k] = np.array(a.df[k])
        # Data complex #######################################################
        elif isinstance(data_type, data_complex):
            y_new = np.hstack((y_new, a.y))

        # TODO: Implement for data_grid ######################################
        else:
            print(a)
            raise Exception('Currently not implemented for this data type')

    # Sort frequencies #######################################################
    idxs_sorted = np.argsort(x_new)
    x_new = x_new[idxs_sorted]
    y_new = y_new[idxs_sorted]

    # Average if multiple values for same x value ############################

    # Data table
    if isinstance(data_type, data_table):

        # With additional columns
        if len(add_cols.keys()) > 0:

            # Sort additional columns
            for k in add_cols.keys():
                add_cols[k] = add_cols[k][idxs_sorted]

            labels = [data_type.name_x, data_type.name_y]
            data = [x_new, y_new]

            # Average additional columns and append them to datamodule
            for k in add_cols.keys():
                labels.append(k)
                data.append(add_cols[k])
            data = average_duplicates(data[0], *data[1:])

            # Createfirst two columns
            tmp_dm = data_table([data[0], data[1]], [labels[0], labels[1]])

            for idx, l in enumerate(labels[2:]):
                tmp_dm.add_column(data[idx], l)

            return tmp_dm

        # Just X and Y columns
        else:
            y_new = y_new[idxs_sorted]
            x_new, y_new = average_duplicates(x_new, y_new)
            return data_table([x_new, y_new],
                              [data_type.name_x, data_type.name_y])

    # Complex data
    elif isinstance(data_type, data_complex):
        x_new, y_new = average_duplicates(x_new, y_new)
        return data_complex(x_new, y_new)

    # Data3d
    else:
        print('Datatype not supported yet.')
        return None


def average_duplicates(index_array, *values):
    """Function which averages values if two values of the index_array are
    the same.

    This is very useful for averaging/binning data points.

    Parameters
    -----------
    index_array : list, np.array
        Array to unique. If there exist multiple values the function will
        averages the correspind *values
    *values : list, np.array
        Multiple arrays with the same length as index_array.
    """
    folded, indices, counts = np.unique(index_array, return_inverse=True,
                                        return_counts=True)

    output_values = []
    for v in values:
        output = np.zeros(folded.shape[0], dtype=type(v[0]))
        np.add.at(output, indices, v)
        output /= counts
        output_values.append(output)

    flatted = tuple(
        np.array(item) for item in [folded.tolist()] + output_values)
    return flatted


def average_data(*args):
    """Perform weighted average of multiple datamodules

    This function performs the weighted average of the data passed as an
    argument, the date must be given in a tuple/list together with the
    weight (1 = normal average).

    The function returns a new data module that contains the same
    parameters of the first argument, it automatically detects different
    data types.

    Parameters
    ----------
    *args
        Tuple of tuples. First element of each tuple is a datamodule, the
        second the weighting of this datamodule for averaging

    Returns
    --------
    DataModule
        DataModule of the same type for the averaged data
    Examples
    --------
        >>> average = average_data((data1,3),(data2,5))
        >>> average = average_data((data1,1),(data2,1))
    """
    if isinstance(args[0][0], data_table):
        tmp = args[0][0].copy()
        tmp_ysel = tmp.return_y()
        tmp_xsel = tmp.return_x()
        acc = np.zeros(len(tmp_ysel))
        weight = 0
        for count, a in enumerate(args):
            if not isinstance(a[0], data_table):
                print('Error, the data type are different')
                raise Exception('TYPE Mismatch')

            if not (a[0].return_x() == tmp_xsel).all():
                print('ERROR: x-axis must be the same: ' + str(count))
                raise Exception('WRONGAXIS')

            acc += a[0].return_y() * a[1]
            weight += a[1]

        tmp = data_table((tmp_xsel, acc / weight))

        return tmp
    else:
        print('Need implementation, only data_table supported')

    """
    elif isinstance(args[0][0], data_complex):
        tmp = args[0][0].copy()
        acc = np.zeros(len(tmp.return_vsel()), dtype=np.complex)
        weight = 0
        for count, a in enumerate(args):
            if not isinstance(a[0], data_complex):
                print('Error, the data type are different')
                raise Exception('TYPE Mismatch')

            if not (a[0].return_xsel() == tmp.return_xsel()).all():
                print('ERROR: x-axis must be the same: '+str(count))
                raise Exception('WRONGAXIS')

            acc += a[0].return_vsel()*a[1]
            weight += a[1]

        tmp.load_cplx_var(tmp.return_xsel(), acc/weight)

        return tmp

    elif isinstance(args[0][0], data_surface):
        tmp = args[0][0].copy()
        acc = np.zeros((len(tmp.return_xsel()), len(tmp.return_ysel())))
        weight = 0
        for count, a in enumerate(args):
            if not isinstance(a[0], data_surface):
                print('Error, the data type are different')
                raise Exception('TYPE Mismatch')

            if not (a[0].return_xsel() == tmp.return_xsel()).all():
                print('ERROR: x-axis must be the same: '+str(count))
                raise Exception('WRONGAXIS')

            if not (a[0].return_ysel() == tmp.return_ysel()).all():
                print('ERROR: y-axis must be the same: '+str(count))
                raise Exception('WRONGAXIS')

            acc += a[0].return_zsel()*a[1]
            weight += a[1]

        tmp.load_var(tmp.return_xsel(), tmp.return_ysel(),  acc/weight)

        return tmp
    else:
        print('No routing for this kind of data')
        return None
    """


def cleandat(x, y):
    """Autoclean data.

    Take a two row data matrix (x,y) and do the following:
    1. Sort data in ascending x order
    2. Find unique values of x and calculate average y
    3. Add a third row with standard deviation for multiple data
    (indication of statistical error)

    Parameters
    ----------
    x : list
        X values. Both arrays get sorted according to ascending x order
    y : list
        Y values.

    Returns
    --------
    tuple(list, list, list)
        x, y, y_errs
        Outputs the processed x and y array and gives also the std deviations
    """
    x = np.array(x)
    y = np.array(y)
    x_sort_idx = np.argsort(x)
    x_srt = x[x_sort_idx]
    y_srt = y[x_sort_idx]

    # Find unique values and average for same frequency
    folded, indices, counts = np.unique(x_srt, return_inverse=True,
                                        return_counts=True)
    # Average if multiple y values exist
    output = np.zeros(folded.shape[0])
    np.add.at(output, indices, y_srt)
    output /= counts

    # Calculate std deviations if multiple values exist
    y_err = np.zeros_like(output, dtype=np.float64)
    np.add.at(y_err, indices, (y_srt - output[indices]) ** 2)
    y_err = np.sqrt(y_err / counts)

    return folded, output, y_err


def split(vector, range_vector):
    """This function gets a list (array) of ranges (must have even members)
    and returns the data in x that fall within those ranges.

    TODO
    ----
    Do we need this function? And for what purpose?
    """
    # Convert input to row vectors
    x = np.array(vector).ravel()
    ranges = np.array(range_vector).ravel()

    # initialize variables
    idx = np.ma.zeros(len(x))

    # check if "ranges" has even elements
    if (len(range_vector) & 1):
        raise Exception('Number of range elements must be even.')
    else:
        # Making the new index and output
        for i in range(int(len(ranges) / 2)):
            idx_tmp = np.logical_and(x >= ranges[2 * i], x <= ranges[2 * i + 1])
            idx = np.logical_or(idx, idx_tmp)
        return idx, vector[idx]


def upgrade_dm(old_dm):
    """Upgrade datamodules to newest version.

    Function to convert old datamodule to current datamodule.

    Parameters
    ----------
    old_dm : DataModule
        DataModule to upgrade

    Note
    -----
     The fit will be lost

    Returns
    --------
    DataModule
        Updated datamodule.
    """
    # Check and initialize right data module type
    try:
        if hasattr(old_dm, '__version__'):
            vers = old_dm.__version__
        else:
            vers = old_dm.version
        vers_f = [int(x) for x in vers.split('.')]
    except AttributeError:
        # Old datamodule had no version number --> assign 0.0.0
        vers = '0.0.0'
        vers_f = [int(x) for x in vers.split('.')]

    new_vers = [int(x) for x in __version__.split('.')]

    if vers_f > new_vers:
        print('Loaded Datamodule newer than your installed version\n' +
              'Please update python_repo (git pull)')

    elif vers_f < new_vers or not isinstance(old_dm, current_types):
        # Check if old datamodule is below current version and upgrade if
        # Initialization
        if str(type(old_dm)) == "<class 'DataModule.data_line.data_line'>":
            old_dm.__class__ = data_line
        elif str(type(old_dm)) == "<class 'DataModule.data_xy.data_xy'>":
            old_dm.__class__ = data_table
            old_dm.name_x = old_dm.df.keys()[0]
            old_dm.name_y = old_dm.df.keys()[1]

        if isinstance(old_dm, data_complex):
            # Needs to be first since it is also instance of data_table
            data_new = data_complex(old_dm.x, old_dm.value)
            try:
                data_new.idx_min = old_dm.idx_min
                data_new.idx_max = old_dm.idx_max
            except AttributeError:
                data_new.idx_min = 0
                data_new.idx_max = None
            try:
                data_new.circuit = old_dm.circuit
            except AttributeError:
                pass
            data_new.fitresults = old_dm.fitresults
            data_new.fitresults_full_model = old_dm.fitresults_full_model


        elif isinstance(old_dm, (data_line, data_table)):
            # Data line
            data_new = data_table([old_dm.x, old_dm.y])
            try:
                data_new.idx_min = old_dm.xmin
                data_new.idx_max = old_dm.xmax  # Selected
            except AttributeError:
                data_new.idx_min = 0#old_dm.idx_min
                data_new.idx_max = None#old_dm.idx_max

            # For newer datamodules: Update dataframe
            try:
                data_new.df = old_dm.df
                data_new.name_x = old_dm.name_x
                data_new.name_y = old_dm.name_y
            except AttributeError:
                pass

            # TODO Sort data
            # Fit
            try:
                tmp = old_dm._fit_function_code
                data_new._fit_executed = True
                data_new._fit_function = old_dm._fit_function
                data_new._fit_function_code = tmp
                data_new._fit_parameters = old_dm._fit_parameters
                data_new._fit_par_errors = old_dm._fit_par_errors
                data_new._fit_data_error = old_dm._fit_data_error
                data_new._fit_labels = old_dm._fit_labels
            except AttributeError:
                pass

        elif isinstance(old_dm, data_grid):
            data_new = data_grid([old_dm.x, old_dm.y, old_dm.z],
                                 [old_dm.name_x,
                                  old_dm.name_y,
                                  old_dm.name_v])
        elif isinstance(old_dm, data_surface):
            data_new = data_grid([old_dm.x, old_dm.y, old_dm.z.T],
                                 ['x', "y", "z"])

        elif isinstance(old_dm, data_IQ):
            data_new = old_dm.copy()
            data_new.__version__ = __version__
        else:
            print('Datamodule {} type not recognized'.format(old_dm))
            raise Exception('TypeError')

        # Update parameters
        data_new.par = old_dm.par
        data_new.comments = old_dm.comments
        data_new.temp_start = old_dm.temp_start
        data_new.temp_stop = old_dm.temp_stop
        data_new.temp_start_time = old_dm.temp_start_time
        data_new.temp_stop_time = old_dm.temp_stop_time
        data_new.time_start = old_dm.time_start
        data_new.time_stop = old_dm.time_stop

        try:
            data_new.date_format = old_dm.date_format
            data_new.save_date_format = old_dm.save_date_format
        except AttributeError:
            pass

        try:
            oV = vers
            nV = data_new.__version__
            print('\rConversion {} --> {} executed'.format(oV, nV),
                  end=' ', flush=True)
            # Added \r to move cursor back to beginning if multiple files are
            # updated.
        except:
            print('\rConversion Unknown version --> {} executed'
                  .format(data_new.__version__), end=' ', flush=True)
        return data_new

    else:
        # No need to upgrade
        return old_dm


def findMinDiff(arr):
    """Helper function to find minimal resolution in data"""
    # Sort array in non-decreasing order
    arr = sorted(arr)

    # Initialize difference as infinite
    diff = 10 ** 20

    # Find the min diff by comparing adjacent
    # pairs in sorted array
    for i in range(len(arr) - 1):
        if arr[i + 1] - arr[i] < diff:
            diff = arr[i + 1] - arr[i]

            # Return min diff
    return diff

# TODO
###############################################################################
# NEED RETHINKING #############################################################
def data_stack_y(*args):
    """
        This function cannot work, because you stack Frequencies horizontally
        and stack the z value vertically.

        Ideas: - Use vertical x-axis. (can lead to complications with existing
                 functions)
    """
    data_type = args[0][0]
    if isinstance(data_type, data_surface):
        datas = list(args[0])
        tmp = datas.pop(0)
        tmp = tmp.copy()

        for count in range(len(datas)):
            a = datas.pop(0)
            if type(a) != data_type:
                print('Error, the data type are different')
                raise Exception('TYPE Mismatch')

            if not (a.x == tmp.x).all():
                print('Error: x-axis must be the same: ' + str(count))
                return Exception('WRONGAXIS')

            tmp.y = np.hstack((tmp.y, a.y))
            tmp.z = np.vstack((tmp.z, a.z))
        return tmp

    data_type = type(args[0][0])
    if data_type is type(data_table()):
        # all modules must be the same
        datas = list(args[0])
        tmp = datas.pop(0)
        tmp = tmp.copy()  # I have to copy it, otherwise the argument will be modified

        for count in range(len(datas)):
            a = datas.pop(0)
            if type(a) != data_type:
                print('Error, the data type are different')
                raise Exception('TYPE Mismatch')

            if not (a.x == tmp.x).all():
                print('ERROR: x-axis must be the same: ' + str(count))
                raise Exception('WRONGAXIS')

            tmp.x = np.hstack((tmp.x, a.x))
            tmp.y = np.hstack((tmp.y, a.y))

        tmp.select()
        return tmp

    elif data_type is type(data_complex()):
        # all modules must be the same
        datas = list(args[0])
        tmp = datas.pop(0)
        tmp = tmp.copy()  # I have to copy it, otherwise the argument will be modified

        for count in range(len(datas)):
            a = datas.pop(0)
            if type(a) != data_type:
                print('Error, the data type are different')
                raise Exception('TYPE Mismatch')

            if not (a.x == tmp.x).all():
                print('ERROR: x-axis must be the same: ' + str(count))
                raise Exception('WRONGAXIS')

            tmp.x = np.hstack((tmp.x, a.x))
            tmp.value = np.hstack((tmp.value, a.value))

        tmp.select()
        return tmp


    elif data_type is type(data_surface()):
        datas = list(args[0])
        tmp = datas.pop(0)
        tmp = tmp.copy()  # I have to copy it, otherwise the argument will be modified

        for count in range(len(datas)):
            a = datas.pop(0)
            if type(a) != data_type:
                print('Error, the data type are different')
                raise Exception('TYPE Mismatch')

            if not (a.x == tmp.x).all():
                print('Error: x-axis must be the same: ' + str(count))
                return Exception('WRONGAXIS')

            tmp.y = np.hstack((tmp.y, a.y))
            tmp.z = np.vstack((tmp.z, a.z))

        return tmp
    else:
        print('No routing for this kind of data')
        return None


def nice_plot(figsize=(15, 15), plot_style='bo', xlabel='', ylabel='', title='',
              fontsize=22, xoffset=None, yoffset=None,
              fontname='Bitstream Vera Serif', box_param=None):
    '''
        functionf nice_plot(figsize=(15,15),plot_style='bo',xlabel='',ylabel='',title='',fontsize=22,xoffset=None, yoffset=None,fontname='Bitstream Vera Serif',legend=None,legend_pos=0,box_param=None):
        
        given a data module 2D, this function can be used to plot the data in a nicer way (for talks or articles). If the data has been fitted, it will also print the fit (it can be disabled, def is on).

                
        xoffset and yoffset can be used to use a scientific notation on the axis
        fontname='Bitstream Vera Serif' HINT: Use font_list() to have a list of the available fonts

        legend_pos: 0 is auto, 1 to 4 are the angles
        legend: must be a list of two strings, ex: ['data','fit']
        box_param: must be a list of 4 coordinates and a text, ex: ([0.01,0.01,0.1,0.1], 'testing the box')
        The first two coordinates are the position of the top-left angle in % of the total plot size.
        The others are width and length, always in total % plot size
        '''
    plt.rcParams.update({'font.size': fontsize})
    plt.figure(figsize=figsize)

    plt.xlabel(xlabel, fontname=fontname), plt.ylabel(ylabel, fontname=fontname)

    plt.title(title, fontname=fontname)

    # axis ticks and tick labels

    plt.tick_params(axis='both', length=6, width=2, pad=10)
    # axis ticks and tick labels

    tickfonts = matplotlib.font_manager.FontProperties(family=fontname,
                                                       size=fontsize)
    tmp = plt.axes()
    for label in (tmp.get_xticklabels() + tmp.get_yticklabels()):
        label.set_fontproperties(tickfonts)

    # plt.xaxis.get_major_formatter().set_useOffset(False)
    # plt.yaxis.get_major_formatter().set_useOffset(False)
    if xoffset is None:
        xconf = plt.ScalarFormatter(useOffset=False)
    else:
        xconf = plt.ScalarFormatter(useOffset=True)
        xconf.set_useOffset(xoffset)

    if yoffset is None:
        yconf = plt.ScalarFormatter(useOffset=False)
    else:
        yconf = plt.ScalarFormatter(useOffset=True)
        yconf.set_useOffset(yoffset)

    tmp.xaxis.set_major_formatter(xconf)
    tmp.yaxis.set_major_formatter(yconf)
    if box_param != None:
        plt.text(box_param[0][0], box_param[0][1], box_param[1], size=24,
                 bbox=dict(boxstyle='square', fc=(1, 1, 1)))
        # plt.text(0.1,0.16,box_param[1],fontname=fontname)
        # plt.setp(box,xlim=(0,2),xticks=[],yticks=[])
        # plt.subplot(111)
