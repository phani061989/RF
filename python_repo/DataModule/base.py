# -*- coding: utf-8 -*-
"""Base functions of every datamodule type"""

import os
import time
import pickle
import copy
import numpy as np
from .version import __version__

# For saving to .h5 and .nc
import pandas as pd


class data_module_base(object):
    """Base class of DataModule

    These functions is used by every datamodule.
    """
    def __init__(self):
        self.__version__ = __version__
        self.comments = ''
        self.par = {}  # Parameters for collected data (e.g. VNA settings)
        self.temp_start = None
        self.temp_stop = None
        self.temp_start_time = None
        self.temp_stop_time = None
        self.time_start = None
        self.time_stop = None
        self.date_format = '%Y-%m-%d-%H:%M:%S'
        self.save_date_format = '%Y-%m-%d'
        self.idx_min = 0
        self.idx_max = None

    def insert_par(self, **kwargs):
        """Add parameters to the data module given by keywords.

        Example
        ---------
            >>> data.insert_par(temp= 25e-3, comment='test', foo='bar')
        """
        self.par.update(**kwargs)

    def remove_par(self, key):
        """Remove parameter by key from data module parameters

        Parameters
        -----------
        key : str
            Key of parameter dictionary
        """
        try:
            self.par.pop(key)
        except KeyError:
            raise Exception('Parameters empty or key not found')

    def save(self, fname, useDate=True, force=False):
        """Save DataModule

        The date will be added in front of the filename and a '.dm' extension
        will be added automatically to fname, if not already given.

        If the file already exists, the existing file will be moved in a
        subfolder named duplicates. If this happens multiple times, numbers
        will be added to the files in the duplicates folder.

        Parameters
        -----------
        fname : str
            Filename to save
        useDate : bool
            Add Date in front of fname
        force: Overwrite existing file.
        """
        # Split fname in folder and filename
        path = os.path.split(fname)

        # Create directory if not already existent
        if not os.path.exists(path[0]) and path[0]:
            os.makedirs(path[0])

        # Add date
        if useDate:
            time_string = time.strftime(self.save_date_format,
                                        time.localtime(time.time()))
            file_name = time_string + '-' + path[1]

        else:
            file_name = path[1]

        # Define datatype
        if self.dtype == 'data_complex':
            dtype = 'dm_complex'
            ending = '.dm.h5'
        elif self.dtype == 'data_table':
            dtype = 'dm_table'
            ending = '.dm.h5'
        elif self.dtype == 'data_grid':
            dtype = 'dm_grid'
            ending = '.dm.nc'
        else:
            raise Exception('Error. Datatype not supported')

        # Check for file extension
        if path[1][-3:].lower() == '.dm':
            # If old datatype ending, just add .h5 or .nc
            file_name += ending[-3:]
        elif path[1][-3:].lower() in ['.h5', '.nc'] and path[1][-5:-3] != 'dm':
            # Just added .h5 or .nc but not .dm. in the middle, --> add full end
            file_name = file_name[:-3] + ending
        elif path[1][-5:].lower() not in ['dm.h5', 'dm.nc']:
            # All the other endings without correct filetype
            file_name += ending

        # Append Folder and be adaptive to windows, etc.
        file_name = os.path.normpath(os.path.join(path[0], file_name))
        filename = file_name

        # Whitelist of allowed parameters to be stored
        white_list = (str, dict, bool, float, int, np.float64, np.int64,
                      np.ndarray)

        # Check for overwrite. If file exists move to duplicates
        if not force:
            if os.path.isfile(file_name):
                from shutil import copyfile
                # Add a number if force-Overwrite is False
                fpath, fn = os.path.split(file_name)
                # Create duplicates folder
                if not os.path.exists(os.path.join(fpath, 'duplicates')):
                    os.makedirs(os.path.join(fpath, 'duplicates'))
                fpath = os.path.join(fpath, 'duplicates')
                # Split twice since .dm.h5 (two dots)
                fn, e = os.path.splitext(os.path.splitext(fn)[0])
                file_name2 = os.path.join(fpath, fn + '%s' + file_name[-6:])
                number = ''
                while os.path.isfile(file_name2 % number):
                    number = int(number or "0") + 1
                file_name2 = file_name2 % number  # Add number
                copyfile(file_name, file_name2)
                print('File already exists.\nTo prevent data loss, the old' +
                      'file - eventually with a number appended - has been ' +
                      'moved into the subfolder duplicates.\n',
                      flush=True, end=" ")

        # Save as hdf5 file
        if dtype in ['dm_complex', 'dm_table']:
            # Save with pandas HDF writer
            with pd.HDFStore(filename, 'w') as f:
                # Save dataframe
                f.put(dtype, self.df, format='table')  # , data_columns=True)

                # Save datamodule metadata
                attrs = f.get_storer(dtype).attrs

                metadata_dict = {}
                for i, k in self.__dict__.items():
                    if isinstance(k, np.ndarray):
                        # Special case, since numpy arrays are not that well
                        # serializable
                        metadata_dict[i] = list(k)
                    elif isinstance(k, white_list):
                        metadata_dict[i] = k

                attrs['dm_metadata'] = repr(metadata_dict)
                # attrs['dm_metadata'] = repr({i: k
                #                        for i,k in self.__dict__.items()
                #                        if isinstance(k, white_list)})

        elif dtype in ['dm_grid']:
            # Save with xarray netCDF writer

            # Save datamodule metadata
            attrs = {}
            for i, k in self.__dict__.items():
                if isinstance(k, white_list):
                    attrs[i] = k
            self.df.attrs['dm_metadata'] = repr(attrs)
            self.df.to_netcdf(filename)
        return None

    def copy(self):
        """Copy datamodule.

        Returns
        --------
        Datamodule
            A copy of this datamodule
        """
        return copy.deepcopy(self)

    def select(self, xrng=None):
        """Select range of data.

        Plots, fits, etc will then only be applied on this range.
        If nothing is specified all the data will be select

        Parameters
        ----------
        xrng : list, None
            Start and Stop values of the range in a list [start, stop]. Eg. [1.4e9, 6.5e9]
        """
        self.idx_min = 0
        self.idx_max = None

        if xrng is not None:
            idx = np.where((self.x >= xrng[0]) & (self.x <= xrng[1]))[0]
            self.idx_min = idx[0]
            self.idx_max = idx[-1]
