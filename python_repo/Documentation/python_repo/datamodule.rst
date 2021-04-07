.. _python_repo:

************
DataModule
************

This is a object for storing and processing our data.
It comes with various handy functions and classes. Please chose your class
depending on your data structure.

.. automodule:: DataModule


Basic DataModule Functions
===========================

.. automodule:: DataModule.functions
    :members:

DataModule Classes
===================

The `DataModule` has several classes, depending on the type of data you want to
store.

Base
-----
.. autoclass:: DataModule.data_module_base
    :members:

data_table
----------
Basic class for real x, y data.

.. autoclass:: DataModule.data_table
    :members:

data_complex
-------------
.. autoclass:: DataModule.data_complex
    :members:

data_grid
-------------
.. autoclass:: DataModule.data_grid
    :members:

fit_functions
==============
.. automodule:: DataModule.fit_functions
    :members:

plotting
=========
Useful plot functions and default settings for the DataModule plots

.. automodule:: DataModule.plot_style
    :members:
