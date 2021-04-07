.. _python_repo:

************
Instruments
************

This class is used for communication to the lab devices

To see all available devices use:

.. highlight:: python
.. code-block:: python

	import Instruments
	Instruments.list()

.. automodule:: Instruments
    :members:

Current Source (CS)
====================

Class for communication with current Sources. Currently Yokogawa GS210 and Keysight SMU2900B are supported.

.. automodule:: Instruments.CS
	:members:

Digital Attenuator (DA)
========================

Class for communication with digital step attenuators. Currently Rohde und Schwarz and MiniCircuits are supported. 

.. automodule:: Instruments.DA
	:members:

Labjack (LJ)
=============

Class for communcation with Labjack.

.. automodule:: Instruments.LJ
	:members:

Spectrum Analyzer (SA)
========================
Class for communcation with spectrum analyzers.
Currently supported:
- Rohde+Schwarz FSV Series
- Tektronix RSA Series

.. automodule:: Instruments.SA
	:members:

Signal Generator (SG)
=======================
Class for communcation with signal generators.
Currently supported:
- Anapico 
- Keysight

.. automodule:: Instruments.SG
	:members:

Vector Network Analyzer (VNA)
===============================
Class for communcation with VNAs
Currently supported:
- Keysight 5071C
- Keysight 5080A
- Keysight P5000 + Spectrum Analyzer Option

.. automodule:: Instruments.VNA
	:members:

XXF-1 (XXF1)
=============
Class for communcation Magnicon SQUID Electronics XXF-1

.. automodule:: Instruments.Drivers.Magnicon.magnicon32
	:members:






