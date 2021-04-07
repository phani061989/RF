#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:33:09 2018

@author: oscar

v6.0.4 - OSC:
    - improved __measure_waves function, it will lock on the down_mixed frequency now
v6.0.3 - OSC:
    - bugfix: phase in Rabi was wrong, it was leading to a wrong Rabi period
v6.0.2 - OSC:
    - forgot to close spec. anal. connection after calibration
v6.0.1 - OSC
    - bugfix in do_calibration function, it wasn't possible to choose RS

v6.0.0 - Implementing automation for usual qubit characterisation experiments


"""

qubexp_version = '6.0.4'