#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:16:36 2024

@author: DhruvB
"""

#CAGE_master.py

import os
import argparse

my_parser = argparse.ArgumentParser()

my_parser.add_argument('csv_path', type=str, help='the path to the csv input file.')
args = my_parser.parse_args()

csv_path = args.csv_path

os.system('python3 -W ignore kinms_runner.py '+csv_path)
#os.system('python3 -W ignore modelling_summary_plotter.py.py '+csv_path) #haven't fully formatted this yet.
os.system('python3 -W ignore anomalous_gas_estimator.py '+csv_path)
