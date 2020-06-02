# -*- coding: utf-8 -*-

#import the libraries
import numpy as np

from lib.LightCurve import Target
from lib.algorithm import TransitSearch

import argparse
import time
import os

#choose the backend for matplotlib
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.figure()
    plt.close('all')
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')


parser = argparse.ArgumentParser(description='-o for name of the output folder \
                  -i for the location of the input location -n for the location \
                  of the lightcurve file')
parser.add_argument('-i', nargs=1, help="location of the input light curve")
parser.add_argument('-n', nargs=1, help="name for target light curve")
parser.add_argument('-o', nargs=1, help="location of the output folder")

args = parser.parse_args()


Path = "data" if not(args.i) else args.i[0]
OutputDir = "" if not(args.o) else args.o[0]


#Initiate the target
if args.n:
    CurrentTarget = Target(Location=Path, Name=args.n[0], Output=OutputDir)
else:
    raise NameError("The name of the target to be run is not provided.")

plt.figure()
plt.plot(CurrentTarget.DailyData[0][:,0], CurrentTarget.DailyData[0][:,1], "ko")
plt.show()

TransitModel = TransitSearch()
TransitModel.LinearSearch(CurrentTarget)
TransitModel.PeriodicSearch(CurrentTarget)
