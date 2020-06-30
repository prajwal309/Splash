# -*- coding: utf-8 -*-

#import the libraries
import numpy as np

from lib.splash import Target
from lib.algorithm import GeneralTransitSearch
from lib.sampler import TransitFit

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

#version 0 for txt file formats with headers
#version 1 for version 1 of
#version 2 for Cambridge version 2.
parser.add_argument('-v', nargs=1, help="version of the pipeline")

args = parser.parse_args()

#Assign default value if the
Path = "data" if not(args.i) else args.i[0]
OutputDir = "" if not(args.o) else args.o[0]
Version = 1 if not(args.v) else int(args.v[0])

#Initiate the target
if args.n:
    CurrentTarget = Target(Location=Path, Name=args.n[0], Output=OutputDir, version=Version)
else:
    raise NameError("The name of the target to be run is not provided.")


CurrentTarget.PreClean(CutOff=5, NIter=2, Columns=-1, \
              MinDataCount=75, SavePlot=True)

SVDSearch = GeneralTransitSearch()
SVDSearch.Run(CurrentTarget, SavePlot=True)


#SVDSearch.PeriodicSearch(CurrentTarget, method="TLSLikelihood", \
#ShowPlot=False, SavePlot=True)

#SVDSearch.PeriodicSearch(CurrentTarget, method="TLS", \
#ShowPlot=False, SavePlot=True)

SVDSearch.PeriodicSearch(CurrentTarget, method="TransitPair", \
ShowPlot=False, SavePlot=True)

TransitFit(CurrentTarget, SVDSearch, TDur=2.0)
