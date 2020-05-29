# -*- coding: utf-8 -*-

#Modify the path




#import the libraries
import numpy as np


import matplotlib.pyplot as plt
from lib import Functions as f
from lib import TransitSearch as TS
from lib import Periodogram as PG
import matplotlib
matplotlib.use('Agg')

import argparse
import time

parser = argparse.ArgumentParser(description='-o for name of the output folder \
                  -i for the location of the input location')
parser.add_argument('-o', nargs=1, help="name for name of the output folder")
parser.add_argument('-i', nargs=1, help="location of the input light curve")

args = parser.parse_args()


ReadFileLocation=open("SearchParams.ini", "r")
AllData =  ReadFile.readlines()

#Parse the input folder
FolderName = AllData[10].split(":")[1].split("#")[0].replace(" ","")
LC_Location = AllData[12].split(":")[1].split("#")[0].replace(" ","")
DefaultDetrendMethod = int(AllData[6].split(":")[1].split("#")[0].replace(" ",""))

print(FolderName)
print(LC_Location)
print(DefaultDetrendMethod)
input("Wait here...")

#if argument for the light curve is provided
if args.i:
    LC_Location = args.i[0]




#Reading the data
try:
    #Read the data
    JD_UTC, Flux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS, ExpTime = np.loadtxt(LC_Location, unpack=True)
except:
    #Read the data separated by comma
    JD_UTC, Flux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS, ExpTime = np.loadtxt(LC_Location, delimiter=",",unpack=True)
#in order to inject sign
JD_UTC, Flux, LC_Location = f.InjectTransit(JD_UTC, Flux, LC_Location, Parameter='b')




#if the output folder is provided during the run
if args.o:
    RUNID, OutputDir = f.CreateUniqueID(FolderName=args.o[0])
elif FolderName:
    RUNID, OutputDir = f.CreateUniqueID(FolderName=FolderName)
else:
    RUNID, OutputDir = f.CreateUniqueID()


NormalizedFlux = f.NormalizeFlux(JD_UTC, Flux, OutputDir, FlattenMethod=3, Plot=True)
STD = np.std(NormalizedFlux)

#Create the corner plot for diagnosis
ParametersName = ["NormalizedFlux", "Err", "XShift", "YShift", "FWHM_X",  "FWHM_Y", "FWHM",	"SKY", "AIRMASS"]
Parameters = np.column_stack((NormalizedFlux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS))
#f.CreateCornerPlot(Parameters, ParametersName, OutputDir)

SearchFileLocation = os.getcwd()+"/SearchParams.ini"


#Preclean the light curve
##Lionel


#Find the phase CoverageNightl
PG.PhaseCoverage(JD_UTC, OutputDir)

#Search for the single transit evenets within each night
TS.SingleEventSearch(JD_UTC, Flux, Parameters, ParametersName, SearchFileLocation, STD, OutputDir)


PG.SearchPeriodicSignal(FolderName, OutputDir)


#Perform detailed fit
###Section to be completed by Lionel

print("Now generating report...")
#f.GeneratePdfReport(OutputDir, LC_Location)


print("Now generating report...")
#f.GeneratePdfReport(OutputDir, LC_Location)
