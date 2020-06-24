'''
Will contain the core functionality of reading
the data from the Cambridge Pipeline.
'''

import os
import re
from time import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from .Functions import ReadTxtData, ReadFitsData,\
     ParseFile, GetID, FindQuality


#formatting for the image
import matplotlib as mpl
mpl.rc('font',**{'sans-serif':['Helvetica'], 'size':15,'weight':'bold'})
mpl.rc('axes',**{'labelweight':'bold', 'linewidth':1.5})
mpl.rc('ytick',**{'major.pad':22, 'color':'k'})
mpl.rc('xtick',**{'major.pad':10,})
mpl.rc('mathtext',**{'default':'regular','fontset':'cm','bf':'monospace:bold'})
mpl.rc('text.latex',preamble=r'\usepackage{cmbright},\usepackage{relsize},'+r'\usepackage{upgreek}, \usepackage{amsmath}')
mpl.rc('contour', **{'negative_linestyle':'solid'})


class Target:
    '''
    This is a class for a target of data
    '''

    def __init__(self, Location="data", Name ="", Output="", version=1):
        '''
        Expect a continuous data concatenated data with header in the first row.
        First row is expected to be time. Second row is expected to be flux.

        ParamName: The name of the parameters.
        ParamValue: The values of the parameters.
        '''

        if len(Location)>0 and len(Name)>0:
            if version==0:
                print("Loading from txt file.")
                self.ParamNames, self.ParamValues = ReadTxtData(Location, Name)
            elif version==1 or version==2:
                print("Loading from fits file.")
                self.ParamNames, self.ParamValues = ReadFitsData(Location, Name, version=version)
            else:
                input("Only three version are available")

        self.AllTime = self.ParamValues[:,0]
        self.AllFlux = self.ParamValues[:,1]

        #Find the id from the database
        if "Sp" in Name:
            self.SpeculoosID = Name
            self.GaiaID  = GetID(Name, IdType="SPECULOOS")
        elif re.search("[0-9]{17}",Name):
            #GAIA ID is at least 17 digit long
            self.GaiaID = Name
            self.SpeculoosID = GetID(Name, IdType="GAIA")
        else:
            self.SpeculoosID="Not Found"
            self.GaiaID = "Not Found"

        Output = self.SpeculoosID if not(Output) else Output
        #Generate the output folder
        self.MakeOutputDir(FolderName=Output)

        self.DailyData = self.NightByNight()
        self.NumberOfNights = len(self.DailyData)

        #Flag to produce light curves once the data been processed
        self.Processed = False
        self.QualityFactor = np.ones(len(self.ParamValues[:,0])).astype(np.bool)

        #Break the quality factory into nights
        self.BreakQualityFactor()
        self.PhaseCoverage()




    def NightByNight(self):
        '''
        Slice the data into night by night elements.
        '''
        #Find where the location

        self.NightLocations = []
        Time = self.ParamValues[:,0]
        BreakLocations = np.where(np.diff(Time)>0.20)[0]+1

        SlicedData = []
        Start = 0
        for ChunkCount in range(len(BreakLocations)+1):
            if ChunkCount<len(BreakLocations):
                Stop = BreakLocations[ChunkCount]
            else:
                Stop = len(Time)+1
            SlicedData.append(self.ParamValues[Start:Stop])
            self.NightLocations.append([Start, Stop])
            Start = Stop
        return SlicedData


    def MakeOutputDir(self, FolderName=""):
        '''
        Check if the output directory exists, and
        make the directory if they do not exist

        Parameters
        -----------
        FolderName: string
                    Name of the Output directory to be created

        Yields:

        '''

        if FolderName:
            self.ID = FolderName
        else:
            self.ID = hex(int(time()*1000)).replace('0x16',"")
        self.OutputDir = os.getcwd()+"/Output"

        if not(os.path.exists(self.OutputDir)):
            os.system("mkdir %s" %self.OutputDir.replace(" ", "\ "))

        self.ResultDir = self.OutputDir+"/"+self.ID

        if not(os.path.exists(self.ResultDir)):
            print("Creating the folder.")
            os.system("mkdir %s" %self.ResultDir.replace(" ", "\ "))
        else:
            print("The output folder already exists. Deleting all previous files and folders within it.")
            os.system("rm  -rf %s/*" %self.ResultDir.replace(" ", "\ "))
        return 1




    def PreClean(self, CutOff=7, NIter=2, Columns=-1, MinDataCount=50,
                 ShowPlot=False, SavePlot=False):
        '''
        Pre cleans the data

        Parameters
        -------------

        NumPoint: integer
                  Number of points for generating the gaussian function

        NIter: integer

        Columns:-1/1
                -1 - Consider all column except time to look for outlier
                 1 - Consider only the differential flux to look for the outlier

        MinDataCount: integer
                default value 50. Consider the data for the night if at least
                this number of data is present.

        ShowPlot: bool
                 Plots the figure for viewing

        SavePlot: bool
                 Saves the figure at the location of the output cirector

        Return:
            Initiates the quality factor list which can be used to select data

        '''

        StartIndex = 0
        #Measure a quality factor for a data
        for NightNumber in range(self.NumberOfNights):
            CurrentData = self.DailyData[NightNumber]
            nRow, nCol = np.shape(CurrentData)
            CurrentQuality = np.ones(nRow).astype(np.bool)
            CurrentTime = CurrentData[:,0]
            CurrentFlux = CurrentData[:,1]

            #Use only flux. Override the value of nCol
            if Columns ==1:
                nCol = 2

            for j in range(1,nCol):
                if "TIME" in self.ParamNames[j].upper() or "SKY" in self.ParamNames[j].upper() or "AIRMASS" in self.ParamNames[j].upper():
                    continue
                Data2Consider = CurrentData[:,j]
                if len(Data2Consider)<MinDataCount:
                    CurrentQuality = np.zeros(len(CurrentData)).astype(np.bool)
                else:
                    Quality = FindQuality(CurrentTime, Data2Consider, CutOff=7.5, NIter=1)
                    CurrentQuality = np.logical_and(CurrentQuality, Quality)
                    NumDataPoints = np.sum(~CurrentQuality)

                    if NumDataPoints>10:
                        warning("More than 10 points marked with bad quality flag.")

            StartIndex,StopIndex = self.NightLocations[NightNumber]
            self.QualityFactor[StartIndex:StopIndex] = CurrentQuality

            if SavePlot or ShowPlot:
                T0_Int = round(min(CurrentTime),6)

                plt.figure(figsize=(10,8))
                plt.plot(CurrentTime[CurrentQuality]-T0_Int, CurrentFlux[CurrentQuality], "ko", label="Good Data")
                plt.plot(CurrentTime[~CurrentQuality]-T0_Int, CurrentFlux[~CurrentQuality], "ro", label="Bad Data")
                plt.xlabel("JD "+str(T0_Int), fontsize=25)
                plt.ylabel("Normalized Flux", fontsize=25)
                plt.legend(loc=1)
                plt.tight_layout()
                if SavePlot:
                    self.OutlierLocation = os.path.join(self.ResultDir, "Outliers")
                    if not(os.path.exists(self.OutlierLocation)):
                        os.system("mkdir %s" %self.OutlierLocation)
                    SaveName = os.path.join(self.OutlierLocation, str(NightNumber).zfill(9)+".png")
                    plt.savefig(SaveName)
                if ShowPlot:
                    plt.show()
                plt.close('all')


    def BreakQualityFactor(self):
        '''
        Method to break up the quality indices night by night
        '''

        self.QualityFactorFromNight = []
        for Start, Stop in self.NightLocations:
            self.QualityFactorFromNight.append(self.QualityFactor[Start:Stop])


    def PhaseCoverage(self, StepSize=0.05, PLow=0.30, PUp=250.0, NTransits=2, Tolerance=0.005):
        '''
        This function calculates the phase coverage of the data

        ################################
        Parameters:
        =================
        PLow: float
              Lowest Period to consider

        PUp: float
             Largest Period. Default value le

        StepSize: float
                  the stepsize of the period to be considered

        NTransits: integer
                  The number of transits to be used with

        Tolerance:
                    is to see the phase coverage for different phases
        '''


        #Redefine maximum periodicity to be looked for phase coverage
        ExpPeriodCoverage = (max(self.AllTime)-min(self.AllTime))
        if ExpPeriodCoverage<PUp:
            PUp=ExpPeriodCoverage

        self.PhasePeriod  = np.arange(PLow, PUp, StepSize)
        self.PhaseCoverage = np.ones(len(self.PhasePeriod ))

        for Count, Period in enumerate(self.PhasePeriod):
            Phase = self.AllTime%Period
            Phase = Phase[np.argsort(Phase)]
            Diff = np.diff(Phase)

            self.PhaseCoverage[Count] -= Phase[0]/Period
            self.PhaseCoverage[Count] -= (Period-Phase[-1])/Period

            Locations = np.where(Diff>0.005)[0]
            for Location in Locations:
                PhaseUncovered = Phase[Location+1]-Phase[Location]
                self.PhaseCoverage[Count] -= PhaseUncovered/Period

        self.PhaseCoverage*=100.
