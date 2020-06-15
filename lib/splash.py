'''
Will contain the core functionality of reading
the data from the Cambridge Pipeline.
'''

import numpy as np
from time import time
import os
from astropy.io import fits
import re

from .Functions import ReadTxtData, ReadFitsData,\
     ParseFile, GetID


class Target:
    '''
    This is a class for a target of data
    '''

    def __init__(self, Location="data", Name ="", Output=""):
        '''
        Expect a continuous data concatenated data with header in the first row.
        First row is expected to be time. Second row is expected to be flux.

        ParamName: The name of the parameters.
        ParamValue: The values of the parameters.
        '''

        if len(Location)>0 and len(Name)>0:
            #self.ParamNames, self.ParamValues = ReadOldData(Location, Name)
            self.ParamNames, self.ParamValues = ReadFitsData(Location, Name)

        #Generate the output folder
        self.ID, self.OutputPath = self.MakeOutputDir(FolderName=Output)
        self.DailyData = self.NightByNight()
        self.NumberOfNights = len(self.DailyData)

        #Flag to produce light curves once the data been processed
        self.Processed = False

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


    def NightByNight(self):
        '''
        Slice the data into night by night elements.
        '''
        #Find where the location

        Time = self.ParamValues[:,0]
        Locations = np.where(np.diff(Time)>0.20)[0]
        SlicedData = []
        Start = 0
        for ChunkCount in range(len(Locations)+1):
            if ChunkCount<len(Locations):
                Stop = Locations[ChunkCount]
            else:
                Stop = len(Time)
            SlicedData.append(self.ParamValues[Start:Stop])
            Start = Stop
        return SlicedData


    def MakeOutputDir(self, FolderName=""):
        '''
        Check if the output directory exists, and
        make the directory if they do not exist
        '''

        if FolderName:
            ID = FolderName
        else:
            ID = hex(int(time()*1000)).replace('0x16',"")
        OutputDir = os.getcwd()+"/Output"

        if not(os.path.exists(OutputDir)):
            os.system("mkdir %s" %OutputDir.replace(" ", "\ "))

        ResultDir = OutputDir+"/"+ID

        if not(os.path.exists(ResultDir)):
            print("Creating the folder.")
            os.system("mkdir %s" %ResultDir.replace(" ", "\ "))
        else:
            print("The output folder already exists. Deleting all previous files and folders within it.")
            os.system("rm  -rf %s/*" %ResultDir.replace(" ", "\ "))
        return ID, ResultDir



    @property
    def MaskFlares(self):
        '''
        Check if the output directory exists
        '''
        input("Inside masking flares. Yet to be implemented")
        pass



    def NormalizeFlux(self):
        '''
        This method normalizes the flux
        '''
        input("Normalized Flux")
        pass



    def GetNormalizedFlux(self):
        '''
        This method normalizes the flux
        '''
        input("Normalized Flux")
        pass


    def GetDataByNight(NightNum):
        '''
        NightNumber is the night with the first case being 1.
        '''
        input("Inside get data by night")
        pass


    def GetTimeFluxByNight(NightNum):
        '''
        NightNumber is the night with the first case being 2.
        '''
        input("Inside get Time Flux  by night")
        pass


    def MakePlots(self):
        '''
        This method will make plots for the figures.
        '''

        if not(self.Processed):
            print("The data have not be processed yet.")

        for i in range(self.NumberNights):
            plt.figure()
            plt.plot(self.Time[i], self,Flux[i], "ko")
            plt.show()

        pass
