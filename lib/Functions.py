'''
This file contains the important function that is imported within the module
'''

import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import glob

from astropy.io import fits
from functools import reduce
from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.signal import gaussian
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import filters


def FindLocalMaxima(Data, NData=4):
    '''
    This function finds the value of the local maxima

    Input Parameter:
    ------------------

    Data: numpy array
          Data where the local maxima is to be found.

    NData: integer
          Number of neighboring data points to be considered.

    Returns
    ------------------
    returns an array of index for the night

    '''

    Index = np.zeros(len(Data)).astype(np.bool)
    for counter in range(len(Data)):
        Data[counter-NData:counter+NData]
        StartIndex = counter-NData
        if StartIndex<0:
            StartIndex = 0
        StopIndex = counter+NData+1
        if StopIndex>len(Data):
            StopIndex=len(Data)
        Index[counter] = Data[counter]>0.999999*max(Data[StartIndex:StopIndex])
    return Index



def RunningResidual(Time, Flux, NumBins):
    '''
    Function yields the moving average of the data

    Parameters
    ------------
    Time: array of float
            array for which the moving average is to be determined

    Residual: array of float
              array for which the moving average is to be determined

    NumBins: integer
             Number of points for generating the gaussian function

    Returns
    -------------
    arrays
    Value of standard deviation
    '''

    NumPoints = int(len(Time)/NumBins)
    CurrentSTD = []
    for i in range(NumBins):
        Start = i*NumPoints
        Stop = (i+1)*NumPoints
        CurrentSTD.append(np.std(Flux[Start:Stop])/(np.sqrt(NumPoints)))
    CurrentSTD = np.array(CurrentSTD)
    return CurrentSTD




def moving_average(series, sigma=5, NumPoint=75):
    '''
    Function yields the moving average of the data

    Parameters
    ------------
    series: array of float
            array for which the moving average is to be determined

    sigma: float
           Standard deviation used to construct the normal function

    NumPoint: integer
              Number of points for generating the gaussian function

    Returns
    -------------
    This function returns
    '''
    b = gaussian(NumPoint, sigma)
    average = filters.convolve1d(series, b/b.sum())
    var = filters.convolve1d(np.power(series-average,2), b/b.sum())
    return average, var


def FindQuality(Time, Data, CutOff=6.0, NIter=2):
    '''
    Function to find quality based on all the data

    Parameter
    ----------
    Time: array
          The time series of the data

    Data: array
          The data series for finding the outliers

    CutOff: float
            The cutoff value for finding the threshold for the cutoff

    NIter: int
          The number if interation for finding the outlier

    Returns
    ------------
    Array of boolean based on


    '''

    NanIndex = np.logical_or(np.isnan(Time),np.isnan(Data))
    SelectIndex = ~NanIndex

    for IterCount in range(NIter):
        _ , var = moving_average(Data[SelectIndex], )

        spl =  UnivariateSpline(Time[SelectIndex], Data[SelectIndex], w=1.0/np.sqrt(var))
        trend = spl(Time)
        Residual = Data- trend
        STD = np.std(Residual[SelectIndex])
        Value = np.abs(Residual)/STD
        SelectIndex = np.logical_and(SelectIndex, Value<CutOff)
    return SelectIndex



def ParseFile(Location):
    '''
    This function parse Search Parameters initialization file

    Input
    #####################################
    Location of the search initialization file

    Output
    #####################################
    The parameters in dictionary format
    '''

    with open(Location,'r') as f:
        Data = f.readlines()

    ValueDict = {}

    for Line in Data[1:]:
        LineItem = Line.split("#")[0]
        Key, Value = LineItem.split(":")
        ValueDict[Key] = Value.replace(" ", "")

    return ValueDict



def ReadTxtData(Location, TargetName):
    '''
    This function reads the input file

    Input
    #####################################
    Location: Path to the folder containing the light curve.
    TargetName: Name of the target for identifying the files.

    Output
    #####################################
    Name of the parameters
    Values of the parameters
    '''
    if len(Location) <1 or len(TargetName)<1:
        raise NameError("No location or target available")

    FileList = glob.glob(Location+"/*%s*.txt*" %TargetName)
    NumFiles = len(FileList)

    if NumFiles == 0:
        raise NameError("No Files found")


    AllData = []

    for Counter,FileItem in enumerate(FileList):
        #Headers
        if Counter ==0 :
            Header = open(FileItem,'r').readline().upper()
            CSVFileFlag = "," in Header
            if CSVFileFlag:
                TempParameter = Header.split(",")
            else:
                TempParameter = Header.split("\t")

            ParamName = []

            for Param in TempParameter:
                ParamName.append(Param.replace(" ", "").replace("#","").replace("\n",""))


        try:
            Data = np.loadtxt(FileItem,skiprows=1, delimiter=",")
        except:
            Data = np.loadtxt(FileItem, skiprows=0)

        AllData.extend(Data)

    AllData = np.array(AllData)
    ParamName = np.array(ParamName)
    return ParamName, AllData


def ReadFitsData(Location, TargetName, version=1):
    '''
    This function reads the input file from Cambridge Pipeline

    Parameter
    ------------
    Location: string
              Path to the folder containing the light curve.

    TargetName: string
                Name of the target used to identifying the files.
                Either SpeculoosID or GAIAID is expected

    version: int
            Version of the dataproduct being used. Version 1 is
            different from version 2.
    Yields
    ---------
    Name of the parameters
    Values of the parameters
    '''


    if len(Location) <1 or len(TargetName)<1:
        raise NameError("No location or target available")

    FileList = glob.glob(Location+"/*%s*.fits*" %TargetName)
    NumFiles = len(FileList)


    if NumFiles == 0:
        raise NameError("No Files found")


    AllData = []

    if version==1:
        ParamName = ["TIME", "FLUX", "AIRMASS", "FWHM", \
                     "DX", "DY", "FWHM_X", "FWHM_Y", "SKY"]
    elif version==2:
        ParamName = ["TIME", "FLUX", "COMP_LC1", \
                     "COMP_LC2", "COMP_LC3", "AIRMASS", "FWHM", \
                     "DX", "DY", "FWHM_X", "FWHM_Y"]

    AllData = []

    for Counter,FileItem in enumerate(FileList):

        FitsFile = fits.open(FileItem, memmap='r')

        Time = FitsFile[1].data["JD-OBS"]

        #Generate the array to save the data
        CurrentData = np.zeros((len(Time), len(ParamName)))


        if version==1:
            CurrentData[:,0] = Time
            CurrentData[:,1] = FitsFile[20].data[0,:]
            CurrentData[:,2] = FitsFile[1].data['RA_MOVE']
            CurrentData[:,3] = FitsFile[1].data['DEC_MOVE']
            CurrentData[:,4] = FitsFile[1].data['PSF_A_1']
            CurrentData[:,5] = FitsFile[1].data['PSF_B_1']
            CurrentData[:,6] = FitsFile[1].data["AIRMASS"]


            input("Trying to figure out the content of the fits file...")

            plt.figure()
            plt.plot(CurrentData[:,0] , CurrentData[:,1], "ko")
            plt.plot(CurrentData[:,0] , FitsFile[24].data[0,:], "rd")
            #plt.plot(FitsFile[1].data['TMID'], CurrentData[:,1], "ro")
            plt.show()
        elif version ==2:
            CurrentData[:,0] = Time
            CurrentData[:,1] = FitsFile[3].data[:,0]
            CurrentData[:,2] = FitsFile[3].data[:,1]
            CurrentData[:,3] = FitsFile[3].data[:,2]
            CurrentData[:,4] = FitsFile[3].data[:,3]
            CurrentData[:,5] = FitsFile[1].data["AIRMASS"]
            CurrentData[:,6] = FitsFile[1].data["FWHM"]
            CurrentData[:,7] = FitsFile[1].data["RA_MOVE"]
            CurrentData[:,8] = FitsFile[1].data["DEC_MOVE"]
            CurrentData[:,9] = FitsFile[1].data["PSF_A_5"]
            CurrentData[:,10] = FitsFile[1].data["PSF_B_5"]

        AllData.extend(CurrentData)

    AllData = np.array(AllData)
    ParamName = np.array(ParamName)

    return ParamName, AllData


def TransitBoxModel(Time, T0=None, TDur=None, Delta=1):
    '''
    This function creates a box shaped transit

    Parameters:
    ============
    Time: numpy array
         Array of time vector for which the transit is to be evaluated

    T0: float
        The mid point of the transit in unit of Time

    TDur: float
          Transit Duration in days


    Returns
    ==========
    A vector of transit the same size as time


    '''
    TransitIndex = np.abs((Time-T0))<TDur/2
    TransitModel = np.zeros(len(Time))
    TransitModel[TransitIndex]-=Delta
    return TransitModel


def SVDSolver(A, b, T0, TDur, Combination):
    '''
    Returns the least square coefficients based on basis matrix
    using Singular Value Decomposition

    Parameters
    ----------
    A: (M,N) sized array which serves as the basis function

    Flux: N size array
        Flux series

    T0: The mid transit time

    TDur: The transit Duration

    Combination: The columns used for the getting the combination vector

    Returns
    --------
    array(M), array(M), float

    returns T0, TDur, Combination
    '''
    b = b.T
    N, M = np.shape(A)

    U,S,V = np.linalg.svd(A, full_matrices=False)

    S = np.diag(S)
    S[S==0] = 1.0e10
    W = 1./S

    CalcCoef = reduce(np.matmul,[U.T, b, W, V])
    Cov = reduce(np.matmul,[V.T,W*W,V])
    Residual = np.sum((np.matmul(A,CalcCoef)-b)**2.0)
    ChiSquaredReduced = Residual/(N-M)
    Cov = ChiSquaredReduced*Cov
    Uncertainty = np.sqrt(np.diag(Cov))

    Model = np.dot(A,CalcCoef)

    DetrendedCoef = np.copy(CalcCoef)
    DetrendedCoef[-2] = 0.0
    DetrendedModel = np.dot(A, DetrendedCoef)

    return CalcCoef, Uncertainty, Residual, Model, \
           DetrendedModel,T0, TDur, Combination


def SplineFlattening(Time, Flux, period, NIter = 4, StdCutOff=2.5, poly=3, knot=1):
    '''
    This fit a spline to the data
    '''
    TimeCopy = np.copy(Time)#[~OutliersIndex]
    FluxCopy = np.copy(Flux)#[~OutliersIndex]
    KnotSpacing = knot                          #increase ChunkSize or decrease ChunkSize
    PolyDeg = int(poly)
    for i in range(NIter):
        NumOrbits =  int((TimeCopy[-1]-TimeCopy[0])/period)
        if NumOrbits<1:
            NumOrbits=1
        ChunkSize = KnotSpacing*len(TimeCopy)/NumOrbits
        N = int(len(Time)/ChunkSize)
        Location = [int((i+0.5)*ChunkSize) for i in range(0,N)]
        knots = TimeCopy[Location]
        spl = spline(TimeCopy, FluxCopy, knots, k=PolyDeg)
        FluxPred = spl(TimeCopy)
        Residual = FluxCopy-FluxPred
        Std = np.std(Residual)
        GoodIndex = np.abs(Residual)<StdCutOff*Std
        TimeCopy = TimeCopy[GoodIndex]
        FluxCopy = FluxCopy[GoodIndex]

    FluxPred = spl(Time)
    return FluxPred


def GetID(Name, IdType=None):
    '''
    Method to get Speculoos ID/GAIA ID from viceversa
    '''

    #Loading the database
    Data = np.loadtxt("database/Targets.csv", delimiter=",", skiprows=1, dtype=np.str)
    SpName = Data[:,0]
    SpName = np.array([Item.upper() for Item in SpName])
    GaiaID = Data[:,2].astype(np.int)

    if "SPECULOOS" in IdType.upper():
        return GaiaID[SpName == Name.upper()][0]

    elif "GAIA" in IdType.upper():
        return SpName[GaiaID==int(Name)][0]
    else:
        return "Not Found"
