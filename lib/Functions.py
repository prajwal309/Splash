'''
This file contains the important function that is imported within the module
'''

import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import glob
import batman
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
        StopIndex = counter+NData
        Index[counter] = Data[counter]>0.99*max(Data[StartIndex:StopIndex])
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
    This array of
    '''
    NumPoints = int(len(Time)/NumBins)

    CurrentSTD = []
    for i in range(NumBins):
        CurrentSTD.append(np.std(Flux[i*NumPoints:i*(NumPoints+1)])/(np.sqrt(NumPoints)))
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


def InjectTransit(Time, Flux, FileLocation, Parameter="b"):
    '''
    Injects the transits

    Parameters
    ============
    Flux

    Rp_Rs

    T0


    Specify the

    '''
    #Read the parameters from the file

    #remove the signal from the data
    os.system("rm data/*temp*")
    if "b" in Parameter:
        print("Injecting TRAPPIST-1b transits")
        InjectParams = open("InjectData/TRAPPIST_1b.ini").readlines()
    elif "c" in Parameter:
        print("Injecting TRAPPIST-1c transits")
        InjectParams = open("InjectData/TRAPPIST_1c.ini").readlines()
    elif "d" in Parameter:
        print("Injecting TRAPPIST-1d transits")
        InjectParams = open("InjectData/TRAPPIST_1d.ini").readlines()

    Period = float(InjectParams[0].split("#")[0].split(":")[1])
    T0 = float(InjectParams[1].split("#")[0].split(":")[1])
    Rp_Rs = float(InjectParams[2].split("#")[0].split(":")[1])
    a_Rs = float(InjectParams[3].split("#")[0].split(":")[1])
    b = float(InjectParams[4].split("#")[0].split(":")[1])
    u = float(InjectParams[5].split("#")[0].split(":")[1])

    Inc = np.rad2deg(np.arccos(b/a_Rs))

    # Evaluate a batman model
    paramsBatman = batman.TransitParams()
    paramsBatman.t0 = T0                        #time of conjunction. This offset is taken care when Phase is changed.
    paramsBatman.per = Period                   #orbital period
    paramsBatman.rp = Rp_Rs                     #planet radius (in units of stellar radii)
    paramsBatman.a = a_Rs                       #semi-major axis (in units of stellar radii)
    paramsBatman.inc = Inc                      #orbital inclination (in degrees)
    paramsBatman.ecc = 0                        #eccentricity
    paramsBatman.w = 90.0                       #longitude of periastron (in degrees)
    paramsBatman.limb_dark = "linear"           #limb darkening model
    paramsBatman.u = [u]                        #limb darkening parameters

    #Add the transit to all data
    mTransit = batman.TransitModel(paramsBatman, Time, supersample_factor = 15, exp_time = 30.0/(86400.0))#initializes model
    ModelTransitFlux =  (mTransit.light_curve(paramsBatman)-1.0)

    #Save the data night by night
    Diff1D = np.diff(Time)
    Index = np.concatenate((np.array([False]), Diff1D>0.25))
    Locations = np.where(Index)[0]
    Start = 0

    if not(os.path.exists("InjectedSignalsFigure")):
        os.system("mkdir InjectedSignalsFigure")
    os.system("rm InjectedSignalsFigure/*")
    for i in range(len(Locations)+1):
        Night = i+1
        if i<len(Locations):
            Stop = Locations[i]
        else:
            Stop = len(Flux)
        TimeChunk = Time[Start:Stop]
        FluxChunk = Flux[Start:Stop]
        #Generate the transit light curve now
        mTransit = batman.TransitModel(paramsBatman, TimeChunk, supersample_factor = 15, exp_time = 30.0/(86400.0))#initializes model
        ModelTransitFluxChunk =  (mTransit.light_curve(paramsBatman)-1.0)

        #Plot only if transit is present:

        if min(ModelTransitFluxChunk)<-1e-5:
            print("Saving Figure Corresponding to Injected Data for Night-", Night)

            T0Plot = int(min(TimeChunk))

            plt.figure(figsize=(20,5))
            plt.plot(TimeChunk - T0Plot, FluxChunk+ModelTransitFluxChunk - np.mean(FluxChunk), "ko", label="Data+Signal")
            plt.plot(TimeChunk - T0Plot, ModelTransitFluxChunk, "ro-", lw=2, label="Injected Signal")
            plt.xlabel("UTC Time")
            plt.ylabel("Normalized Flux")
            plt.legend()
            plt.title(str(T0Plot))
            plt.tight_layout()
            plt.savefig("InjectedSignalsFigure/InjectedSignal_Night"+str(Night).zfill(3)+".png")
            plt.close('all')
        Start=Stop

    #Create another temporary file
    JD_UTC, Flux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS, ExpTime = np.loadtxt(FileLocation, unpack=True)

    TempFileLocation = FileLocation.replace(".txt", "temp.txt")
    np.savetxt(TempFileLocation,np.transpose((JD_UTC, Flux+ModelTransitFlux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS, ExpTime)), header="JD_UTC, Flux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS, ExpTime")
    return Time, Flux + ModelTransitFlux, TempFileLocation


def NormalizeFlux(UTC_Time, Flux, OutputDir, FlattenMethod=0, Plot=False):
    '''
    1 Method ---> Divide by the median
    2 Method ---> Divide by mean
    3 Method ---> Spline flattening
    4 Method ---> Savitsky Golay Flattening
    '''
    #Find the segments in the data
    UTC_Time = np.array(UTC_Time)
    Flux = np.array(Flux)
    NormalizedFlux = np.zeros(len(Flux))-1.0

    Diff1D = np.diff(UTC_Time)
    Index = np.concatenate((np.array([False]), Diff1D>0.25))
    Locations = np.where(Index)[0]

    FigSaveLocation = OutputDir+"/global_LC.png"

    T0 = int(min(UTC_Time))

    plt.figure(figsize=(14,6))
    Start = 0
    for i in range(len(Locations)+1):
        if i<len(Locations):
            Stop = Locations[i]
        else:
            Stop = len(Flux)
        TimeChunk = UTC_Time[Start:Stop]
        FluxChunk = Flux[Start:Stop]


        if FlattenMethod==1:
            #Normalize them using bilinear method
            ModelFlux = np.ones(len(FluxChunk))*np.mean(FluxChunk)
        elif FlattenMethod==2:
            #Normalize them using bilinear method
            ModelFlux = np.ones(len(FluxChunk))*np.mean(FluxChunk)
        elif FlattenMethod==3:
            ModelFlux = SplineFlattening(TimeChunk, FluxChunk, 0.25) #Fit a spline of order 3 with knots every 0.1 day interval
        elif FlattenMethod==4:
            ModelFlux = savgol_filter(FluxChunk, 51, 1)
        else:
            print("*******************************************************")
            print("*Currently only following methods are available       *")
            print("*Method 1: Median Filtering                           *")
            print("*Method 2: Mean Filtering                             *")
            print("*Method 3: Spline Flattening                          *")
            print("*Method 4: Savitsky Golay Flattening                  *")
            print("*******************************************************")
            raise("Correct Method ")
        NormalizedFlux[Start:Stop] = FluxChunk/ModelFlux
        plt.plot(TimeChunk -T0, FluxChunk, marker="o", linestyle="None")
        plt.plot(TimeChunk - T0, ModelFlux, "k-", lw=2)
        Start=Stop
    plt.xlabel("JD Time -- "+str(T0), fontsize=20)
    plt.ylabel("Normalized Flux", fontsize=20)
    plt.tight_layout()
    SaveName = OutputDir+"/global_LC.png"
    plt.savefig(SaveName)
    plt.close('all')
    return NormalizedFlux
