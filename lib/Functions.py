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

from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic


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

    FileList = glob.glob(Location+"/*%s.txt*" %TargetName)
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


def ReadFitsData(Location, TargetName):
    '''
    This function reads the input file from Cambridge Pipeline

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

    print
    FileList = glob.glob(Location+"/*%s*.fits*" %TargetName)
    NumFiles = len(FileList)

    if NumFiles == 0:
        raise NameError("No Files found")


    AllData = []

    ParamName = ["Time", "Flux", "CompLC1", "CompLC2", \
                 "CompLC2", "CompLC3", "Airmass", "FWHM", \
                 "RA_MOVE", "DEC_MOVE", "PSF_A", "PSF_B"]

    AllData = []

    for Counter,FileItem in enumerate(FileList):
        FitsFile = fits.open(FileItem, memmap='r')

        Time = FitsFile[1].data["JD-OBS"]

        #Generate the array to save the data
        CurrentData = np.zeros((len(Time), len(ParamName)))

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


def SigmaClip(Time, Flux, SigmaValue=3.0):
    '''
    This function performs the sigma clip
    #####################################
    Input Parameters
    =================
    Time: Time Array
    Flux: Normalized Flux Array. Note the trend in the data should be taken out.

    Returns
    ===================
    Returns the index
    '''
    Time = np.array(Time)
    Flux = np.array(Flux)

    STD = np.std(Flux)

    Index = np.abs(Flux-np.mean(Flux))>SigmaValue*STD
    return Index


def BoxFit(Time, T0=None, TDur=None, Delta=100):
    '''
    This function creates a box shaped transit

    Parameters:
    ============
    Time: Array of the time
    T0: The mid point of the transit in unit of Time
    TDur: Transit Duration in days
    '''
    TransitIndex = np.abs((Time-T0))<TDur/2
    TransitModel = np.zeros(len(Time))
    TransitModel[TransitIndex]-=Delta
    return TransitModel


def CustomSVDSolver(Basis, Flux):
    '''
    Matlab SVD function modified for python.
    '''
    A = np.copy(Basis)
    b = np.copy(Flux).T
    N, M = np.shape(A)

    U,S,V = np.linalg.svd(A, full_matrices=False)

    d = S
    S = np.diag(S)
    S[S==0] = 1.0e10
    W = 1./S

    CalcCoef = reduce(np.matmul,[U.T, b, W, V])
    Cov = reduce(np.matmul,[V.T,W*W,V])
    Residual = np.sum((np.matmul(A,CalcCoef)-b)**2.0)
    ChiSquaredReduced = Residual/(N-M)
    Cov = ChiSquaredReduced*Cov
    return CalcCoef, Cov, Residual


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
        return GaiaID[SpName == Name][0]

    elif "GAIA" in IdType.upper():
        return SpName[GaiaID==int(Name)][0]
    else:
        return "Not Found"




def InjectTransit(Time, Flux, FileLocation, Parameter="b"):
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
