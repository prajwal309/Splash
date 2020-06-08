'''
This file contains the important function that is imported within the module
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
import corner
from time import time
import os
import glob
import batman


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



def ReadData(Location, TargetName):
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


def CreateOutputDir(FolderName=None):
    '''
    This function creates a unique ID and its corresponding dictionary.
    '''



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


def CreateCornerPlot(Parameters, ParametersName, OutputDir):
    #Generate a diagnostic corner plot
    plt.figure(figsize=(14,14), dpi=200)
    corner.corner(Parameters, labels=ParametersName, show_titles=True, quantiles=[0.158, 0.50, 0.842])
    plt.tight_layout()
    plt.tick_params(which="both", direction="in")
    plt.savefig(OutputDir+"/CornerPlot.png")
    plt.close('all')


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


def GeneratePdfReport(OutputDir, InputFileLocation):
    '''
    This function generates the output directory
    '''

    Global_TimeList = []
    Global_FluxList = []

    Global_TimeList_Binned = []
    Global_FluxList_Binned = []


    JD_UTC, Flux, _, _, _, _, _, _,	_, _, _= np.loadtxt(InputFileLocation, unpack=True)
    Diff1D = np.diff(JD_UTC)
    Index = np.concatenate((np.array([False]), Diff1D>0.25))
    Locations = np.where(Index)[0]

    ChunkCount = 0
    Start=0
    for Value in range(len(Locations)+1):
        if ChunkCount<len(Locations):
            Stop = Locations[ChunkCount]
        else:
            Stop = len(Flux)
        ChunkCount+=1

        SelectedTime = JD_UTC[Start:Stop]
        SelectedFlux = Flux[Start:Stop]

        #Sigma Clip
        OutliersIndex = SigmaClip(SelectedTime, SelectedFlux, SigmaValue=5.0)
        SelectedTime = SelectedTime[~OutliersIndex]
        SelectedFlux = SelectedFlux[~OutliersIndex]
        SelectedFlux -= np.mean(SelectedFlux)

        Global_TimeList.append(SelectedTime)
        Global_FluxList.append(SelectedFlux)

        NBins = int(len(SelectedTime)/16)
        TempBinnedTime = binned_statistic(SelectedTime, SelectedTime, statistic="median", bins=NBins)[0]
        TempBinnedFlux = binned_statistic(SelectedTime, SelectedFlux, statistic="median", bins=NBins)[0]

        Global_TimeList_Binned.append(TempBinnedTime)
        Global_FluxList_Binned.append(TempBinnedFlux)
        Start = Stop

    ModelFileList = np.array(glob.glob(OutputDir+"/PerNightAnalysisData/*.txt"))

    SavePdfLocation = OutputDir+"/PerNightPdfSummary"
    if not(os.path.exists(SavePdfLocation)):
        os.system("mkdir %s" %(SavePdfLocation.replace(" ", "\ ")))


    NightArray = np.array([int(Item.split("/")[-1][5:8]) for Item in ModelFileList])
    RankArray = np.array([int(Item.split("/")[-1][-7:-4]) for Item in ModelFileList])
    IndexOrder = np.argsort(NightArray*1000+RankArray)
    ModelFileList = ModelFileList[IndexOrder]
    NightArray = NightArray[IndexOrder]
    RankArray = RankArray[IndexOrder]

    NightCounter = 0

    FirstNight = True
    for FileCounter, N in enumerate(NightArray):

        if N!=NightCounter:
            NightCounter = N

            #Save the file
            if not(FirstNight):
                print("Saving it now")
                PdfFile.output(PdfFileName)
                print("New Summary File Generated in %s." %SavePdfLocation)

            #Number of Models
            NumModels = np.sum(NightArray==N)
            Height = 115+75*NumModels//2
            Width = 150

            PdfFileName = SavePdfLocation+"/"+"SummaryNight"+str(N).zfill(3)+".pdf"
            PdfFile = FPDF('P', 'mm', (Width,Height))
            PdfFile.add_page()

            #Load the model File
            ReducedChiSqrDataName = OutputDir+"/Data/Night"+str(N)+".csv"
            T0_Range, TransitDepth, Signal_Strength, ReducedChiSq, LocalSTD, Uncertainty, TDur = np.loadtxt(ReducedChiSqrDataName, delimiter=",", skiprows=1, unpack=True)

            #If range is present
            RangePresent = max(Uncertainty) - min(Uncertainty)>1e-5

            #Generate the first figure
            PlotX0 = int(min(Global_TimeList[N-1]))


            plt.figure()
            fig, ax = plt.subplots(figsize=(5.51181*2,2.75591*2), nrows=2, ncols=1)
            ax[0].plot(Global_TimeList[N-1] - PlotX0, Global_FluxList[N-1], "ko")
            ax[0].set_xticks([])
            ax[0].set_ylabel("Normalized Flux", fontsize=15)
            ax[0].tick_params(direction="in", which="both", length=7.5, width=2.5)

            if RangePresent:
                ax[1].errorbar(T0_Range - PlotX0, TransitDepth, yerr=Uncertainty, marker="d", markersize=3, color="navy", linestyle="None", elinewidth=2, capsize=3, ecolor="navy")
            else:
                ax[1].plot(T0_Range - PlotX0, TransitDepth, linestyle="None", marker="*")

            ax[1].set_xlabel("JD-%s" %(PlotX0), fontsize=15)
            ax[1].set_ylabel("Transit Depth", fontsize=15, color="navy")
            ax[1].spines["left"].set_color("orange")
            ax[1].tick_params(direction="in", which="both", length=7.5, width=2.5,color="navy")

            ax1_twin = ax[1].twinx()
            ax1_twin.plot(T0_Range - PlotX0, ReducedChiSq, linestyle="None", marker="*", color="red")
            ax1_twin.tick_params(direction="in", which="both", color="red")
            ax1_twin.set_ylabel("$\\chi^2_\\nu$", fontsize=15, color="red")
            ax1_twin.spines["right"].set_color("red")
            ax1_twin.tick_params(direction="in", which="both", length=7.5, width=2.5)

            SaveNameTemp = "Temp%s.png" %(str(N).zfill(3))
            plt.tick_params(direction="in", which="both")
            plt.tight_layout()
            plt.savefig(SaveNameTemp)

            PdfFile.set_font('Helvetica', 'B', 16)
            PdfFile.cell(140, 10, 'Night -'+str(N), align="C")
            PdfFile.image(SaveNameTemp, x=10, y=20, w=140, h=70)
            PdfFile.line(0, 90, 170, 90)
            PdfFile.line(0, 91, 170, 91)

            ModelCount = 1
            os.system("rm %s" %(SaveNameTemp))
            LocalSTD = np.mean(LocalSTD)

            FirstNight = False

        #Save the Model

        XLocation = 10 + (ModelCount%2==0)*65
        YLocation = 100 + (ModelCount-1)//2*65
        #print(N, ModelFileList[FileCounter])

        JD_UTC, TransitModel, BackgroundContinuum = np.loadtxt(ModelFileList[FileCounter], skiprows=1, delimiter=",", unpack=True)
        Model = TransitModel+BackgroundContinuum
        Model -= np.mean(Model)
        ModelSaveName = "Temp%s_%s.png" %(str(N).zfill(3), str(ModelCount).zfill(3))

        ReadTitleOnly = open(ModelFileList[FileCounter],'r').readline()
        TMidPoint = float(ReadTitleOnly.split("@")[-1])

        plt.figure(figsize=(6.5,6))
        plt.plot(Global_TimeList[N-1] - PlotX0, Global_FluxList[N-1], "ko", markersize=2)
        plt.plot(JD_UTC - PlotX0, TransitModel, color="navy", lw=3.5)
        plt.plot(JD_UTC - PlotX0, Model, "r-", lw=2)
        plt.errorbar(Global_TimeList_Binned[N-1]- PlotX0, Global_FluxList_Binned[N-1], linestyle="None", yerr=np.ones(len(Global_TimeList_Binned[N-1]))*LocalSTD/4.0, capsize=3, ecolor="green", marker="d", markersize=3, color="green")
        plt.axvline(x=TMidPoint-PlotX0, color="navy", linestyle=":", lw=2.5)
        TitleTextModel = "Promise Factor:"+str(RankArray[FileCounter])
        plt.tick_params(direction="in", which="both", length=7.5, width=2.5)
        plt.title(TitleTextModel)
        plt.tight_layout()
        plt.savefig(ModelSaveName)
        plt.close('all')

        #Add figure to the file
        PdfFile.image(ModelSaveName, x=XLocation, y=YLocation, w=65, h=60)


        #plt.plot(CurrentFile[])
        os.system("rm %s" %ModelSaveName)
        ModelCount+=1

    #For the last night that still needs to be saved
    PdfFile.output(PdfFileName)
