import numpy as np
import matplotlib.pyplot as plt
import glob
from warnings import warn
import os
from scipy.stats import binned_statistic

from .sampler import BoxFit
from .TransitSearch import Find_Closest
from .Functions import SigmaClip
from .FinalAnalysis import TransitComparison


def ConstructModel(T04Plotting, Basis, Night, OutputDir, Count, RangePresent, PlotFileName, PromiseFactor, DataFileName):
    #Returns the related data
    InitialBasis = Basis
    Basis = np.array(Basis.split(","))

    #remove the \n part
    Basis[-1] = Basis[-1][-2]
    LENGTH = len(Basis)

    #Number basis are there
    NumBasis = 0

    for b in Basis[::-1]:
        try:
            float(b)
            break
        except:
            NumBasis+=1


    DEG_POLY = int((LENGTH-NumBasis-3)/(NumBasis)) -1
    BasisNames = Basis[::-1][:NumBasis][::-1]

    #Load the data from corresponding night
    JD_UTC, Flux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS, ExpTime = np.loadtxt(DataFileName, unpack=True)
    Diff1D = np.diff(JD_UTC)
    Index = np.concatenate((np.array([False]), Diff1D>0.25))
    Locations = np.where(Index)[0]

    #Select the corresponding data
    if Night==1:
        Start = 0
    else:
        Start = Locations[Night-2]

    try:
        Stop = Locations[Night-1]
    except:
        #Go till the last data point for the last night
        Stop = len(JD_UTC)

    JD_UTC_Selected = JD_UTC[Start:Stop]
    Flux_Selected = Flux[Start:Stop]
    Flux_Selected-=np.mean(Flux_Selected)

    #If sigma clip have to clip all data points
    OutliersIndex = SigmaClip(JD_UTC_Selected, Flux_Selected, SigmaValue=5.0)

    JD_UTC_Selected = JD_UTC_Selected[~OutliersIndex]
    Flux_Selected = Flux_Selected[~OutliersIndex]
    Err_Selected = Err[Start:Stop][~OutliersIndex]
    XShift_Selected = XShift[Start:Stop][~OutliersIndex]
    YShift_Selected = YShift[Start:Stop][~OutliersIndex]
    FWHM_X_Selected = FWHM_X[Start:Stop][~OutliersIndex]
    FWHM_Y_Selected = FWHM_Y[Start:Stop][~OutliersIndex]
    FWHM_Selected = FWHM[Start:Stop][~OutliersIndex]
    SKY_Selected = SKY[Start:Stop][~OutliersIndex]
    AIRMASS_Selected = AIRMASS[Start:Stop][~OutliersIndex]
    ExpTime_Selected = ExpTime[Start:Stop][~OutliersIndex]

    ##Construct the basis
    BasisMatrix = np.zeros((len(JD_UTC_Selected),NumBasis))

    #Reconstruct the basis vector
    for ModelCount,param in enumerate(BasisNames):
        if "T" in param.upper():
            BasisMatrix[:,ModelCount] = JD_UTC_Selected - np.mean(JD_UTC_Selected)
        elif "X"==param.upper():
            XShift_Assign = XShift_Selected -np.mean(XShift_Selected)
            BasisMatrix[:,ModelCount] = XShift_Assign/np.std(XShift_Assign)
        elif "Y" == param.upper():
            YShift_Assign = YShift_Selected -np.mean(YShift_Selected)
            BasisMatrix[:,ModelCount] = YShift_Assign/np.std(YShift_Assign)
        elif "D" == param.upper():
            XY_Assign = np.sqrt(XShift_Selected*XShift_Selected+YShift_Selected*YShift_Selected)
            XY_Assign-= np.mean(XY_Assign)
            BasisMatrix[:,ModelCount] = XY_Assign/np.std(XY_Assign)
        elif "F" in param.upper():
            FWHW_Assign = FWHM_Selected - np.mean(FWHM_Selected)
            BasisMatrix[:,ModelCount] = FWHW_Assign/np.std(FWHW_Assign)
        elif "S" in param.upper():
            SKY_Assign = SKY_Selected - np.mean(SKY_Selected)
            BasisMatrix[:,ModelCount] = SKY_Assign/np.std(SKY_Assign)
        elif "A" in param.upper():
            AIRMASS_Assign = AIRMASS_Selected - np.mean(AIRMASS_Selected)
            BasisMatrix[:,ModelCount] = AIRMASS_Assign/np.std(AIRMASS_Assign)
        else:
            continue
            raise NameError("The parameter to be considered is %s, and it could not be parsed." %(param))
    #Now generate the model
    Coeffs = Basis[:NumBasis*(DEG_POLY+1)].astype(np.float32)
    T0 = float(Basis[-3-NumBasis])
    TransitDur = float(Basis[-2-NumBasis])
    TransitDepth = float(Basis[-1-NumBasis])
    TransitModel = BoxFit(JD_UTC_Selected, T0=T0, TDur=TransitDur, Delta=TransitDepth)

    BackgroundContinuum = np.zeros(len(JD_UTC_Selected))

    for i in range(NumBasis):
        BackgroundContinuum += np.polyval(Coeffs[i*(DEG_POLY+1):(i+1)*(DEG_POLY+1)], BasisMatrix[:,i])

    Model = TransitModel+BackgroundContinuum
    Model -= np.mean(Model)

    SaveFolder = OutputDir+"/PromisingCases"
    if not(os.path.exists(SaveFolder)):
        os.system("mkdir %s" %(SaveFolder).replace(" ","\ "))


    #Read from the plot file list
    T0_RangePlot, TransitDepthPlot, SignalStrengthPlot, ReducedChiSqrPlot, LocalSTDPlot, UncertaintyPlot, TDuration = np.loadtxt(PlotFileName, delimiter=',', unpack=True)

    #Check if the right night is selected
    assert((T0>min(T0_RangePlot)-TransitDur) and T0<max(T0_RangePlot)+TransitDur), ("Wrong night selected")


    T0_Index = Find_Closest(T0_RangePlot, T04Plotting)
    BestIndex =  np.zeros(len(T0_RangePlot)).astype(np.bool)
    BestIndex[T0_Index] = True

    #Save to the file
    PerNightAnalysisFolder = OutputDir+"/PerNightAnalysisData"
    if not(os.path.exists(PerNightAnalysisFolder)):
        os.system("mkdir %s" %PerNightAnalysisFolder.replace(" ", "\ "))

    #NameFile
    #Now write the information to the file
    FluxFile = np.transpose(JD_UTC_Selected)
    NightString = "Night"+str(Night).zfill(3)

    ModelFileSaveName = "Night"+str(Night).zfill(3)+"_"+"RankIndex"+str(Count+1).zfill(3)+".txt"

    np.savetxt(PerNightAnalysisFolder+"/"+ModelFileSaveName, np.transpose((JD_UTC_Selected, TransitModel, BackgroundContinuum)), delimiter=",", header="JD_UTC, TransitModel, BackgroundContinuum @ "+str(T0))

    #Save the File for each night
    T0_Plot = int(min(JD_UTC_Selected))

    #Bin the Flux
    BinSize = 5.0 #minutes

    #The time differennce in  days
    TimeDifference = max(JD_UTC_Selected) - min(JD_UTC_Selected)
    NumBins = int(TimeDifference*86400.0/(BinSize*60.0))

    BinnedTime = binned_statistic(JD_UTC_Selected, JD_UTC_Selected, statistic='mean', bins=NumBins)[0]
    BinnedFlux = binned_statistic(JD_UTC_Selected, Flux_Selected, statistic='median', bins=NumBins)[0]


    STD = np.std(Flux_Selected-Model)
    NumInABin = len(JD_UTC_Selected)/len(BinnedTime)
    ScaledSTD = STD*1.0/np.sqrt(NumInABin)
    ErrorSTD = np.ones(len(BinnedTime))*ScaledSTD

    fig, ax = plt.subplots(2,2,figsize=(20,14), sharex=True)
    ax[0,0].plot(JD_UTC_Selected - T0_Plot, Flux_Selected, color="cyan", marker="o", linestyle="None", markersize=2, label="Data")
    ax[0,0].errorbar(BinnedTime-T0_Plot, BinnedFlux, yerr=ErrorSTD, zorder=5, color="black", ecolor="black", capsize=3, linestyle="None", marker="d")

    ax[0,0].plot(JD_UTC_Selected - T0_Plot, Model, "g-", lw=4, label="Combined Model")
    ax[0,0].plot(JD_UTC_Selected - T0_Plot, TransitModel, "r-", zorder=10, lw=4, label="Transit Model")
    ax[0,0].set_ylabel("Normalized Flux", fontsize = 20)
    ax[0,0].set_xlim(min(T0_RangePlot - T0_Plot), max(T0_RangePlot-T0_Plot))
    #ax[0,0].set_ylim(-0.01, 0.02)
    ax[0,0].tick_params(which='both', direction='in')
    ax[0,0].legend(loc=1)
    ax[0,0].set_title("Best $\\chi^2$ Model")

    #SNR
    if RangePresent:
        ax[0,1].plot(T0_RangePlot - T0_Plot, TransitDepthPlot/UncertaintyPlot, marker="3",  markersize=10, linestyle=':', color="navy")
        ax[0,1].plot(T0_RangePlot[BestIndex] - T0_Plot, TransitDepthPlot[BestIndex]/UncertaintyPlot[BestIndex], marker="3", markersize=15, linestyle='None', color="red", )
    else:
        ax[0,1].plot(T0_RangePlot - T0_Plot, SignalStrengthPlot/LocalSTDPlot, marker="3",  markersize=10, linestyle=':', color="navy", )
        ax[0,1].plot(T0_RangePlot[BestIndex] - T0_Plot, SignalStrengthPlot[BestIndex]/LocalSTDPlot[BestIndex], marker="3", markersize=10, linestyle='None', color="red", )

    ax[0,1].set_xlabel("Time (JD) --" +str(T0_Plot), fontsize = 20)
    ax[0,1].set_ylabel("SNR", fontsize = 20)
    ax[0,1].tick_params(which='both', direction='in')
    ax[0,1].set_xlim(min(T0_RangePlot - T0_Plot), max(T0_RangePlot-T0_Plot))

    #Transit Depth
    if RangePresent:
        ax[1,0].errorbar(T0_RangePlot - T0_Plot, TransitDepthPlot, yerr=UncertaintyPlot, marker="*", capsize=3, elinewidth=2, linestyle='None', color="navy", ecolor="navy")
        ax[1,0].errorbar(T0_RangePlot[BestIndex] - T0_Plot, TransitDepthPlot[BestIndex], yerr=UncertaintyPlot[BestIndex], marker="*", capsize=3, elinewidth=2, linestyle='None', color="red", ecolor="red", label="Best Model")
    else:
        ax[1,0].plot(T0_RangePlot - T0_Plot, TransitDepthPlot, marker="*", linestyle='None', color="navy")
        ax[1,0].plot(T0_RangePlot[BestIndex] - T0_Plot, TransitDepthPlot[BestIndex],  marker="*", linestyle='None', color="red", label="Best Model")
    ax[1,0].set_xlabel("Time (JD) --" +str(T0_Plot), fontsize = 20)
    ax[1,0].set_ylabel("Transit Depth", fontsize = 20)
    ax[1,0].tick_params(which='both', direction='in')
    ax[1,0].legend(loc=1)
    ax[1,0].set_xlim(min(T0_RangePlot - T0_Plot), max(T0_RangePlot-T0_Plot))

    #ChiSquare
    ax[1,1].plot(T0_RangePlot - T0_Plot, ReducedChiSqrPlot,  marker="3", markersize=10,  linestyle=':', color="navy", )
    ax[1,1].plot(T0_RangePlot[BestIndex] - T0_Plot, ReducedChiSqrPlot[BestIndex], marker="3", markersize=15, linestyle='None', color="red", label="Best Model")
    ax[1,1].set_xlabel("Time (JD) --" +str(T0_Plot), fontsize = 20)
    ax[1,1].set_ylabel("$\\chi^2_\\nu$", fontsize = 20)
    ax[1,1].tick_params(which='both', direction='in')
    ax[1,1].set_xlim(min(T0_RangePlot - T0_Plot), max(T0_RangePlot-T0_Plot))


    TitleText = "Night: "+str(Night)+"\n"+"T0: "+str(round(T0,5))+"\n TDur: "+str(round(TransitDur*24,2))+"\n Promise Value:"+str(round(PromiseFactor,2))
    plt.suptitle(TitleText, fontsize=14)
    plt.subplots_adjust(hspace=0, wspace=0.3)
    #plt.show()
    plt.savefig(SaveFolder+"/"+str(Night).zfill(4)+"Night_PromisingIndex"+"_"+str(Count+1)+".png")
    plt.close('all')

    return True, JD_UTC_Selected, Flux_Selected, TransitModel, Model



def Figure4PromisingCases(OutputDir, DataFileName, AnalysisFlag):
    '''
    This function will take make promising figures from the overall data.
    '''
    #Arrange the function in the ascending order
    ParamFileLists = np.array(glob.glob(OutputDir+"/Data/*.param"))
    ParamNightList = np.array([int(File.split("/")[-1].split(".")[0][5:]) for File in ParamFileLists]).astype(np.int32)
    ParamFileLists = ParamFileLists[np.argsort(ParamNightList)]
    ParamNightList = ParamNightList[np.argsort(ParamNightList)]

    PlotFileLists = np.array(glob.glob(OutputDir+"/Data/*.csv"))
    PlotNightList = np.array([int(File.split("/")[-1].split(".")[0][5:]) for File in PlotFileLists]).astype(np.int32)
    PlotFileLists = PlotFileLists[np.argsort(PlotNightList)]
    PlotNightList = PlotNightList[np.argsort(PlotNightList)]

    assert len(PlotFileLists)>0, ("No files are available for the specific run.")
    #Read all the basis vectors
    BasisValues = []
    NightIndicator = []

    #Read the parameters for the best fit parameter

    for Num, FileName in enumerate(ParamFileLists):
        DataLoadtxt = open(FileName,'r+').readlines()
        for Item in DataLoadtxt:
            BasisValues.append(Item)
        M = len(DataLoadtxt)
        NightIndicator.extend(np.ones(M)*(Num+1))


    BasisValues = np.array(BasisValues)
    NightIndicator = np.array(NightIndicator)

    #Read all the values for the plot file
    #Initializing the lists
    ParamValues = []
    LocalSignificance = []

    for File in PlotFileLists:
        # T0_Range, TransitDepth, Signal Strength, Reduced Chi Square Value, Local STD Array, Uncertainty in Transit Depth, Transit Duration
        DataLoadtxt = np.loadtxt(File, skiprows=1, delimiter=",")
        ParamValues.extend(DataLoadtxt)
        TempReducedChiSq = DataLoadtxt[:,3]
        NormalizedLocalSig = (TempReducedChiSq)/np.median(TempReducedChiSq)
        #NormalizedLocalSig = np.median(TempReducedChiSq) - TempReducedChiSq
        LocalSignificance.extend(NormalizedLocalSig)


    ParamValues = np.array(ParamValues)
    LocalSignificance = np.array(LocalSignificance)

    #Calculate how promising is a signal
    T0_Range = ParamValues[:,0]
    TransitDepth = ParamValues[:,1]
    SignalStrength = ParamValues[:,2]
    ReducedChiSqr = ParamValues[:,3]
    LocalSTD = ParamValues[:,4]
    TransitDepthUncertainty = ParamValues[:,5]
    TDuration = ParamValues[:,6]


    #Finding the most promising case based on the transit depth, transit uncertainty, and decrease in the local chisquared value
    RangePresent = (max(TransitDepthUncertainty) - min(TransitDepthUncertainty))>1e-6

    if RangePresent:
        SNRArray = TransitDepth/TransitDepthUncertainty
    else:
        SNRArray = np.array(SignalStrength)


    #Calculate the Promise FactorArrangedBasisValues
    PromiseFactor = (SNRArray/(LocalSignificance)**2.0)

    PromiseIndex = np.argsort(PromiseFactor)[::-1]


    #Arrange by promise index
    PromiseFactor = PromiseFactor[PromiseIndex]

    ArrangedBasisValues = BasisValues[PromiseIndex]
    ArrangedT0_Range = T0_Range[PromiseIndex]
    Arranged_TDepth = TransitDepth[PromiseIndex]
    Arranged_TDur = TDuration[PromiseIndex]
    Arranged_TDepthUncty = TransitDepthUncertainty[PromiseIndex]
    ArrangedNightIndicator = NightIndicator[PromiseIndex].astype(np.int32)

    #Now look for the best case scenario. Reject cases in the 30 minute neighborhood of best cases
    counter = 0
    LoopCounter = -1


    SearchContent = open("SearchParams.ini", "r+").readlines()
    NumPromisingCases = int(SearchContent[11].split("#")[0].split(":")[1].replace(" ",""))

    #Now look for the similarity cases
    T0_BestCases = []
    TDepth_BestCases = []
    TDur_BestCases = []
    TDepUCertainty_BestCases = []
    PromiseValue_BestCases = []

    Time_BestCases = []
    Flux_BestCases = []
    TransitModel_BestCases = []
    Model_BestCases = []


    #Find the best case scenarios
    while counter<NumPromisingCases and LoopCounter<len(Arranged_TDepth):     #Note loop counter starts from -1
        LoopCounter+=1
        #Check if files are present
        if not(LoopCounter<len(PromiseIndex)):
            break
        Test_T0Flag = True
        for T0_Test in T0_BestCases:
            #Should be at least 2 hours aways the duration
            if np.abs(ArrangedT0_Range[LoopCounter] - T0_Test)<0.75/24.: #(Reject anything within 0.75 hours)
                Test_T0Flag = False
                continue

        if not(Test_T0Flag):
            continue

        Success, TimeValues, FluxValues, TransitModelValues, ModelValues = ConstructModel(ArrangedT0_Range[LoopCounter], ArrangedBasisValues[LoopCounter], ArrangedNightIndicator[LoopCounter], OutputDir, counter, RangePresent, PlotFileLists[ArrangedNightIndicator[LoopCounter]-1], PromiseFactor[LoopCounter], DataFileName)

        if Success:
            T0_BestCases.append(ArrangedT0_Range[LoopCounter])
            TDepth_BestCases.append(Arranged_TDepth[LoopCounter])
            TDur_BestCases.append(Arranged_TDur[LoopCounter])
            TDepUCertainty_BestCases.append(Arranged_TDepthUncty[LoopCounter])
            PromiseValue_BestCases.append(PromiseFactor[LoopCounter])

            Time_BestCases.append(TimeValues)
            Flux_BestCases.append(FluxValues)
            Model_BestCases.append(ModelValues)
            TransitModel_BestCases.append(TransitModelValues)

            counter+=1
        else:
            pass

    #Save the promising cases for the planet

    #Now performing the comparison of the transit
    if AnalysisFlag:
        TransitComparison(DataFileName, OutputDir,T0_BestCases,TDepth_BestCases,TDur_BestCases,TDepUCertainty_BestCases, PromiseValue_BestCases, Time_BestCases, Flux_BestCases, Model_BestCases, TransitModel_BestCases)
        print("Highlighted Best Case Scenarios")


def GlobalPeriodogram(OutputDir, FancyPlot=True):
    '''
    This function performs a analysis of all the time series data and build diagnostic plot
    '''

    FileLists = np.array(glob.glob(OutputDir+"/Data/*.csv"))

    #Arranging the files in the order of time
    NightList = np.array([int(File.split("/")[-1].split(".")[0][5:]) for File in FileLists]).astype(np.int32)
    FileLists = FileLists[np.argsort(NightList)]

    if len(FileLists)<1:
        warn("No files available for the analysis.")
        return 0

    #Initializing the lists
    T0_Range_List = []
    TransitDepth_List = []
    Signal_Strength_List = []
    ReducedChi_Square_List = []
    StdLocal_List = []
    TDepth_Uncty_List = []

    for File in FileLists:
        DataLoadtxt = np.loadtxt(File, skiprows=1, delimiter=",")
        T0_Range_List.extend(DataLoadtxt[:,0])
        TransitDepth_List.extend(DataLoadtxt[:,1])
        Signal_Strength_List.extend(DataLoadtxt[:,2])
        ReducedChi_Square_List.extend(DataLoadtxt[:,3])
        StdLocal_List.extend(DataLoadtxt[:,4])
        TDepth_Uncty_List.extend(DataLoadtxt[:,5])

    T0_Range_Array = np.array(T0_Range_List)

    RangePresent = (max(TDepth_Uncty_List) - min(TDepth_Uncty_List))>1e-6

    if FancyPlot:
        import matplotlib as mpl
        mpl.rc('font',**{'family':'sans-serif', 'serif':['Computer Modern Serif'],'sans-serif':['Helvetica'], 'size':20,'weight':'bold'})
        mpl.rc('axes',**{'labelweight':'bold', 'linewidth':1.5})
        mpl.rc('ytick',**{'major.pad':22, 'color':'k'})
        mpl.rc('xtick',**{'major.pad':10,})
        mpl.rc('mathtext',**{'default':'regular','fontset':'cm','bf':'monospace:bold'})
        mpl.rc('text', **{'usetex':True})
        mpl.rc('text.latex',preamble=r'\usepackage{cmbright},\usepackage{relsize},'+r'\usepackage{upgreek}, \usepackage{amsmath}')
        mpl.rc('contour', **{'negative_linestyle':'solid'})


    T0 = int(min(T0_Range_List))


    SaveName = OutputDir+"/"+"0.GlobalAnalysis.png"

    if RangePresent:
        plt.figure(figsize=(14,10))
        plt.subplot(311)
        plt.plot(T0_Range_Array - T0, Signal_Strength_List, "ko-")
        plt.ylabel("Signal Strength")
        plt.xticks([])
        plt.subplot(312)
        plt.plot(T0_Range_Array - T0, np.array(TransitDepth_List)/np.array(TDepth_Uncty_List), "go-")
        plt.ylabel("$\\chi^2_\\nu$", fontsize=20)
        plt.xticks([])
        plt.subplot(313)
        plt.plot(T0_Range_Array - T0, ReducedChi_Square_List, "ro-")
        plt.ylabel("SNR ", fontsize=20)
        plt.xlabel("Time (JD) - "+str(T0), fontsize=20)
        plt.subplots_adjust(hspace=0)
        plt.tick_params(which='both', direction='in', length=7, width=2)
        plt.savefig(SaveName)
        plt.close('all')

    else:

        #Calculate SNR in a more robust way
        print("Range is not present in Transit Depth Uncertainty")
        plt.figure(figsize=(16,10))
        plt.subplot(311)
        plt.plot(T0_Range_Array - T0, TransitDepth_List, "ko--")
        plt.ylabel("Transit Depth")
        plt.xticks([])
        plt.tick_params(which='both', direction='in', length=7, width=2)
        plt.subplot(312)
        plt.plot(T0_Range_Array - T0, np.array(TransitDepth_List)/np.array(TDepth_Uncty_List)*1000.0, "go--")
        plt.xlabel("Time (JD) - "+str(T0), fontsize=20)
        plt.ylabel("SNR", fontsize=20)
        plt.subplot(313)
        plt.plot(T0_Range_Array - T0, np.array(Signal_Strength_List)/np.array(StdLocal_List), "go--")
        plt.ylabel("$\\chi^2_\\nu$", fontsize=20)
        plt.subplots_adjust(hspace=0, wspace=0.1)
        plt.tick_params(which='both', direction='in', length=10, width=2)
        plt.savefig(SaveName)
        plt.close('all')
