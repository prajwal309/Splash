import numpy as np
import matplotlib.pyplot as plt

import glob
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import itertools
import os

import matplotlib as mpl
mpl.rc('font',**{'sans-serif':['Helvetica'], 'size':15,'weight':'bold'})
mpl.rc('axes',**{'labelweight':'bold', 'linewidth':1.5})
mpl.rc('ytick',**{'major.pad':22, 'color':'k'})
mpl.rc('xtick',**{'major.pad':10,})
mpl.rc('mathtext',**{'default':'regular','fontset':'cm','bf':'monospace:bold'})
mpl.rc('text', **{'usetex':True})
mpl.rc('text.latex',preamble=r'\usepackage{cmbright},\usepackage{relsize},'+r'\usepackage{upgreek}, \usepackage{amsmath}')
mpl.rc('contour', **{'negative_linestyle':'solid'})


def FindLocalMaxima(Data, NData=4):
    '''
    This function finds the value of the local maxima
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


def CalcLikelihood(ListTransitDepth, ListTDepthUncty, ListMetric):
    '''
    This function calculates the likelihood given set of transit, transit depth,
    transit uncertainty and the promise factor into a single likelihood value.
    '''
    #Convert list to Array
    ArrayTransitDepth = np.array(ListTransitDepth)
    ArrayTDepthUncty = np.array(ListTDepthUncty)
    ArrayMetric = np.array(ListMetric)

    #CalcTransitDepthUncty = 0.25*ArrayTransitDepth

    MeanTDepth = np.mean(ArrayTransitDepth)
    FirstTerm = np.sum(ArrayMetric)*3.0
    RelativeError = np.abs(MeanTDepth-ArrayTransitDepth)/ArrayTransitDepth*100.0
    #print("The relative error is given by:", RelativeError)

    SecondTerm = 1.0 #np.sum(RelativeError)**0.10
    return FirstTerm/(SecondTerm)


def fold_data(t,y, period):
  # simple module to fold data based on period

  folded = t%period
  inds = np.array(folded).argsort()
  t_folded = folded[inds]
  y_folded = y[inds]
  return t_folded, y_folded


def SearchPeriodicSignal(Name, OutputDir, MedianCutOff=2.5, NDataPoints=5):
    '''
    Function to search for the periodic signals in the likelihood function.
    #######################################################################
    Input
    =======================
    Name: Name of the folder under which results are saved
    OutputDir: Location of the folder
    NDataPoints: The number of points to be considered for the phase curves
                 to find the

    =======================
    '''

    #The list of the file is following
    FileLists = np.array(glob.glob(OutputDir+"/DailyModels/*.txt"))
    NightNumber = [int(Item.split("/")[-1][11:15]) for Item in FileLists]

    ArrangeIndex = np.argsort(NightNumber)
    FileLists = FileLists[ArrangeIndex]

    Time = []
    TransitDepth = []
    Uncertainty = []
    ChiSQR = []

    for File in FileLists:
        LocalTime, LocalTDepth, LocalTUncty, LocalChi = np.loadtxt(File, unpack=True)
        Time.extend(LocalTime)
        TransitDepth.extend(LocalTDepth)
        Uncertainty.extend(LocalTUncty)
        ChiSQR.extend(LocalChi/np.median(LocalChi))

    Time = np.array(Time)
    TransitDepth = np.array(TransitDepth)
    Uncertainty = np.array(Uncertainty)
    ChiSQR = np.array(ChiSQR)

    Metric = TransitDepth/Uncertainty*1./ChiSQR

    SNR = TransitDepth/Uncertainty

    #The metric has to be likelihood

    import matplotlib as mpl
    mpl.use('TkAgg')

    #Metric = SNR
    #Find the local extrema
    Peaks = argrelextrema(Metric, np.greater)[0]
    Peaks = FindLocalMaxima(Metric, NData=NDataPoints)

    CutOff = 2.0*np.median(Metric)
    MetricCutOff= np.copy(Metric)
    MetricCutOff[MetricCutOff<CutOff] = 0.0

    SelectPeakIndex = FindLocalMaxima(MetricCutOff)

    Potential_T0s = Time[SelectPeakIndex]
    MetricCutOff = MetricCutOff[SelectPeakIndex]
    #Arrange the metric according to their value

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(figsize=(24,16), nrows=4, ncols=1)
    ax0.plot(Time, TransitDepth, color="black", marker=".", linestyle="None")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Metric", fontsize=20)

    ax1.plot(Time, SNR, color="navy", marker=".", linestyle="None" )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SNR", fontsize=20)

    ax2.plot(Time, ChiSQR, color="red", marker=".", linestyle="None")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Chi squared Value ")

    ax3.plot(Time, Metric, color="orange", marker=".", linestyle="None")
    ax3.set_xlabel("Time", fontsize=20)
    ax3.set_ylabel("Metric", fontsize=20)

    plt.tight_layout()
    plt.tick_params(which="both", direction="in")
    plt.savefig("SearchPeriodicSignal.png")
    plt.close('all')

    ArrangeIndex = np.argsort(-MetricCutOff)

    Potential_T0s = Potential_T0s[ArrangeIndex]
    MetricCutOff = MetricCutOff[ArrangeIndex]

    Combination_T0 = list(itertools.combinations(Potential_T0s, 2))


    #Tolerance for looking for maxima for the periodicity
    TimeTolerance = 0.01

    PeriodListValues = []
    FPP_Values = []
    T0_Values = []


    for T1, T2 in Combination_T0:
        L_Period = abs(T1-T2)
        #print("The local period is::", L_Period)
        HarmonicNum = int(L_Period)



        for Num in range(1, HarmonicNum):
            CurrentPeriod = L_Period/Num
            PeriodListValues.append(CurrentPeriod)

            #In order to take into account
            T0 = T1
            while T0>min(Time):
                T0-=CurrentPeriod
            NumberTransits = T1-min(Time)


            #Find all the transits like events
            TransitEvents = np.abs(Time-T0+TimeTolerance/2.0)%CurrentPeriod<TimeTolerance

            #Find the number of clusters
            AllLocations = np.where(np.diff(Time[TransitEvents])>CurrentPeriod/1.5)[0]
            NumClusters = len(AllLocations)+1

            ListTransitDepth = []
            ListTDepthUncty = []
            ListTransitDur = []

            ListMetric = []

            for LocCounter in range(NumClusters):

                if LocCounter<NumClusters-1.01:
                    FirstCriterionLocation = Time<Time[TransitEvents][AllLocations[LocCounter]]+CurrentPeriod/10000.0
                else:
                    FirstCriterionLocation = Time>Time[TransitEvents][AllLocations[LocCounter-1]]+CurrentPeriod/10000.0

                CurrentSelectIndex = np.logical_and(TransitEvents, FirstCriterionLocation)

                CopiedMetric = np.copy(Metric)
                CopiedMetric[~CurrentSelectIndex] = 0.0

                CurrentMaximaIndex = np.argmax(CopiedMetric)

                ListMetric.append(Metric[CurrentMaximaIndex])
                ListTDepthUncty.append(Uncertainty[CurrentMaximaIndex])
                ListTransitDepth.append(TransitDepth[CurrentMaximaIndex])
                #ListTransitDur.append(Metric[CurrentMaximaIndex])


            CurrentLikelihood = CalcLikelihood(ListTransitDepth, ListTDepthUncty, ListMetric)
            FPP_Values.append(CurrentLikelihood)
            T0_Values.append(min([T1,T2]))


    FPP_Values =  np.array(FPP_Values)
    T0_Values = np.array(T0_Values)
    PeriodListValues = np.array(PeriodListValues)

    #Arrange the values
    ArrangeIndex = np.argsort(PeriodListValues)
    PeriodListValues = PeriodListValues[ArrangeIndex]
    T0_Values = T0_Values[ArrangeIndex]
    FPP_Values = FPP_Values[ArrangeIndex]

    #Find the most promising period
    BestIndex = np.argmax(FPP_Values)
    BestPeriod = PeriodListValues[BestIndex]


    #Read the phase coverage data
    PeriodPC, PhaseCoverage = np.loadtxt(os.path.join(OutputDir,"PhaseCoverage.txt"), skiprows=1, unpack=True)

    fig, ax1 = plt.subplots(figsize=(14,8))
    ax2= ax1.twinx()

    for i in range(0,4):
        if i == 0:
            ax1.axvline(x=0.5*BestPeriod, color="cyan", linestyle=":", lw=3, alpha=0.8, label="True Period")
        else:
            ax1.axvline(x=i*BestPeriod, color="cyan", linestyle=":", lw=3, alpha=0.8)
    ax1.plot(PeriodListValues, FPP_Values, "r-", lw=2)
    ax2.plot(PeriodPC, PhaseCoverage*100.0, color="green", alpha=0.8, lw=2.0, label="Phase Coverage")
    ax1.set_xlabel("Period (Days)", fontsize=20)
    ax2.set_ylabel("Phase Coverage (\%)", color="green", labelpad=3.0,fontsize=20, rotation=-90)
    ax1.set_ylabel("Signal Detection Efficiency", color="red", fontsize=20)
    MinXLim = max([min(PeriodListValues), min(PeriodPC)])
    MaxXLim = min([max(PeriodListValues), max(PeriodPC)])
    ax1.set_xlim([MinXLim, MaxXLim])
    ax1.text(0.98*MaxXLim,0.98*max(FPP_Values), "Best Period:"+str(round(BestPeriod,5)), horizontalalignment="right")
    ax1.set_ylim([0, 1.2*max(FPP_Values)])
    ax1.tick_params(which="both", direction="in", colors="red")
    ax2.tick_params(which="both", direction="in", colors="green")
    ax1.spines['left'].set_color('red')
    ax1.spines['right'].set_color('green')
    ax2.spines['right'].set_color('green')
    ax2.spines['left'].set_color('red')
    plt.tight_layout()
    SaveFigureName = os.path.join(OutputDir, Name+"_FPP.png")
    plt.savefig(SaveFigureName)
    plt.close('all')

    SaveName = os.path.join(OutputDir,Name+".pgram")
    print("Saving under::", SaveName)
    np.savetxt(SaveName, np.transpose((T0_Values, PeriodListValues, FPP_Values)), header="T0, Period, FPP")


def PhaseCoverage(Time, OutputDir, StepSize=0.05, PLow=0.30, PUp=25.0, NTransits=2, Tolerance=0.005):
    '''
    This function calculates the phase coverage of the data

    ################################
    Input Parameters:
    =================
    PLow: Lowest Period
    PUp: Largest Period
    StepSize: StepSize of the Period
    NTransits: The number of transits to be used with
    Tolerance is to see the phase coverage for different phases
    '''
    print("The Output directory is:", OutputDir)
    #Redefine maximum periodicity to be looked for phase coverage
    if (Time[-1]-Time[0])<PUp:
        PUp=(Time[-1]-Time[0])

    PeriodAll = np.arange(PLow, PUp, StepSize)
    PhaseCoverage = np.ones(len(PeriodAll))

    for Count, Period in enumerate(PeriodAll):
        #Period = 3.0

        Phase = Time%Period
        Phase = Phase[np.argsort(Phase)]
        Diff = np.diff(Phase)


        PhaseCoverage[Count] -= Phase[0]/Period
        PhaseCoverage[Count] -= (Period-Phase[-1])/Period

        Locations = np.where(Diff>0.005)[0]
        for Location in Locations:
            PhaseUncovered = Phase[Location+1]-Phase[Location]
            PhaseCoverage[Count] -= PhaseUncovered/Period


    plt.figure(figsize=(12,8))
    plt.plot(PeriodAll, PhaseCoverage*100, "k-", lw=3)
    plt.xlabel("Days", fontsize=25)
    plt.ylabel("Phase Covered(\%)", fontsize=25)
    plt.tick_params(which="both", direction="in")
    plt.savefig("PhaseCoverage.png")

    print(OutputDir)
    np.savetxt(os.path.join(OutputDir,"PhaseCoverage.txt"), np.transpose((PeriodAll, PhaseCoverage)), header="Period, Phase Coverage")
    print("In the phase coverage transit")
