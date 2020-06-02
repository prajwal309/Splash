# -*- coding: utf-8 -*-
import numpy as np
import emcee
import multiprocessing as mp
from scipy.optimize import minimize, leastsq
import matplotlib.pyplot as plt
from functools import reduce
from warnings import warn
import itertools
from scipy.stats import binned_statistic

import matplotlib as mpl
mpl.use('Agg')


def BoxFit(Time, T0=None, TDur=None, Delta=100):
    TransitIndex = np.abs((Time-T0))<TDur/2
    TransitModel = np.zeros(len(Time))
    TransitModel[TransitIndex]-=Delta
    return TransitModel


def FitFunction(CBVs,TimeChunk, FluxChunk, FitVariables, PolynomialOrder, LengthModelParam, TStep, T_Init):
    '''
    Function to be used in conjunction with the fit using scipy.optimize ---> minimize function
    '''
    BackgroundContinuum = np.zeros(len(TimeChunk))
    BackgroundContinuum += np.sum(np.array([np.polyval(CBVs[i*(PolynomialOrder+1):(i+1)*(PolynomialOrder+1)], FitVariables[:,i]) for i in range(LengthModelParam)]), axis=0)
    T0_Value = CBVs[-3]
    TDur_Value = CBVs[-2]
    Delta_Value = CBVs[-1]

    TransitModel = BoxFit(TimeChunk, T0=T0_Value, TDur=TDur_Value, Delta=Delta_Value)
    Model = TransitModel+BackgroundContinuum
    Residual = np.sum((FluxChunk - Model)*(FluxChunk - Model))
    return Residual


def SVD_Solver(TimeChunk, FluxChunk, FitVariables, PolynomialOrder, LengthModelParam, T0_Range, TDurArray):
    '''
    This function uses SVD to find the least square fit using the variables
    ######################################################################
    Input ParametersName
    ====================
    TimeChunk: The chunk of the data to be fitted
    FluxChunk: The chunk of the flux to be fitted
    FitVariables: The variables that are to be fitted.
    PolynomialOrder: The order to be used for the parameter
    LengthModelParam: The length of the parameters
    T0_Range: The range of T0 values
    TDur: The length of T0 Duration values
    '''

    print("Inside the SVD solver")
    #Generate a new fit variables matrix
    CBVMatrix = np.zeros((len(TimeChunk), (PolynomialOrder+1)*(LengthModelParam)+1))


    for OuterCounter in range(LengthModelParam):
        RowArray = FitVariables[:,OuterCounter]
        TempArray = np.ones(len(TimeChunk))
        for InnerCounter in range(PolynomialOrder+1):
            Index = OuterCounter*(PolynomialOrder+1)+InnerCounter
            CBVMatrix[:,Index]=TempArray
            TempArray=TempArray*RowArray

    UnnecessaryDCIndex = np.where(np.arange((PolynomialOrder+1)*(LengthModelParam)+1)%(PolynomialOrder+1)==0)[0][1:-1]
    CBVMatrix[:,UnnecessaryDCIndex]=0.0

    BestParameters = np.ones(LengthModelParam*(PolynomialOrder+1)+3)*10
    BestResidual = np.inf
    Uncertainty = np.inf


    ResidualArray = np.zeros((len(T0_Range), len(TDurArray)))
    UncertaintyArray = np.zeros((len(T0_Range), len(TDurArray)))
    CoeffMatrix = np.zeros((10, len(T0_Range), len(TDurArray)))

    for T0_Counter, T0_Local in enumerate(T0_Range):
        for TDur_Counter, TDur_Local in enumerate(TDurArray):

            TransitModel = BoxFit(TimeChunk, T0=T0_Local, TDur=TDur_Local, Delta=1e-3)
            CBVMatrix[:, -1] = TransitModel

            #Custom SVD
            CalcCoef, Cov, Residual = CustomSVDSolver(CBVMatrix, FluxChunk)

            if BestResidual>Residual:
                BestResidual = Residual
                Uncertainty = np.sqrt(Cov[-1][-1])/1000.0

                Model = np.matmul(CBVMatrix,CalcCoef)

                #Construct the Best Parameters
                BestParameters = np.zeros(LengthModelParam*(PolynomialOrder+1)+3)
                CalcCoef[UnnecessaryDCIndex]=0.0
                #Reverse the storage:
                for i in range(LengthModelParam):
                        BestParameters[i*(PolynomialOrder+1):(i+1)*(PolynomialOrder+1)]=CalcCoef[i*(PolynomialOrder+1):(i+1)*(PolynomialOrder+1)][::-1]
                BestParameters[-3] = T0_Local
                BestParameters[-2] = TDur_Local
                BestParameters[-1] = CalcCoef[-1]*1e-3
                BestModel = np.dot(CalcCoef, CBVMatrix.T)

    #Looks at the figure
    plt.figure()
    plt.imshow(ResidualArray, cmap="viridis")
    plt.show()

    return BestResidual, BestParameters, Uncertainty



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


def FinalMCMC_Prior(Theta, Time):

    global T0_Original, Period_Original, Mean_TDur, Mean_TDepth, Diff_TDepth, Diff_TDur

    #Bound T0 to certain values ...
    T0, TDur, Period, TDepth = Theta[-4:]

    if max(np.abs(Theta[:-4]))>10.0:
        return -np.inf

    #Sometimes large TTV means using a strict value is not preferred.
    #Four hours of deviation allowed possible
    if abs(Period-Period_Original)>4.0/24.0:
        return -np.inf

    if abs(T0-T0_Original)>4.0/24.0:
        return -np.inf
    #Less than 8 minutes or larger than 1.5 hours are not to be considered
    if TDur<8.0/(24.0*60.) or TDur>1.5/(24.):
        return -np.inf
    #if abs(TDepth-Mean_TDepth)>(3.0*Diff_TDepth):
    #    return -np.inf
    if TDepth<0.0:
        return -np.inf
    return 0

def CurrentBoxFit(Time, T0=None, TDur=None, Delta=None):
    TransitIndex = np.abs((Time))<TDur
    TransitModel = np.zeros(len(Time))
    TransitModel[TransitIndex]-=Delta
    return TransitModel


def Final_LogLikelihood(Theta, Time, Flux, AllBasisVectors, Locations):

    global LeastChiSq, SaveNum
    T0, TDur, Period, TDepth = Theta[-4:]

    #Evaluate the transit model
    TransitModel  = CurrentBoxFit((Time-T0+TDur/2)%Period, T0=0.0, TDur=TDur, Delta=TDepth)


    NumberNights = len(Locations)+1
    BackgroundContinuum = np.zeros(len(Time))

    Start = 0
    StartCoeffs = 0
    for NightCount in range(NumberNights):
        if NightCount<len(Locations):
            Stop = Locations[NightCount]
        else:
            Stop = len(Flux)

        #Number of Basis is given by:
        IsThereZero = np.all(AllBasisVectors[NightCount][1]==0)

        if IsThereZero:
            NumBasis = 1
        else:
            NumBasis = 2

        CurrentBkg = np.zeros(len(Time[Start:Stop]))
        #1. Find the relevant coefficients
        if NumBasis==1:
            RelevantCoeffs = Theta[StartCoeffs:StartCoeffs+2]
            CurrentBkg += np.polyval(RelevantCoeffs,AllBasisVectors[NightCount][0][Start:Stop])
            StartCoeffs+=2
        if NumBasis==2:
            for SubBasis in range(2):
                RelevantCoeffs = Theta[StartCoeffs+SubBasis*2:StartCoeffs+SubBasis*2+2]
                CurrentBkg += np.polyval(RelevantCoeffs+[0.0], AllBasisVectors[NightCount][SubBasis][Start:Stop])

            StartCoeffs+=4 #when two variables are selected then there are five coefficients.

        #This offset does not include the
        CurrentOffset = np.mean(Flux[Start:Stop])-np.mean(CurrentBkg+TransitModel[Start:Stop])
        CurrentBkg+=CurrentOffset

        #Now construct the Background for each night
        BackgroundContinuum[Start:Stop] += CurrentBkg
        Start = Stop

    AllModel = TransitModel+BackgroundContinuum


    Residual = Flux - AllModel
    ChiSquare = np.sum(Residual**2/(0.5*0.007**2))

    if ChiSquare<LeastChiSq:
        FileName = "temp/CurrentRunCoeff"+str(SaveNum)+".dat"
        with open(FileName, 'wb') as f:
            np.savetxt(f, Theta)
        LeastChiSq = ChiSquare
        #print("The least chisquare value is given by::", int(LeastChiSq))
        #Then make the figure

        '''Start = 0
        for NightCount in range(NumberNights):
            if NightCount<len(Locations):
                Stop = Locations[NightCount]
            else:
                Stop = len(Flux)

            plt.figure(figsize=(12,6*(NumberNights)))
            T0_String =  str(int(min(Time[Start:Stop])))
            plt.plot(Time[Start:Stop], Flux[Start:Stop], "ko")
            plt.plot(Time[Start:Stop], TransitModel[Start:Stop], "r-")
            plt.plot(Time[Start:Stop], BackgroundContinuum[Start:Stop], color="green", lw=3)
            plt.title(T0_String)
            plt.tight_layout()
            plt.savefig("BestFigure%s.png" %str(NightCount).zfill(3))
            plt.close('all')
            Start = Stop'''

    return -ChiSquare


def Final_MCMCPosterior(Theta, Time, Flux, AllBasisVectors, Locations):
    Value = FinalMCMC_Prior(Theta, Time)
    if Value==0:
        Likelihood = Final_LogLikelihood(Theta, Time, Flux, AllBasisVectors, Locations)
        return Likelihood
    else:
        return -np.inf


def MCMC_FinalFit(UniqueSaveNum, Time, Flux, AllBasisVectors, T0, Period, TDur1, TDur2, TDepth1, TDepth2, NumRuns, PolyNDim):
    '''
    This function performs MCMC fit


    #Input....
    '''
    global T0_Original, Period_Original, Mean_TDur, Mean_TDepth, Diff_TDepth, Diff_TDur, LeastChiSq, SaveNum
    SaveNum = UniqueSaveNum

    T0_Original = T0
    Period_Original = Period

    Mean_TDur = (TDur1+TDur2)/2.0
    Diff_TDur = np.abs(TDur1-TDur2)
    Mean_TDepth = (TDepth1+TDepth2)/2.0
    Diff_TDepth = np.abs((TDepth1-TDepth2))

    LeastChiSq = np.inf

    NDim = PolyNDim+4
    NWalkers = 10*(PolyNDim+4)

    #Find the Location for each nights
    Diff1D = np.diff(Time)
    Index = np.concatenate((np.array([False]), Diff1D>0.25))
    Locations = np.where(Index)[0]

    MeanTDur = (TDur1+TDur2)/2
    MeanTDepth = (TDepth1+TDepth2)/2

    PolyCoeff_Init = np.random.normal(0,0.05,(NWalkers, PolyNDim))

    #Start to
    T0_Init = np.random.normal(T0, 0.005, (NWalkers,1))
    TDur_Init = np.random.normal(Mean_TDur,2./(24.*60.), (NWalkers,1))
    Period_Init = np.random.normal(Period, 2e-4, (NWalkers,1))
    Delta_Init = np.abs(np.random.normal(MeanTDepth,0.05*MeanTDepth,(NWalkers,1)))

    #Stacking for the initial guesses
    StartingGuesses = np.hstack((PolyCoeff_Init, T0_Init, TDur_Init, Period_Init, Delta_Init))

    #start the basis vectors
    sampler = emcee.EnsembleSampler(NWalkers, NDim, Final_MCMCPosterior, args=(Time, Flux, AllBasisVectors, Locations), threads=1)
    pos, prob, state = sampler.run_mcmc(StartingGuesses, NumRuns)

    #Read the Best Parameters
    Theta = np.loadtxt("temp/CurrentRunCoeff"+str(UniqueSaveNum)+".dat")
    T0, TDur, Period, TDepth = Theta[-4:]

    BestTDepth = TDepth
    BestPeriod = Period
    BestT0 = T0
    BestTDur = TDur

    #Evaluate the model
    TransitModel  = CurrentBoxFit((Time-T0+TDur/2)%Period, T0=0.0, TDur=TDur, Delta=TDepth)


    NumberNights = len(Locations)+1
    BackgroundContinuum = np.zeros(len(Time))

    Start = 0
    StartCoeffs = 0
    for NightCount in range(NumberNights):

        if NightCount<len(Locations):
            Stop = Locations[NightCount]
        else:
            Stop = len(Flux)

        #Number of Basis is given by:
        IsThereZero = np.all(AllBasisVectors[NightCount][1]==0)
        if IsThereZero:
            NumBasis = 1
        else:
            NumBasis = 2

        CurrentBkg = np.zeros(len(Time[Start:Stop]))
        #1. Find the relevant coefficients
        if NumBasis==1:
            RelevantCoeffs = Theta[StartCoeffs:StartCoeffs+2]
            CurrentBkg += np.polyval(RelevantCoeffs,AllBasisVectors[NightCount][0][Start:Stop])
            StartCoeffs+=2
        if NumBasis==2:
            for SubBasis in range(2):
                RelevantCoeffs = Theta[StartCoeffs+SubBasis*2:StartCoeffs+SubBasis*2+2]
                CurrentBkg += np.polyval(RelevantCoeffs+[0.0], AllBasisVectors[NightCount][SubBasis][Start:Stop])

            StartCoeffs+=4 #when two variables are selected then there are five coefficients.

        CurrentOffset = np.mean(Flux[Start:Stop])-np.mean(CurrentBkg+TransitModel[Start:Stop])
        CurrentBkg+=CurrentOffset
        BackgroundContinuum[Start:Stop] += CurrentBkg
        Start=Stop


    AllModel =  BackgroundContinuum+TransitModel

    Residual = Flux - AllModel
    STD = np.std(Residual)
    ResidualSum = np.sum(Residual**2)


    #Estimate TDepth_STD
    TDepthValues = sampler.chain[:,-100:,-1]
    TDepth_STD = np.std(TDepthValues)

    #Save figure for each night
    Start = 0

    for NightCount in range(NumberNights):
        if NightCount<len(Locations):
            Stop = Locations[NightCount]
        else:
            Stop = len(Flux)

        T0_String =  str(int(min(Time[Start:Stop])))

        T0_Min = int(min(Time[Start:Stop]))

        #Bin the data
        PlotTime = Time[Start:Stop]
        PlotFlux = Flux[Start:Stop]

        #Binning size
        BinSize = 5.0 #minutes

        #The time differennce in  days
        TimeDifference = max(PlotTime) - min(PlotTime)

        NumBins = int(TimeDifference*86400.0/(BinSize*60.0))

        BinnedTime = binned_statistic(PlotTime, PlotTime, statistic='mean', bins=NumBins)[0]
        BinnedFlux = binned_statistic(PlotTime, PlotFlux, statistic='median', bins=NumBins)[0]

        NumInABin = len(PlotTime)/len(BinnedTime)


        ScaledSTD = STD*1.0/np.sqrt(NumInABin)
        ErrorSTD = np.ones(len(BinnedTime))*ScaledSTD

        T0_Values = np.arange(-300,301)*Period+T0
        MidTimeValue = np.mean(PlotTime)
        PlotT0 = T0_Values[np.argmin(np.abs(MidTimeValue-T0_Values))]

        plt.figure(figsize=(14,6))
        plt.plot(PlotTime-T0_Min, PlotFlux, color="cyan",marker="o", markersize=3, linestyle="None", alpha=0.8)
        plt.plot(PlotTime-T0_Min, TransitModel[Start:Stop], "r-")
        plt.plot(PlotTime-T0_Min, AllModel[Start:Stop], color="green", lw=3)
        plt.errorbar(BinnedTime-T0_Min, BinnedFlux, yerr=ErrorSTD, zorder=10, color="black", ecolor="black", capsize=3, linestyle="None", marker="d")
        plt.axvline(x=PlotT0 - T0_Min, color="orange", lw=3)
        plt.title(T0_String, fontsize=25)
        plt.xlabel("JD-Time")
        plt.ylabel("Normalized Flux")
        plt.tight_layout()

        SaveName = "temp/"+str(UniqueSaveNum)+"Figure"+str(NightCount).zfill(3)
        plt.savefig(SaveName)
        plt.close('all')
        Start = Stop


    return BestTDepth, TDepth_STD, BestPeriod, BestT0, BestTDur, ResidualSum, STD


#Now fit without transit case

def Final_LogLikelihood_NoTransit(Theta, UniqueSaveNum, Time, Flux, AllBasisVectors, Locations):

    #To track down the best fit scenario
    global LeastChiSqrNoTransit

    NumberNights = len(Locations)+1
    BackgroundContinuum = np.zeros(len(Time))

    Start = 0
    StartCoeffs = 0
    for NightCount in range(NumberNights):
        if NightCount<len(Locations):
            Stop = Locations[NightCount]
        else:
            Stop = len(Flux)

        #Number of Basis is given by:
        IsThereZero = np.all(AllBasisVectors[NightCount][1]==0)

        if IsThereZero:
            NumBasis = 1
        else:
            NumBasis = 2

        CurrentBkg = np.zeros(len(Time[Start:Stop]))
        #1. Find the relevant coefficients
        if NumBasis==1:
            RelevantCoeffs = Theta[StartCoeffs:StartCoeffs+2]
            CurrentBkg += np.polyval(RelevantCoeffs,AllBasisVectors[NightCount][0][Start:Stop])
            StartCoeffs+=2
        if NumBasis==2:
            for SubBasis in range(2):
                RelevantCoeffs = Theta[StartCoeffs+SubBasis*2:StartCoeffs+SubBasis*2+2]
                CurrentBkg += np.polyval(RelevantCoeffs+[0.0], AllBasisVectors[NightCount][SubBasis][Start:Stop])

            StartCoeffs+=4 #when two variables are selected then there are five coefficients.

        #This offset does not include the
        CurrentOffset = np.mean(Flux[Start:Stop])-np.mean(CurrentBkg)
        CurrentBkg+=CurrentOffset

        #Now construct the Background for each night
        BackgroundContinuum[Start:Stop] += CurrentBkg
        Start = Stop

    Residual = (Flux - BackgroundContinuum)
    SumResidual = np.sum(Residual*Residual)

    ChiSquare = np.sum(SumResidual)/(0.5*0.007**2)

    if SumResidual<LeastChiSqrNoTransit:
        LeastChiSqNoTransit = SumResidual
        STD = np.std(Residual)

        #Save the parameters and the residual
        FileName =  "temp/Residual"+str(SaveNum)+".dat"
        with open(FileName, 'wb') as f:
            np.savetxt(f, np.array([SumResidual, STD]))
    return -ChiSquare



def MCMC_FinalFit_NoTransit(UniqueSaveNum, Time, Flux, AllBasisVectors, NumRuns, PolyNDim):

    global LeastChiSqrNoTransit
    LeastChiSqrNoTransit = np.inf


    NDim = PolyNDim
    NWalkers = 10*(PolyNDim+4)

    #Find the Location for each nights
    Diff1D = np.diff(Time)
    Index = np.concatenate((np.array([False]), Diff1D>0.25))
    Locations = np.where(Index)[0]

    PolyCoeff_Init = np.random.normal(0,0.05,(NWalkers, PolyNDim))

    #Stacking for the initial guesses
    StartingGuesses = PolyCoeff_Init

    #start the basis vectors
    sampler = emcee.EnsembleSampler(NWalkers, NDim, Final_LogLikelihood_NoTransit, args=(UniqueSaveNum, Time, Flux, AllBasisVectors, Locations), threads=1)
    pos, prob, state = sampler.run_mcmc(StartingGuesses, NumRuns)

    #print("The mean acceptance rate for non-transit MCMC is::", np.mean(sampler.acceptance_fraction))

    ResidualSum, STD = np.loadtxt("temp/Residual"+str(SaveNum)+".dat")
    return ResidualSum, STD
