import numpy as np
import matplotlib.pyplot as plt
import glob
import itertools
from fpdf import FPDF
import multiprocessing as mp
from scipy.stats import binned_statistic
import os
from .sampler import MCMC_FinalFit, MCMC_FinalFit_NoTransit
from .Functions import SigmaClip
from .sampler import BoxFit
from warnings import warn

def PerformQuickFit(UniqueSaveNum, OutputDir, Time, Flux, AllBasis, T0, Period, TDur1, TDur2, TDepth1, TDepth2, NumRuns):
    '''Performs quick MCMC fit for each night'''
    #Divide into different night here
    Diff1D = np.diff(Time)
    Index = np.concatenate((np.array([False]), Diff1D>0.2))

    Locations = np.where(Index)[0]
    NumberNights = len(Locations)+1

    #Basis function based on the same length as time but the components will depend on which night
    Start = 0

    TimeList = []
    FluxList = []
    SelectBasisIndex = np.zeros(len(Time))

    AllBasisVectors = []
    AllBasisNames = [] #This is just for sanity check

    NDim = 0
    for NightCount in range(NumberNights):


        if NightCount<len(Locations):
            Stop = Locations[NightCount]
        else:
            Stop = len(Flux)

        IndividualTime = Time[Start:Stop]
        IndividualFlux = Flux[Start:Stop]

        #do not fit if less than 50 data points are present
        if len(IndividualTime)<50:
            Start = Stop
            continue

        TimeList.extend(IndividualTime)
        IndividualFlux -= np.median(IndividualFlux)

        FluxList.extend(IndividualFlux)
        SelectBasisIndex[Start:Stop]=1


        #Find the night value
        File4Basis = OutputDir+"/Data/BestBasis_"+str(int(min(IndividualTime)))+".txt"
        BasisNames = open(File4Basis,"r+").readline()



        if(len(BasisNames))>2.01:
            warn("More than 2 basis were chosen. Choosing the first two basis")
            BasisNames = BasisNames[:2]


        SameLengthVector = np.zeros((2,len(Time)))

        CurrentDim = 0
        for BasisCount, BasisItem in enumerate(BasisNames):
            CurrentDim+=2
            if "T" in BasisItem.upper():
                SameLengthVector[BasisCount,Start:Stop] = IndividualTime -np.mean(IndividualTime)
            elif "X" in BasisItem.upper():
                SameLengthVector[BasisCount,Start:Stop] = AllBasis[0,:][Start:Stop]
            elif "Y" in BasisItem.upper():
                SameLengthVector[BasisCount,Start:Stop] = AllBasis[1,:][Start:Stop]
            elif "D" in BasisItem.upper():
                SameLengthVector[BasisCount,Start:Stop] =  AllBasis[2,:][Start:Stop]
            elif "F" in BasisItem.upper():
                SameLengthVector[BasisCount,Start:Stop] =  AllBasis[3,:][Start:Stop]
            elif "S" in BasisItem.upper():
                SameLengthVector[BasisCount,Start:Stop] =  AllBasis[4,:][Start:Stop]
            elif "A" in BasisItem.upper():
                SameLengthVector[BasisCount,Start:Stop] =  AllBasis[5,:][Start:Stop]
        NDim+=CurrentDim

        AllBasisVectors.append(SameLengthVector)
        Start=Stop

    TimeArray = np.array(TimeList)
    FluxArray = np.array(FluxList)

    SelectBasisIndex = SelectBasisIndex.astype(np.bool)
    AllBasisVectors = np.array(AllBasisVectors)
    AllBasisVectors = AllBasisVectors[:,:,SelectBasisIndex]

    #Run MCMC with 20000 steps, and get error in the process...
    TDepth, TDepth_STD, Period, T0, TDur, Residual1, STD1 = MCMC_FinalFit(UniqueSaveNum, TimeArray, FluxArray, AllBasisVectors, T0, Period, TDur1, TDur2, TDepth1, TDepth2, NumRuns, NDim)
    #Residual2, STD2 = MCMC_FinalFit_NoTransit(UniqueSaveNum, TimeArray, FluxArray, AllBasisVectors, NumRuns, NDim)

    FeasibilityFlag = FeasibleCalcTDur(Period, TDur*0.75)

    if FeasibilityFlag:
        #Metric = TDepth/max([STD1, TDepth_STD]))
        Metric = TDepth/(0.1*STD1+TDepth_STD)
        #Metric = (STD2-STD1)*10000
    else:
        #Metric = TDepth/(max([STD1, TDepth_STD]))*1/5.
        Metric = TDepth/(0.1*STD1+TDepth_STD)*1./5.
        #Metric = (STD2-STD1)*1./5*10000
    return TDepth, TDepth_STD, Period, T0,  TDur, Metric

def FeasibleCalcTDur(Period, TDur, Period_Error=0.0, MStar=0.14, RStar=0.14):
    '''For calculating if the transit duration is a feasible one'''
    PeriodInSecs = Period*86400.0
    RSun = 6.957e8
    a = (6.672e-11*MStar*1.99e30*PeriodInSecs**2/(4.*np.pi**2))**(1./3.)
    TransitDur = (Period/np.pi)*np.arcsin(RStar*RSun/a)
    return TDur<TransitDur

def TransitComparison(DataFileName, OutputDir, T0_Values,TDepth_Values,TDur_Values,TDepUCertainty_Values, PromiseValues, TimeData, FluxData, ModelData, TransitModelData):

    TargetFileName = OutputDir.split("/")[-1]

    #Create a folder
    if not(os.path.exists("temp")):
        os.system("mkdir temp")
    else:
        print("Removing the contents from temp folder.")
        os.system("rm -r temp/*")

    #Read all the files
    JD_UTC, Flux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS, ExpTime = np.loadtxt(DataFileName, unpack=True)

    NUM_MCMC_RUNS = int(open("SearchParams.ini","r+").readlines()[13].split("#")[0].split(":")[1])

    ZippedValues = zip(T0_Values,TDepth_Values,TDur_Values,TDepUCertainty_Values, PromiseValues,TimeData, FluxData, ModelData, TransitModelData)
    CombinationZip = list(itertools.combinations(ZippedValues,2))
    CombinedPromiseFactor = []

    for Value in CombinationZip:
        #Unpack the value
        T1,TDepth1,TDur1,TDepU1,PromiseValue1,_,_,_,_ = Value[0]
        T2,TDepth2,TDur2,TDepU2,PromiseValue2,_,_,_,_ = Value[1]

        #TDepthU = np.sqrt(TDepU1**2+TDepU2**2)
        TDepthU = min([TDepU1, TDepU2])

        TDiff = np.abs(TDepth1-TDepth2)
        TDur_Diff = np.abs(TDur1 - TDur2)

        CurrentPromiseFactor = PromiseValue1**0.3+PromiseValue2**0.6
        #SimilarityValue = PromiseValue1**0.1+PromiseValue2**0.6

        #To see if different cases are considered.
        #print(round(PromiseValue1,2),  round(T1,2), round(PromiseValue2,2), round(T2,2), round(SimilarityValue,2))
        CombinedPromiseFactor.append(CurrentPromiseFactor)


    #Find indices of the values
    CombinedPromiseFactor = np.array(CombinedPromiseFactor)
    OrderIndex = np.argsort(CombinedPromiseFactor)
    CombinedPromiseFactor = CombinedPromiseFactor[OrderIndex]


    ReadFile=open("SearchParams.ini", "r")
    AllData =  ReadFile.readlines()

    NumbersOfModels2Consider = int(AllData[14].split(":")[1].split("#")[0].replace(" ",""))
    print("Now comparing the top %d pairs of transits." %(NumbersOfModels2Consider))
    MCMCSignificanceArray = np.ones(NumbersOfModels2Consider)*-1e5

    Num = 0
    AssignNum = 0
    TString2SaveList = ["  "]*len(MCMCSignificanceArray)
    while AssignNum<len(MCMCSignificanceArray) and Num<len(CombinationZip):
        Num+=1
        print("Now Running::", Num, " and assign number is::", AssignNum)
        Location = np.where(OrderIndex==max(OrderIndex))[0][0]
        OrderIndex[Location]=-3000


        T1,TDepth1,TDur1,TDepU1,PromiseValue1,TS1,Flux1,Model1,TModel1 = CombinationZip[Location][0]
        T2,TDepth2,TDur2,TDepU2,PromiseValue2,TS2,Flux2,Model2,TModel2 = CombinationZip[Location][1]

        MeanTDur = (TDur1+TDur2)/2.0

        CalcPeriod = abs(T1 - T2)              #The calculated period


        #The range of the plot has to be same
        YRange = [min([min(Flux1),min(Flux2)])*1.05, max([max(Flux1),max(Flux2)])*1.05]

        #Bin the light curve for five Minutes
        BinSize = 5.0 #minutes

        #The time differennce in  days
        TimeDifference1 = max(TS1) - min(TS1)
        NumBins1 = int(TimeDifference1*86400.0/(BinSize*60.0))

        BinnedTime1 = binned_statistic(TS1, TS1, statistic='mean', bins=NumBins1)[0]
        BinnedFlux1 = binned_statistic(TS1, Flux1, statistic='median', bins=NumBins1)[0]


        STD1 = np.std(Flux1-Model1)
        NumInABin1 = len(TS1)/len(BinnedTime1)
        ScaledSTD1 = STD1*1.0/np.sqrt(NumInABin1)
        ErrorSTD1 = np.ones(len(BinnedTime1))*ScaledSTD1


        #For the second set of data
        TimeDifference2 = max(TS2) - min(TS2)
        NumBins2 = int(TimeDifference1*86400.0/(BinSize*60.0))

        BinnedTime2 = binned_statistic(TS2, TS2, statistic='mean', bins=NumBins2)[0]
        BinnedFlux2 = binned_statistic(TS2, Flux2, statistic='median', bins=NumBins2)[0]


        STD2 = np.std(Flux2-Model2)
        NumInABin2 = len(TS2)/len(BinnedTime2)
        ScaledSTD2 = STD2*1.0/np.sqrt(NumInABin2)
        ErrorSTD2 = np.ones(len(BinnedTime2))*ScaledSTD2



        #Plotting the two cases being considered
        plt.figure(figsize=(14,6))

        # Ensure the first figure has smaller T0
        if T1<T2:
            plt.subplot(121)
            plt.ylabel("Normalized Flux")
        else:
            plt.subplot(122)

        T1Plot = int(min(TS1))


        plt.plot(TS1 - T1Plot, Flux1, color="cyan",marker="o", linestyle="None", label="Data", markersize=2)
        plt.errorbar(BinnedTime1-T1Plot, BinnedFlux1, yerr=ErrorSTD1, zorder=5, color="black", ecolor="black", capsize=3, linestyle="None", marker="d")
        plt.plot(TS1 - T1Plot, Model1, "g-", lw=2)
        plt.plot(TS1 - T1Plot, TModel1, "r-", zorder=10, label="Transit Model")
        plt.axvline(x=T1-T1Plot,color="orange", lw=2)
        plt.ylim(YRange)
        plt.title(str(T1Plot))
        plt.legend()

        if T1<T2:
            plt.subplot(122)
        else:
            plt.subplot(121)
            plt.ylabel("Normalized Flux")

        T2Plot = int(min(TS2))

        plt.plot(TS2 - T2Plot, Flux2, color="cyan", marker="o", label="Data", linestyle="None", markersize=2)
        plt.errorbar(BinnedTime2-T2Plot, BinnedFlux2, yerr=ErrorSTD2, zorder=5, color="black", ecolor="black", capsize=3, linestyle="None", marker="d")

        plt.plot(TS2 - T2Plot, Model2, "g-")
        plt.plot(TS2 - T2Plot, TModel2, "r-", zorder=10, label="Transit Model")
        plt.axvline(x=T2-T2Plot,color="orange", lw=2)
        plt.ylim(YRange)
        plt.title(str(T2Plot))
        plt.tight_layout()
        SaveLocationFigName = "temp/"+str(AssignNum+1).zfill(3)+"TransitsCompared.png"
        plt.savefig(SaveLocationFigName)
        plt.close('all')

        #NumHarmonicsTrial = 1

        if CalcPeriod<2:
            NumHarmonicsTrial = 1
        elif CalcPeriod<5:
            NumHarmonicsTrial = 3
        else:
            NumHarmonicsTrial = 4

        #Assign arbitrary value to avoid errors
        BestSignificance = -1e20
        BestPeriod = -1

        if T1<T2:
            Save_TString = str(int(T1))+"_"+str(int(T2))
        else:
            Save_TString = str(int(T2))+"_"+str(int(T1))


        for HarmonicNUM in range(1,NumHarmonicsTrial+1):
            SelectDuration = 3.0*(max([TDur1, TDur2])+abs(TDur1-TDur2))  #four and a half hours converted to days
            T0 = min([T1,T2])
            PeriodHarmonic = CalcPeriod/HarmonicNUM

            #SelectIndex = np.abs((JD_UTC-T0)%PeriodHarmonic)-SelectDuration/2.<SelectDuration
            SelectIndex = np.abs((JD_UTC-T0+SelectDuration/2.)%PeriodHarmonic)<SelectDuration
            SelectedTime = JD_UTC[SelectIndex]

            SelectedFlux = Flux[SelectIndex]

            SelectedXShift = XShift[SelectIndex]
            SelectedXShift -= np.mean(SelectedXShift)
            SelectedXShift /= np.std(SelectedXShift)

            SelectedYShift = YShift[SelectIndex]
            SelectedYShift -= np.mean(SelectedYShift)
            SelectedYShift /= np.std(SelectedYShift)

            SelectedXY = np.sqrt(SelectedXShift*SelectedXShift+SelectedYShift*SelectedYShift)

            SelectedFWHM = FWHM[SelectIndex]
            SelectedFWHM -= np.mean(FWHM[SelectIndex])
            SelectedFWHM /= np.std(SelectedFWHM)

            SelectedSKY =  SKY[SelectIndex]
            SelectedSKY -= np.mean(SelectedSKY)
            SelectedSKY /= np.std(SelectedSKY)

            SelectedAIRMASS =  AIRMASS[SelectIndex]
            SelectedAIRMASS -= np.mean(SelectedAIRMASS)
            SelectedAIRMASS /= np.std(SelectedAIRMASS)


            #Remove the outliers index
            OutliersIndex = SigmaClip(SelectedTime, SelectedFlux, SigmaValue=5.0)
            SelectedTime = SelectedTime[~OutliersIndex]
            SelectedFlux = SelectedFlux[~OutliersIndex]
            SelectedXShift = SelectedXShift[~OutliersIndex]
            SelectedYShift = SelectedYShift[~OutliersIndex]
            SelectedXY = SelectedXY[~OutliersIndex]
            SelectedFWHM = SelectedFWHM[~OutliersIndex]
            SelectedSKY =  SelectedSKY[~OutliersIndex]
            SelectedAIRMASS = SelectedAIRMASS[~OutliersIndex]


            #This takes data and basis vectors for MCMC
            BasisVectors = np.vstack((SelectedXShift, SelectedYShift, SelectedXY, SelectedFWHM, SelectedSKY,SelectedAIRMASS))

            #Only run if the periodic harmonic is less than 0.5 days
            if PeriodHarmonic>0.5:
                UniqueSaveNum = 1000*(AssignNum+1)+HarmonicNUM
                #Now fit using quick MCMC using the basis found by SVD
                TDepth, TDepth_STD, Period, T0, TDur, Metric= PerformQuickFit(UniqueSaveNum, OutputDir, SelectedTime,SelectedFlux, BasisVectors, T0, PeriodHarmonic, TDur1, TDur2, TDepth1, TDepth2, NUM_MCMC_RUNS)
            else:
                continue

            #Save the data in the data

            PeriodData = open("Periodogram.data","a")
            PeriodData.write(str(T0)+","+ str(Period)+","+str(Metric)+"\n")
            PeriodData.close()

            NumDataPoints = len(SelectedTime)
            CurSignificance = Metric

            if BestSignificance<CurSignificance:
                BestSignificance = CurSignificance
                BestPeriod = Period
                BestUniqueSaveNum = UniqueSaveNum

                BestTDepth = TDepth
                BestTDepth_STD = TDepth_STD
                Best_T0 = T0
                Best_TDur = TDur


        #If the best period was not found properly
        if BestPeriod==-1:
            continue


        #For arranging the figures in respect to their significance
        MCMCSignificanceArray[AssignNum] = BestSignificance
        TString2SaveList[AssignNum] = Save_TString
        AssignNum+=1

        #Get the relevant file
        RelevantFileFigures = glob.glob("temp/"+str(BestUniqueSaveNum)+"*.png")

        #Arrange the figures
        RelevantFileFigures = np.array(RelevantFileFigures)
        IndexRelevantFileFigures = np.array([int(Item[-7:-4]) for Item in RelevantFileFigures])
        RelevantFileFigures = RelevantFileFigures[np.argsort(IndexRelevantFileFigures)]

        Height = 14.0+(6.0*2.54)*(len(RelevantFileFigures)+1)
        #Temporirily Save the File
        PdfFile = FPDF('P', 'cm', (20*2.54,Height))
        #Add information on the Fit signficance
        Text1 = 'Period : '+str(round(BestPeriod,5))
        Text2 = 'Delta : '+str(round(TDepth,6))+ " pm "+str(round(TDepth_STD,6))
        Text3 = "T0: "+str(round(Best_T0,5))
        Text4 = "TDur: "+str(round(Best_TDur*60.0*24.0,2))+" Minutes"

        PdfFile.add_page()
        PdfFile.set_font('Helvetica','B', 35)
        PdfFile.cell(ln=1, h=2.0, align='C', w=0, txt=Text1, border=0)
        PdfFile.cell(ln=2, h=2.0, align='C', w=0, txt=Text2, border=0)
        PdfFile.cell(ln=3, h=2.0, align='C', w=0, txt=Text3, border=0)
        PdfFile.cell(ln=4, h=2.0, align='C', w=0, txt=Text4, border=0)
        PdfFile.line(0, 9.5, 170, 9.5)
        PdfFile.line(0, 25.5, 170, 25.5)
        PdfFile.image(SaveLocationFigName, 5.0,10.0,14.0*2.54,6.0*2.54)

        for count,FileName in enumerate(RelevantFileFigures):
            Y = 12.0+(6.0*2.54)*(count+1)
            PdfFile.image(FileName, 5,Y,14.0*2.54,6.0*2.54)

        #Save under trackable name
        PdfFile.output("temp/Test"+str(AssignNum).zfill(3)+".pdf")
        PdfFile.close()

        #Delete all the irrelevant files
        os.system("rm -r temp/*.png")
        os.system("rm -r temp/*.dat")



    #GlobalAnalysis Folder
    Folder2Save = OutputDir+"/GlobalAnalysis"
    if not(os.path.exists(Folder2Save)):
        os.system("mkdir %s" %Folder2Save)
    else:
        os.system("rm %s/*" %Folder2Save)

    print("Now saving the folder")

    #Remove the portion which are not assigned any values at all
    MCMCSignificanceArray = MCMCSignificanceArray[:AssignNum]
    TString2SaveList = np.array(TString2SaveList[:AssignNum])


    MCMCIndex = np.argsort(MCMCSignificanceArray)[::-1]

    TString2SaveList = TString2SaveList[MCMCIndex]


    PdfFileNamesArray = np.array(glob.glob("temp/*.pdf"))

    #Arranging the files according to their indexing
    PdfFileNamesNumbers = np.array([int(Item[-7:-4]) for Item in PdfFileNamesArray])
    PdfFileNamesArray = PdfFileNamesArray[np.argsort(PdfFileNamesNumbers)]
    PdfFileNamesArray = PdfFileNamesArray[MCMCIndex]


    for Start, OriginalFile in enumerate(PdfFileNamesArray):
        DestinationFile = OutputDir+"/GlobalAnalysis/"+str(Start+1).zfill(3)+TargetFileName+"_"+TString2SaveList[Start]+".pdf"
        os.system("mv %s %s" %(OriginalFile, DestinationFile))
    np.savetxt("MCMC.txt", MCMCSignificanceArray)
