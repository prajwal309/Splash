'''
This file containts the run algorithm for the transit search
'''

import matplotlib.pyplot as plt
import itertools
import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm
import pickle
from transitleastsquares import transitleastsquares


from scipy.stats import binned_statistic
from math import ceil
from .splash import Target
from .Functions import ParseFile, SVDSolver, \
      TransitBoxModel, RunningResidual, FindLocalMaxima


class GeneralTransitSearch:
    '''
    Description
    ------------
    This method implements the linearized search for the transits


    Parameters
    ---------------
    This method does a night by night basis for the function

    Current default method is SVD
    method="svd"

    More methods to be implemented in the future.

    Linearize the flux as a function of the
    Yields
    ---------------

    '''

    def __init__(self):
        '''
        Initialized the transit search parameters from Search.ini
        '''
        self.transitSearchParam = ParseFile("SearchParams.ini")



    def Run(self, Target, ShowPlot=False, SavePlot=False, SaveData=True):
        '''
        Runs SVD algorithm to find the

        Parameters
        -------------

        Target: class object from splash
                SPECULOOS target whose method allows access to the data

        ShowPlot: bool
                  Shows the plot if True

        SavePlot: bool
                  Save the plot if True

        SaveData: bool
                  Save the pickled data of pickled data
        '''

        self.TStepSize = float(self.transitSearchParam["TStepSize"])/(24.0*60.0)
        self.TDurStepSize = float(self.transitSearchParam["TDurStepSize"])/(24.0*60.0)
        self.TDurLower, self.TDurHigher = [float(Item)/(24.0*60.0) for Item in self.transitSearchParam["TransitDuration"].split(",")]
        self.TDurValues = np.arange(self.TDurLower, self.TDurHigher+self.TDurStepSize, self.TDurStepSize)
        self.BasisItems = self.transitSearchParam['Params'].split(",")

        NCPUs = int(self.transitSearchParam["NCPUs"])

        if NCPUs==-1:
            self.NUM_CORES = mp.cpu_count()
        elif NCPUs>0 and NCPUs<64:
            self.NUM_CORES = int(NCPUs)

        self.ParamColumns = self.getParamCombination(Target)
        self.AllMetricMatrix = []
        self.AllModeledT0 = []
        self.AllDetrendedFlux = []

        for NightNum in range(Target.NumberOfNights):
            print("Running %d Night" %(NightNum+1))
            self.RunNight(Target, NightNum)


            self.AllDetrendedFlux.extend(self.BestDetrendedModel)

            #Save the data
            self.AllMetricMatrix.append(self.CurrentMetricMatrix)

            #Now save the data as a pickle
            self.DataDir = os.path.join(Target.ResultDir, "Data")

            if not(os.path.exists(self.DataDir)):
                os.system("mkdir %s" %self.DataDir)

            if SaveData:
                FileName = os.path.join(self.DataDir, "Night%sData.pkl" %(NightNum+1))

                with open(FileName, 'wb') as f:
                    pickle.dump(self.CurrentTransitDepthMatrix, f)
                    pickle.dump(self.CurrentUnctyTransitMatrix, f)
                    pickle.dump(self.CurrentResidualMatrix, f)


            Title = Target.ParamNames[self.BestBasisColumns]
            TitleText = "  ".join(Title)


            if ShowPlot or SavePlot:

                BestPixelRow, BestPixelCol = np.where(self.CurrentMetricMatrix==np.max(self.CurrentMetricMatrix))
                BestT0 = self.T0_Values[BestPixelRow][0]
                BestTDur = self.TDurValues[BestPixelCol][0]

                #Bin the data
                NumBins = int((max(self.CurrentTime) - min(self.CurrentTime))*24.0*60.0/5.0)

                self.CurrentResidual = self.CurrentFlux - self.BestModel
                self.BinnedTime = binned_statistic(self.CurrentTime, self.CurrentTime, bins=NumBins)[0]
                self.BinnedFlux = binned_statistic(self.CurrentTime, self.CurrentFlux, bins=NumBins)[0]
                self.BinnedResidual = RunningResidual(self.CurrentTime, self.CurrentResidual, NumBins)
                #Find the running errorbar


                T0_Int = int(min(self.T0_Values))

                XPlot = self.CurrentTime - T0_Int
                YPlot = self.CurrentFlux
                fig, ax = plt.subplots(figsize=(15,8), nrows=2, ncols=1)

                ax[0].plot(self.CurrentTime - T0_Int, self.CurrentFlux,\
                           linestyle="None", color="cyan", marker="o", \
                           markersize=2)

                ax[0].errorbar(self.BinnedTime - T0_Int, self.BinnedFlux, \
                               yerr=self.BinnedResidual, marker="o", \
                               markersize=2, linestyle="None", capsize=3,\
                               color="black", ecolor="black")

                ax[0].plot(self.CurrentTime - T0_Int, self.BestModel, "r-")


                ax[0].axvline(BestT0 - T0_Int, color="red")
                ax[0].set_xlim(min(self.T0_Values- T0_Int), max(self.T0_Values- T0_Int))
                ax[0].set_ylim([0.98,1.02])
                ax[0].set_xticklabels([])
                ax[0].set_ylabel("Normalized Flux", fontsize=20)
                ax[0].set_title(TitleText)

                T0Offset = np.mean(np.diff(self.T0_Values))/2.0
                TDurOffSet = np.mean(np.diff(self.TDurValues))/2
                ax[1].imshow(self.CurrentMetricMatrix.T, aspect='auto', origin='lower', \
                      extent=[min(self.T0_Values- T0_Int-T0Offset), max(self.T0_Values - T0_Int+T0Offset), \
                      min(self.TDurValues-TDurOffSet)*24.0*60.0, \
                      max(self.TDurValues+TDurOffSet)*24.0*60.0])

                ax[1].axvline(BestT0 - T0_Int, color="red", linestyle=":")
                ax[1].axhline(BestTDur*24.0*60.0, color="red", linestyle=":")
                ax[1].set_ylabel("Transit Duration (mins)", fontsize=20)
                ax[1].set_xlabel("Time %s JD " %(T0_Int), fontsize=20)
                plt.tight_layout()
                if SavePlot:
                    self.DailyBestFolder = os.path.join(Target.ResultDir, "DailyBestCases")
                    if not(os.path.exists(self.DailyBestFolder)):
                        os.system("mkdir %s" %self.DailyBestFolder)
                    SaveName = os.path.join(self.DailyBestFolder,"Night"+str(NightNum+1).zfill(4)+".png")
                    plt.savefig(SaveName)
                if ShowPlot:
                    plt.show()
                plt.close('all')



        self.AllModeledT0 = np.array(self.AllModeledT0)
        self.AllDetrendedFlux = np.array(self.AllDetrendedFlux)



        #Save the file
        np.savetxt(os.path.join(self.DataDir,"DetrendedFlux.csv"), \
                  np.transpose((Target.AllTime[Target.QualityFactor], self.AllDetrendedFlux)), \
                  delimiter="," , header="Time, Detrended Flux")


    def getParamCombination(self, Target):
        '''
        This method will yield different combination of the parameters
        to be used as the basis vector.

        Parameter:
        -----------
        Target: class
                Target class that allows access to the lightcurve


        Yields
        -----------
        The combination of column numbers of data which are to be
        tested as the basis vector.
        '''

        ColumnValues = []

        for Basis in self.BasisItems:
            for ItemCount, Item in enumerate(Target.ParamNames):
                if Basis.upper() in Item.upper():
                    ColumnValues.append(ItemCount)

        ColumnValues = list(set(ColumnValues))

        ColumnArray = np.array(ColumnValues)
        self.ColumnArray = np.array(ColumnValues)

        self.BasisCombination = list(itertools.combinations(self.ColumnArray,int(self.transitSearchParam["Combination"])))

        self.BasisCombination = []
        for i in range(1,int(self.transitSearchParam["Combination"])+1):
            Basis = [list(itertools.combinations(self.ColumnArray,i))]
            self.BasisCombination.extend(Basis[0])



    def ConstructBasisMatrix(self, T0, TDur, BasisColumn):
       '''
       This method constructs

       Parameters
       ============
       T0: float
           The value of

       TDur: float
            The value of

       BasisColumn: list of integers
                    The value of columns of CurrentData to be used to used as basis functions


       Yields
       ==============
       Basis vector which can be used to
       '''

       PolyOrder = int(self.transitSearchParam['PolynomialOrder'])
       NumParams = PolyOrder*len(BasisColumn)+2
       BasisMatrix = np.ones((len(self.CurrentTime), NumParams))

       for Col_Count, Col in enumerate(BasisColumn):
           for Order in range(PolyOrder):
               AssignColumn = Col_Count*PolyOrder+Order
               BasisMatrix[:,AssignColumn] = np.power(self.CurrentData[:, Col],Order+1)

       BasisMatrix[:,-2] =  TransitBoxModel(self.CurrentTime, T0, TDur)
       return BasisMatrix


    def RunNight(self, Target, NightNum):
        '''
        This functions find


        Input
        =============
        Parameter:
        -----------
        Target: class
                Target class that allows access to the lightcurve


        NightNum: integer
                  The number night is the index of the night to be run e.g. 1
                  Indexing begins at 1.

        Yields
        ==============
        Yields 2D chi squared map (M,N) for each night for the
        where M is the length of T0 Values and N is the length
        of Transit duration arrays. Can be accessed using ChiSquareMap.

        '''

        #Target
        QualityIndex = Target.QualityFactorFromNight[NightNum]
        self.CurrentData = Target.DailyData[NightNum][QualityIndex]
        self.CurrentTime = self.CurrentData[:,0]
        self.CurrentFlux = self.CurrentData[:,1]

        self.T0_Values = np.arange(self.CurrentTime[0],self.CurrentTime[-1], self.TStepSize)
        self.CurrentMetricMatrix = np.ones((len(self.T0_Values), len(self.TDurValues)))*-np.inf
        self.CurrentResidualMatrix = np.ones((len(self.T0_Values), len(self.TDurValues)))*-np.inf
        self.CurrentTransitDepthMatrix = np.ones((len(self.T0_Values), len(self.TDurValues)))*-np.inf
        self.CurrentUnctyTransitMatrix = np.ones((len(self.T0_Values), len(self.TDurValues)))*-np.inf
        self.BestMetric = -np.inf

        self.AllModeledT0.extend(self.T0_Values)

        #Need to run for all of the cases...
        T0_TDur_Basis_Combinations = itertools.product(self.T0_Values, self.TDurValues, self.BasisCombination)
        NumOperations = (len(self.T0_Values)*len(self.TDurValues)*len(self.BasisCombination))

        for i in tqdm(range(ceil(NumOperations/self.NUM_CORES))):
            Tasks = []
            CPU_Pool = mp.Pool(self.NUM_CORES)
            for TaskCounter in range(self.NUM_CORES):
                try:
                    T0, TDur, Combination = next(T0_TDur_Basis_Combinations)
                except:
                    pass
                BasisVector = self.ConstructBasisMatrix(T0, TDur, Combination)
                Tasks.append(CPU_Pool.apply_async(SVDSolver,(BasisVector, self.CurrentFlux, T0, TDur, Combination)))

            CPU_Pool.close()
            CPU_Pool.join()


            for Index, Task in enumerate(Tasks):
                Coeff, Uncertainty, Residual, Model, DetrendedFlux, \
                T0, TDur, Combination = list(Task.get())
                T0_Index = np.argmin(np.abs(self.T0_Values-T0))
                TDur_Index = np.argmin(np.abs(self.TDurValues-TDur))

                Metric = (Coeff[-2]/Uncertainty[-2])/(Residual*Residual)

                if self.BestMetric<Metric:
                    self.BestMetric = Metric
                    self.BestBasisColumns = np.array(Combination)
                    self.BestModel = Model
                    self.BestDetrendedModel = self.CurrentFlux - DetrendedFlux



                if self.CurrentMetricMatrix[T0_Index, TDur_Index]<Metric:
                    self.CurrentResidualMatrix[T0_Index, TDur_Index] = Residual
                    self.CurrentTransitDepthMatrix[T0_Index, TDur_Index] = Coeff[-2]
                    self.CurrentUnctyTransitMatrix[T0_Index, TDur_Index] = Uncertainty[-2]
                    self.CurrentMetricMatrix[T0_Index, TDur_Index] = Metric

        #Replace the missed value with minima
        InfLocation = np.where(~np.isfinite(self.CurrentMetricMatrix))
        self.CurrentResidualMatrix[InfLocation] = np.max(self.CurrentResidualMatrix)
        self.CurrentTransitDepthMatrix[InfLocation] = np.min(self.CurrentTransitDepthMatrix)
        self.CurrentUnctyTransitMatrix[InfLocation] = np.max(self.CurrentUnctyTransitMatrix)
        self.CurrentMetricMatrix[InfLocation] = np.min(self.CurrentMetricMatrix)


    def PeriodicSearch(self, Target, method="TransitPair", SavePlot=True, ShowPlot=False):
        '''
        This function utilizes the linear search information and

        Parameters
        ------------

        Target: splash target object
                Target object initiated with a light curve file

        method: string
                either "TransitPair" or "TLS" is expected.

        ShowPlot: bool
                Shows the plot if True

        SavePlot: bool
                 Saves the plot if True under DiagnosticPlots subfolder


        Yields
        ---------------
        '''

        #Determine the phase coverage
        if method == "TransitPair":
            self.TransitPairing(Target, ShowPlot, SavePlot)

        elif method == "TLS":
            print("Now running TLS")
            self.TLS(Target, ShowPlot, SavePlot)



    def TransitPairing(self, Target, ShowPlot, SavePlot):
        '''
        Method to look at periodicity in the likelihood function

        Parameters
        ------------

        Target: splash target object
                Target object initiated with a light curve file

        ShowPlot: bool
                Shows the plot if True

        SavePlot: bool
                 Saves the plot if True under DiagnosticPlots subfolder


        Return:
        Returns T0, Period, and Likelihood
        '''

        self.UnravelMetric = []
        Row, Col = np.shape(self.AllMetricMatrix[0])
        self.AllArrayMetric = np.zeros((Col,1))
        self.TransitDepthArray = np.zeros((Col,1))
        self.TransitUnctyArray = np.zeros((Col,1))
        self.ResidualArray = np.zeros((Col,1))

        for Counter in range(len(self.AllMetricMatrix)):
            self.AllArrayMetric= np.column_stack((self.AllArrayMetric,self.AllMetricMatrix[Counter].T))


        self.AllArrayMetric = self.AllArrayMetric[:,1:]
        self.UnravelMetric = np.max(self.AllArrayMetric, axis=0)



        #Find the peaks. Ignore any simultaneous peak
        self.PeakLocations = np.where(FindLocalMaxima(self.UnravelMetric, NData=4))[0]

        #Take all transit pairs
        LocationCombination = list(itertools.combinations(self.PeakLocations,2))


        self.T0s = []
        self.TP_Periods = []
        self.SDE = []

        for Loc1,Loc2 in LocationCombination:
            CurrentPeriod = np.abs(self.AllModeledT0[Loc1]-self.AllModeledT0[Loc2])
            PhaseStepSize = 0.50*float(self.transitSearchParam["TStepSize"])/(CurrentPeriod*60.0*24.0)


            if CurrentPeriod<0.25:
                continue
            self.TP_Periods.append(CurrentPeriod)

            T1= self.AllModeledT0[Loc1]
            T2= self.AllModeledT0[Loc2]

            self.T0s.append(min([T1,T2]))

            CurrentPhase = (self.AllModeledT0-T1+CurrentPeriod/2.0)%CurrentPeriod
            CurrentPhase = CurrentPhase/CurrentPeriod


            SelectedColumns = np.abs(CurrentPhase-0.5)<PhaseStepSize
            CalculatedSDE = np.sum(self.AllArrayMetric[:,SelectedColumns], axis=1)

            NumPoints = len(SelectedColumns)
            self.SDE.append(np.max(CalculatedSDE)/np.sqrt(NumPoints))




        #Look for the Metric Array
        self.TP_Periods = np.array(self.TP_Periods)
        self.SDE = np.array(self.SDE)
        self.T0s = np.array(self.T0s)
        BestPeriod = self.TP_Periods[np.argmax(self.SDE)]

        #Arrange the data
        ArrangeIndex = np.argsort(self.TP_Periods)
        self.TP_Periods = self.TP_Periods[ArrangeIndex]
        self.SDE = self.SDE[ArrangeIndex]
        self.T0s = self.T0s[ArrangeIndex]


        SaveName = os.path.join(self.DataDir, "TransitPairingPeriodogram.csv")

        np.savetxt(SaveName, np.transpose((self.TP_Periods, self.SDE)),\
        delimiter=",",header="Period,SDE")

        #plotting
        fig, ax1 = plt.subplots(figsize=(14,8))
        ax2= ax1.twinx()

        for i in range(0,4):
            if i == 0:
                ax1.axvline(x=0.5*BestPeriod, color="cyan", linestyle=":", lw=3, alpha=0.8, label="True Period")
            else:
                ax1.axvline(x=i*BestPeriod, color="cyan", linestyle=":", lw=3, alpha=0.8)
        ax1.plot(self.TP_Periods, self.SDE, "r-", lw=2)
        ax2.plot(Target.PhasePeriod, Target.PhaseCoverage, color="green", alpha=0.8, lw=2.0, label="Phase Coverage")
        ax1.set_xlabel("Period (Days)", fontsize=20)
        ax2.set_ylabel("Phase Coverage (\%)", color="green", labelpad=3.0,fontsize=20, rotation=-90)
        ax1.set_ylabel("Signal Detection Efficiency", color="red", fontsize=20)
        MinXLim = min([min(self.TP_Periods), min(Target.PhasePeriod)])
        MaxXLim = max([max(self.TP_Periods), max(Target.PhasePeriod)])
        ax1.set_xlim([MinXLim, MaxXLim])
        ax1.text(0.98*MaxXLim,0.98*max(self.SDE), "Best Period:"+str(round(BestPeriod,5)), horizontalalignment="right")
        ax1.set_ylim([0, 1.2*max(self.SDE)])
        ax1.tick_params(which="both", direction="in", colors="red")
        ax2.tick_params(which="both", direction="in", colors="green")
        ax1.spines['left'].set_color('red')
        ax1.spines['right'].set_color('green')
        ax2.spines['right'].set_color('green')
        ax2.spines['left'].set_color('red')
        plt.tight_layout()
        if SavePlot:
            self.SavePath = os.path.join(Target.ResultDir, "DiagnosticPlots")
            if not(os.path.exists(self.SavePath)):
                os.system("mkdir %s" %self.SavePath)
            if SavePlot:
                plt.savefig(os.path.join(self.SavePath,"TransitPairing_Periodogram.png"))
            if ShowPlot:
                plt.show()

        plt.close('all')

        #Save all potential T0, Period, Power

    def PairingFunction(self, T0, Period):
        pass

    def TLS(self, Target, ShowPlot, SavePlot):
        '''
        Performs transit least squares search
        on the detrended light curve that preserves the transit

        Parameters
        ------------

        Target: splash target object
                Target object initiated with a light curve file

        ShowPlot: bool
                Shows the plot if True

        SavePlot: bool
                 Saves the plot if True under DiagnosticPlots subfolder
        '''

        if not(ShowPlot or SavePlot):
            print("Either SavePlot or SavePlot should be True. Toggling on SavePlot.")
            SavePlot = True

        model = transitleastsquares(Target.AllTime[Target.QualityFactor], self.AllDetrendedFlux+1.0)

        results = model.power(
        period_min=0.6,
        period_max=(Target.AllTime[-1]-Target.AllTime[0]),
        oversampling_factor=5,
        duration_grid_step=1.02,
        n_transits_min=1
        )

        fig, ax = plt.subplots(figsize=(14,20), ncols=1, nrows=2)
        ax2= ax[0].twinx()
        ax[0].axvline(results.period, alpha=0.4, lw=3)
        for n in range(2, 5):
            ax[0].axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
            ax[0].axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")

        ax[0].set_xlabel('Period (days)', fontsize=20)
        ax[0].plot(results.periods, results.power, color='red', lw=2.0)
        ax[0].set_xlim(min(results.periods), max(results.periods))
        ax2.plot(Target.PhasePeriod, Target.PhaseCoverage, color="green", lw=2)

        ax[0].set_ylabel(r"SDE")
        ax2.set_ylabel(r"Phase Coverage")

        ax[0].spines['left'].set_color('red')
        ax[0].spines['right'].set_color('green')
        ax2.spines['right'].set_color('green')
        ax2.spines['left'].set_color('red')

        ax[1].plot(results.model_folded_phase, results.model_folded_model,color='red')
        ax[1].scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
        ax[1].set_xlim(0.480, 0.520)
        ax[1].set_xlabel('Phase')
        ax[1].set_ylabel('Relative flux');

        plt.tight_layout()


        #Now saving the files
        self.SavePath = os.path.join(Target.ResultDir, "DiagnosticPlots")
        if not(os.path.exists(self.SavePath)):
            os.system("mkdir %s" %self.SavePath)

        if SavePlot:
            plt.savefig(os.path.join(self.SavePath,"TLS_Periodogram.png"))
        if ShowPlot:
            plt.show()
        plt.close('all')



        pass
