'''
Will contain module for the quick transit fit
and robust transit fit
'''

import batman
import numpy as np
import matplotlib.pyplot as plt
import emcee

class TransitFit:
    """
    Description
    -----------

    Parameters Name
    --------------
    """
    def __init__(self, Target, TransitSearch, NumFits=3, TDur=1.5, NRuns=1000):
        '''
        This uses the thing from results from the target as well as Transit Search

        Parameters
        -----------
        Target: Target class object
                Object that provides access the the light curve

        TransitSearch: Transit Search Object
                        Object that provides access to the transit search algorithm performed
                        on the light curve
        TDur: float
            points around the transit to fit.

        method: string
                quick for quick fit

        '''
        #Convert from
        TDur=TDur/24.0
        self.ConstructFitCases(Target, TransitSearch, TDur, NumFits)
        for Counter in range(NumCases):
            print("The value of counter is::", Counter)
            self.Quick(Counter)


    def GetNightNumber(self, Target):
        '''
        Get the night index from which the data is taken

        Parameters
        ------------
        Target: class object
                Target class object from splash in order to

        '''
        #Find the number of night
        self.BreakLocation = np.where(np.diff(self.SelectedTime)>0.30)[0]
        self.AllNightIndex = []

        StartIndex = 0
        for counter in range(len(self.BreakLocation)+1):
            if counter == len(self.BreakLocation):
                StopIndex = len(self.SelectedTime)
            else:
                StopIndex = self.BreakLocation[counter]+1
            CurrentNight = int(min(self.SelectedTime[StartIndex:StopIndex]))
            CurrentNightIndex = np.where(Target.Daily_T0_Values == CurrentNight)[0][0]
            self.AllNightIndex.append(CurrentNightIndex)
            StartIndex = StopIndex
        self.AllNightIndex = np.array(self.AllNightIndex)
        print(self.AllNightIndex)


    def ConstructFitCases(self, Target, TransitSearch, TDur, NumFits):
        '''
        This function finds the number of cases to be fit.

        Parameters:
        -----------
        NumCases: Integer
                  Number of cases to be considered

        '''
        if (TransitSearch.TransitPair_Status):
            T0 =  TransitSearch.T0s
            Period = TransitSearch.TP_Periods

            #In order to choose the largest period
            SDE =  TransitSearch.SDE + Period/10000.00


            #Now fit for the transit

            CaseNumber=1
            while CaseNumber<NumFits:

                SelectIndex = np.argmax(SDE)

                self.CurrentT0 = T0[SelectIndex]
                self.CurrentPeriod = Period[SelectIndex]

                #Data length = 1.5 hours around transit
                SelectDataIndex = np.abs((Target.AllTime -self.CurrentT0 +TDur/2.)%self.CurrentPeriod)<TDur

                #################################################
                self.SelectedTime = Target.AllTime[SelectDataIndex]
                self.SelectedFlux = Target.AllFlux[SelectDataIndex]
                self.SelectedData = Target.ParamValues[SelectDataIndex]
                self.numDataPoints = len(self.SelectedTime)


                #Find the night number()
                self.GetNightNumber(Target)


                #Fit only if greater than 10
                #find where the night data
                self.QuickFit(Target, TransitSearch, CaseNumber)




        elif (TransitSearch.TLS_Status):
            print("Find the peaks")
            print("Find T0")
        input("Wait here..")
        pass


    def Likelihood(self,theta, params):
        #
        T0 = theta[0]
        Period = theta[1]
        a_Rs = theta[2]
        Rp_Rs = theta[3]
        b = theta[4]
        q1 = theta[5]
        q2 = theta[6]


        #apply the priors:
        if q1<0.0 or q1>1.0:
            return -np.inf

        if q2<0.0 or q2>1.0:
            return -np.inf

        if np.abs(T0-self.CurrentT0)>0.05:
            return -np.inf

        if np.abs(Period-self.CurrentPeriod)>0.05:
            return -np.inf

        if b<0.0 or b>1.2:
            return -np.inf

        if a_Rs<3.0 or a_Rs>100.0:
            return -np.inf

        if Rp_Rs<0.005 or Rp_Rs>0.30:
            return -np.inf

        if max(np.abs(theta[7:]))>1000:
            return -np.inf


        Inc = np.rad2deg(np.arccos(b/a_Rs))

        #Calculate u1 and u2 from q1 and q2
        u1 = 2.0*np.sqrt(q1)*q2
        u2 = np.sqrt(q1) - 2.0*np.sqrt(q1)*q2

        params = batman.TransitParams()
        params.t0 = T0                        #time of inferior conjunction
        params.per = Period                   #orbital period
        params.rp = Rp_Rs                     #planet radius (in units of stellar radii)
        params.a = a_Rs                       #semi-major axis (in units of stellar radii)
        params.inc = Inc                      #orbital inclination (in degrees)
        params.ecc = 0.                       #eccentricity
        params.w = 90.                        #longitude of periastron (in degrees)
        params.u = [u1, u2]                 #limb darkening coefficients [u1, u2]
        params.limb_dark = "quadratic"


        m = batman.TransitModel(params, self.SelectedTime)    #initializes model
        TransitModelFlux = m.light_curve(params)          #calculates light curve

        DetrendedFlux = np.dot(self.BasisMatrix,theta[7:])
        ModelFlux = DetrendedFlux+TransitModelFlux

        #Number of offsets for number of nights
        StartIndex = 0
        for CurrentNight in range(self.NumNights):
            if CurrentNight == len(self.BreakLocation):
                StopIndex = len(self.SelectedTime)
            else:
                StopIndex = self.BreakLocation[CurrentNight]+1
            Offset =  np.mean(self.SelectedFlux[StartIndex:StopIndex] - ModelFlux[StartIndex:StopIndex])
            self.SelectedFlux[StartIndex:StopIndex]-=Offset
            Offset =  np.mean(self.SelectedFlux[StartIndex:StopIndex] - ModelFlux[StartIndex:StopIndex])
            StartIndex = StopIndex

        Residual = self.SelectedFlux - ModelFlux

        STD = np.std(Residual)
        SumResidual = np.sum(np.abs(Residual))

        if SumResidual<self.BestResidual:

            for x,y in zip(self.Parameters,theta[:7]):
                print(x,":: ", y)

            print("The other parameters are::")
            print(theta[7:])
            print("Saving figure...")

            plt.figure(figsize=(14,8))
            plt.subplot(311)
            plt.plot(self.SelectedTime, self.SelectedFlux-DetrendedFlux, "ko", label="Flux")
            plt.plot(self.SelectedTime, TransitModelFlux, "r-", lw=2, label="Detrend Flux")
            plt.xlim([8905.80, 8905.90])
            plt.subplot(312)
            plt.plot(self.SelectedTime, self.SelectedFlux-DetrendedFlux, "ko", label="Flux")
            plt.plot(self.SelectedTime, TransitModelFlux, "r-", lw=2, label="Detrend Flux")
            plt.xlim([8915.250, 8915.350])
            plt.subplot(313)
            plt.plot(self.SelectedTime, self.SelectedFlux-DetrendedFlux, "ko", label="Flux")
            plt.plot(self.SelectedTime, TransitModelFlux, "r-", lw=2, label="Detrend Flux")
            plt.xlim([8918.37, 8918.50])

            plt.tight_layout()
            plt.savefig("BestFit.png")
            plt.close('all')
            self.BestResidual = SumResidual

        ChiSquare = -(0.5*np.sum(Residual*Residual)/(STD*STD) + 2.0*np.pi*self.numDataPoints*STD*STD)
        return ChiSquare


    def QuickFit(self, Target, TransitSearch, CaseNumber):
        '''
        This method performs the quick fit

        '''

        NCols = 0
        for NightIndex in self.AllNightIndex:
            NCols+=len(TransitSearch.AllCombinationBasis[NightIndex])
        NCols*=2

        #Construct Basis
        self.BasisMatrix = np.zeros((len(self.SelectedTime),NCols))

        StartIndex = 0
        self.NumNights = len(self.BreakLocation) + 1

        AssignCol = 0
        for CurrentNight in range(self.NumNights):
            if CurrentNight == len(self.BreakLocation):
                StopIndex = len(self.SelectedTime)
            else:
                StopIndex = self.BreakLocation[CurrentNight]+1

            for Col in TransitSearch.AllCombinationBasis[CurrentNight]:
                print(AssignCol, StartIndex, StopIndex)
                MeanValue = np.mean(self.SelectedData[StartIndex:StopIndex,Col])
                self.BasisMatrix[StartIndex:StopIndex, AssignCol] = self.SelectedData[StartIndex:StopIndex,Col]-MeanValue
                self.BasisMatrix[StartIndex:StopIndex, AssignCol+1] = np.power(self.SelectedData[StartIndex:StopIndex,Col]-MeanValue,2)
                AssignCol += 2
            StartIndex = StopIndex

        nWalkers = 100

        self.Parameters = ["T0", "Period", "a_Rs", "Rp_Rs", "b", "q1", "q2"]

        T0_Init = np.random.normal(self.CurrentT0, 0.001, nWalkers)
        Period_Init = np.random.normal(self.CurrentPeriod, 0.2, nWalkers)
        a_Rs_Init = np.random.normal(30.0, 3.0, nWalkers)
        Rp_Rs_Init = np.random.normal(0.01, 0.002, nWalkers)
        b_Init = np.random.normal(0.5, 0.01, nWalkers)
        q1_Init = np.random.uniform(0.0, 1.0, nWalkers)
        q2_Init = np.random.uniform(0.0, 1.0, nWalkers)

        Decorrelator = np.random.normal(0,100,(nWalkers,NCols))

        StartingGuess = np.column_stack((T0_Init, Period_Init, a_Rs_Init, Rp_Rs_Init, b_Init, q1_Init, q2_Init, Decorrelator))

        #intiate batman
        params = batman.TransitParams()
        params.limb_dark = "quadratic"       #limb darkening model

        _, nDim = np.shape(StartingGuess)

        self.BestResidual = np.inf

        input("Before starting MCMC...")
        sampler = emcee.EnsembleSampler(nWalkers, nDim, self.Likelihood, args=[params], threads=8)
        state = sampler.run_mcmc(StartingGuess, 10000,  progress=True)
        #Now make the folded plot...
