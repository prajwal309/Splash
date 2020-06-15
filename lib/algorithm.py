'''
This file containts the run algorithm for the transit search
'''

import matplotlib.pyplot as plt
import itertools
import numpy as np

from .splash import Target
from .Functions import ParseFile

class GeneralTransitSearch:
    """
    Description
    -----------
    This is a generalized transit search class.

    Parameters Name
    --------------
    Target: The target base class from splash


    Output
    --------------
    """
    def __init__(self):
        self.transitSearchParam = ParseFile("SearchParams.ini")
        self.T0StepSize = float(self.transitSearchParam["TStepSize"])/(24.0*60.0)
        self.TDurStepSize = float(self.transitSearchParam["TDurStepSize"])/(24.0*60.0)
        self.TDurLower, self.TDurHigher = [float(Item)/(24.0*60.0) for Item in self.transitSearchParam["TransitDuration"].split(",")]
        self.TDurValues = np.arange(self.TDurLower, self.TDurHigher+self.TDurStepSize, self.TDurStepSize)



class LinearSearch(GeneralTransitSearch):
    '''
    Description
    ------------
    This method implements the linearized search for the transits


    Input Parameters
    ---------------
    This method does a night by night basis for the function

    method="svd"
    Linearize the flux as a function of the
    Yields
    ---------------

    '''

    def __init__(self, Target, method="SVD"):
        GeneralTransitSearch.__init__(self)
        self.getParamCombination(Target)

        input("now starting night by nigh")
        for NightNum in range(Target.NumberOfNights):
            print("Now running Night:", NightNum+1)
            self.RunNight(Target, NightNum)


    def getParamCombination(self, Target):
        '''
        This method will yield different combination of the parameters to be used as the basis vector.

        '''

        BasisCombination = []

        Range = np.arange(len(Target.ParamNames))
        for i in range(1,int(self.transitSearchParam["Combination"])+1):
            Basis = [list(itertools.combinations(Range,i))]
            BasisCombination.extend(Basis[0])
        BasisCombination = BasisCombination


    def constructBasis(self, Target):
        '''
        Constructs the basis function based on the parameter name
        '''
        self.BasisMatrix = np.zeros(Target)
        CPU_Pool = mp.Pool(NUM_CORES)



    def SaveChiSquare():
        CPU_Pool = mp.Pool(NUM_CORES)


    def RunNight(self, Target, NightNum):
        '''
        This functions find


        Input
        =============
        Target is the Speculoos target class which gives
        access to the data.

        NightNum: The number night is the index of the night
        which is to be run.

        Yields
        ==============
        Yield 2D chi squared map (M,N) for each night for the
        where M is the length of T0 Values and N is the length
        of Transit duration arrays.
        '''

        #Target
        self.CurrentData = Target.DailyData[NightNum]
        self.CurrentTime = CurrentData[:,0]
        self.CurrentFlux = CurrentData[:,1]

        T0_Values = np.arange(self.CurrentTime[0],self.CurrentTime[-1], self.T0StepSize)
        ChiSquareData = np.zeros((len(T0_Values), len(self.TDurValues)))
        BasisMatrix = ConstructBasisMatrix(T0_Value, )

    def ConstructBasisMatrix():
        pass





class PeriodicSearch:
    '''
    This function utilizes the linear search information and
    Input Parameters
    ---------------
    This method does a night by night basis for the function

    Yields
    ---------------
    '''

    pass
