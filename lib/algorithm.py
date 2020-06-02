import matplotlib.pyplot as plt
from .TransitSearch import *
from .Functions import ParseFile
from .LightCurve import Target


class TransitSearch():

    def __init__(self):
        self.TransitSearchParam = ParseFile("SearchParams.ini")


    def LinearSearch(self, Target):
        '''
        This method does a night by night basis for the function
        '''
        plt.figure()
        plt.plot(Target.DailyData[0][:,0], Target.DailyData[0][:,1], "ko")
        plt.show()
        input("Wait here...")
        pass


    def PeriodicSearch(self, Target):
        '''
        This function utilizes the linear search information and
        '''
        pass
