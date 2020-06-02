import numpy as  np
import matplotlib.pyplot as plt
import itertools
import os
import multiprocessing as mp
from .sampler import FitFunction, SVD_Solver
from .Functions import SigmaClip
from warnings import warn
import matplotlib as mpl
from scipy.stats import binned_statistic
from astropy.time import Time
mpl.use('agg')


def Find_Closest(Array, Value):
    #Finds the location of the closest value from an array of values
    Diff = np.abs(Array-Value)
    Index = np.argmin(Diff)
    return Index


def CloseEnough(Value1, Value2, Tol=1e-8):
    #Checks if the two scalar values are close enough to one another
    if abs(Value1-Value2)<Tol:
        return True
    else:
        return False
    return Index


def BoxFit(Time, T0=None, TDur=None, Delta=100):
    '''
    Create a transit box model using box
    '''
    TransitIndex = np.abs((Time-T0))<TDur/2
    TransitModel = np.zeros(len(Time))
    TransitModel[TransitIndex]-=Delta
    return TransitModel


def SingleEventSearch(ParamValues, ParamName, SpecParam, SaveFolder):
        '''
        This function identifies the best transit candidates for each night using SVD
        #############################################################################
        Input
        ==================
        ParamValues: The values that are to be used for parameters
        ParamName: Name of the parameter to identify which basis vector is used.
        SpecParam:The parameters from Search.ini which contains the transit duration
        OutputDir: Directory for the outputs to be stored.
        '''

        #Lower and upper bound for the transit duration in minutes
        TDur_Values = [float(Item) for Item in SpecParam['TransitDuration'].split(",")]

        #Convert transit duration in to days
        TDur_Values = [float(i)/(24.*60.0) for i in TDur_Values]
        TDurStepSize = float(SpecParam['TDurStepSize'])/(24.*60.0)

        #Create an array for transit duration
        TDurArray = np.arange(TDur_Values[0],TDur_Values[1]+TDurStepSize, TDurStepSize)


        #Stepsize of the T0
        T0StepSize =float(SpecParam['TStepSize'])/(24.0*60.0)

        if len(TDurArray)<2:
            raise NameError("TStep should be smaller than the transit duration")

        #Basis vectors to consider for detrending
        Parameters2Consider = SpecParam['Params'].split(",")

        #Number of parameters to consider
        NumParameters = int(SpecParam['Combination'])
        NumParameters = min([len(Parameters2Consider), NumParameters])
        DetrendParam = SpecParam['Params'].split(",")

        BasisVector = np.zeros((len(ParamValues[:,0]), len(DetrendParam)))


        for Count, ParamItem in enumerate(DetrendParam):
            if "T" == ParamItem.upper() or "JD" == ParamItem.upper():
                ColumnIndex = 0
            else:
                ColumnIndex = np.where(ParamItem.upper() == ParamName)
                if len(ColumnIndex)<0:
                    raise NameError("Mismatch in the header and the columns")
                else:
                    ColumnIndex = ColumnIndex[0][0]
                    print(ColumnIndex)

            BasisVector[:,Count] = ParamValues[:,ColumnIndex]



        #the order of the polynomial for detrending
        PolyOrder = int(SpecParam['PolynomialOrder'])

        #Two is the preferred number of parameters
        if NumParameters>3:
            warn("Use three parameters at most for the best results")

        if PolyOrder>3:
            warn("Use use larger than three order polynomial for the best results")

        #Number of CPU cores to be used
        NCPUs = int(SpecParam['NCPUs'])

        if NCPUs==-1:
            NUM_CORES = mp.cpu_count()
        elif NCPUs>0 and NCPUs<64:
            NUM_CORES = int(NCPUs)

        print("Using %d cores." %NUM_CORES)


        ParamCombinations = []
        for i in range(1,NumParameters+1):
            ParamCombinations.extend(list(itertools.combinations(DetrendParam,i)))

        print("The parameter combination is given by:")
        print(len(ParamCombinations))


        #Find the segments in the data that are continuously observed with gaps less than 2.4 hours
        Time = ParamValues[:,0]
        Flux = ParamValues[:,1]

        Diff1D = np.diff(Time)
        Index = np.concatenate((np.array([False]), Diff1D>0.2))
        Locations = np.where(Index)[0]

        #To mark the starting chunk of the data
        Start = 0
        for ChunkCount in range(len(Locations)+1):
            if ChunkCount<len(Locations):
                Stop = Locations[ChunkCount]
            else:
                Stop = len(Flux)

            TimeChunk = Time[Start:Stop]
            FluxChunk = Flux[Start:Stop]
            FluxChunk-=np.mean(FluxChunk)
            BasisChunk = BasisVector[Start:Stop, :]

            #SigmaClip the data
            OutliersIndex = SigmaClip(TimeChunk, FluxChunk, SigmaValue=5.0)

            if np.sum(OutliersIndex)>0:
                T0_min = int(min(TimeChunk))
                plt.figure()
                plt.plot(TimeChunk[~OutliersIndex]-T0_min, FluxChunk[~OutliersIndex], "ko")
                plt.plot(TimeChunk[OutliersIndex]-T0_min, FluxChunk[OutliersIndex], "ro", label="Outliers")
                plt.legend(loc=0)
                plt.title(T0_min)
                plt.tight_layout()
                plt.savefig(SaveFolder+"/"+str(T0_min)+"_Outliers.png")
                plt.close('all')


            #Remove the outliers
            TimeChunk = TimeChunk[~OutliersIndex]
            FluxChunk = FluxChunk[~OutliersIndex]

            BasisChunk = BasisChunk[~OutliersIndex,:]


            #The length for the vector
            LENGTH = len(TimeChunk)

            #Updating Start
            Start = Stop

            T0_Range = np.arange(min(TimeChunk), max(TimeChunk), T0StepSize)
            T0_Param_Product = itertools.product(T0_Range, ParamCombinations)

            #Save 2 dimensional array
            ResidualArray = np.ones((len(T0_Range), len(TDurArray)))*1e300
            ReducedChiSqrArray = np.ones((len(T0_Range), len(TDurArray)))*1e300
            TransitDepthArray = np.ones((len(T0_Range), len(TDurArray)))*1e300

            STRSignalArray = np.ones((len(T0_Range), len(TDurArray)))*1e300
            LocalSTDArray = np.ones((len(T0_Range), len(TDurArray)))*1e300

            #The number of expected operations
            N_OPERATIONS = len(ParamCombinations)

            #Flags to determine the daily best transit candidate model
            BestChiSqr = np.inf
            BestSTD = np.inf

            #number of cotrending basis vectors being used
            CBVLength = NumParameters*(PolyOrder+1)+3

            ParamMatrix = np.zeros((len(T0_Range),CBVLength))
            ParamNameMatrix = np.chararray((len(T0_Range),NumParameters))

            #The matrix for each night
            Residuals = np.zeros((len(T0_Range), len(TDurArray)))
            Parameters = np.zeros((len(T0_Range), len(TDurArray)))
            Uncertainty = np.zeros((len(T0_Range), len(TDurArray)))

            print("Just try calling ")

            input("Calling SVD solver next")
            FitVariables = BasisChunk[:,0:2]
            _, LengthModelParam = np.shape(FitVariables)

            print("The lenght of th parameter is ", LengthModelParam)
            input("Wait here...")
            SVD_Solver(TimeChunk, FluxChunk, FitVariables, PolyOrder, LengthModelParam, T0_Range, TDurArray)

            input("Just trying the fit variables")

            for counter in range(int(N_OPERATIONS/NUM_CORES)+1):
                Tasks = []
                CPU_Pool = mp.Pool(NUM_CORES)
                T0_Local_List = []
                TDur_Local_List = []
                TempParamName_List = []

                #Starting the multiprocessing
                for MPICount in range(NUM_CORES):
                    print("The value of MPICount:", MPICount)

                    try:
                        T0_Value, ModelParameters = next(T0_Param_Product)
                    except:
                        break

                    for ModelParam in ModelParameters:
                        FitVariables = np.zeros((LENGTH,len(ModelParameters)))
                        TempParamName = np.chararray(len(ModelParameters))
                        for ModelCount,param in enumerate(ModelParameters):
                            print("The model of the count is::", ModelCount)
                            print("The parameter is given::", param)
                            input("Wait here...")
                            if "T" in param.upper():
                                FitVariables[:,ModelCount] = TimeChunk - np.mean(TimeChunk)
                                TempParamName[ModelCount]='T'
                            else:
                                print("All the model paratmers are given by::", ModelParam)
                                print("The parameter in question is::", param.upper())
                                raise NameError("The parameter to be considered is %s, and it could not be parsed." %(param))
                    LengthModelParam = len(ModelParameters)

                    input("Wait here for the test... ")
                    #Use SVD to find the least square fit
                    Tasks.append(CPU_Pool.apply_async(SVD_Solver,(TimeChunk, FluxChunk, FitVariables, PolyOrder, LengthModelParam, T0_Range, TDurArray)))
                    T0_Local_List.append(T0_Value)
                    TempParamName_List.append(TempParamName)


                CPU_Pool.close()
                CPU_Pool.join()


                TempVector4SNR = []
                #Get the results and check for the minimum Chisquare value
                for Index, Task in enumerate(Tasks):
                    NewResidual, NewCBVs, NewUncertainty = list(Task.get())
                    T0_Current,TDur_Current, TDepth_Current = NewCBVs[-3:]

                    if abs(TDepth_Current)<1e-4:
                        TDepth_Current = 0.0

                    T0_Index = Find_Closest(T0_Range, NewCBVs[-3])

                    #Keep the one that has the best transitdepth/uncertainty
                    CurrentSNR = TDepth_Current/np.abs(NewUncertainty)
                    TempVector4SNR.append(CurrentSNR)

                    #If the new residual is smaller than previous residual, then update the best parameter
                    if NewResidual<ResidualArray[T0_Index] and NewCBVs[-1]>0:
                        IsItBest = NewResidual<min(ResidualArray)

                        #Update the SNRTrackerArray
                        SNRTrackerArray[T0_Index] = CurrentSNR
                        ResidualArray[T0_Index] = NewResidual

                        #Reconstruct the background

                        #remake the basis vector
                        LengthModelParam = len(TempParamName_List[Index])
                        TempFitVariables = np.zeros((len(TimeChunk),LengthModelParam))

                        for ModelCount,param in enumerate(TempParamName_List[Index]):
                            param = str(param)[2:-1]
                            if "T" in param.upper():
                                TempFitVariables[:,ModelCount] = TimeChunk - np.mean(TimeChunk)
                            elif "X"==param.upper():
                                XShiftAssign = XShift_Chunk -np.mean(XShift_Chunk)
                                TempFitVariables[:,ModelCount] = XShiftAssign/np.std(XShiftAssign)
                            elif "Y" == param.upper():
                                YShiftAssign = YShift_Chunk -np.mean(YShift_Chunk)
                                TempFitVariables[:,ModelCount]  = YShiftAssign/np.std(YShiftAssign)
                            elif "D" == param.upper():
                                XY = np.sqrt(XShift_Chunk*XShift_Chunk+YShift_Chunk*YShift_Chunk)
                                XY-= np.mean(XY)
                                XY=XY/np.std(XY)
                                TempFitVariables[:,ModelCount] = XY
                            elif "F" in param.upper():
                                FWHW_Assign = FWHM_Chunk - np.mean(FWHM_Chunk)
                                TempFitVariables[:,ModelCount] = FWHW_Assign/np.std(FWHW_Assign)
                            elif "S" in param.upper():
                                SKY_Assign = SKY_Chunk - np.mean(SKY_Chunk)
                                TempFitVariables[:,ModelCount] = SKY_Assign/np.std(SKY_Assign)
                            elif "A" in param.upper():
                                AIRMASS_Assign = AIRMASS_Chunk - np.mean(AIRMASS_Chunk)
                                TempFitVariables[:,ModelCount] = AIRMASS_Assign/np.std(AIRMASS_Assign)
                            else:
                                print("Error when parsing parameter::", param.upper())
                                raise NameError("The parameter to be considered is %s, and it could not be parsed." %(param))

                        #print("The Length of the model parameter is given by::", LengthModelParam)
                        BackgroundContinuum = np.zeros(len(TimeChunk))
                        for i_bac in range(LengthModelParam):
                            BackgroundContinuum += np.polyval(NewCBVs[i_bac*(PolynomialOrder+1):(i_bac+1)*(PolynomialOrder+1)], TempFitVariables[:,i_bac])

                        T0_Value_Result = NewCBVs[-3]
                        TDur_Value_Result = NewCBVs[-2]
                        Delta_Value_Result = NewCBVs[-1]

                        TransitModel = BoxFit(TimeChunk, T0=T0_Value_Result, TDur=TDur_Value_Result, Delta=Delta_Value_Result)
                        Model = TransitModel+BackgroundContinuum

                        LocalSTD = np.std(FluxChunk-Model)
                        LocalSTDArray[T0_Index] = LocalSTD
                        ReducedChiSqr = NewResidual/(STD*STD*LENGTH)

                        ReducedChiSqrArray[T0_Index] = ReducedChiSqr
                        #Check the strength of the signal
                        BestSignalStrength = np.sum(np.abs(TransitModel))
                        STRSignalArray[T0_Index] = BestSignalStrength

                        #Populate only the relevant locations
                        ParamNameMatrix[T0_Index,:len(TempParamName_List[Index])] =  TempParamName_List[Index]

                        ParamMatrix[T0_Index,:len(NewCBVs)] = NewCBVs

                        #Save if the best criterion is meet
                        if IsItBest:
                            BestIndex = np.zeros(len(T0_Range)).astype(np.bool)
                            BestIndex[T0_Index] = True
                            BestTransitModel = TransitModel
                            BestContinuum = BackgroundContinuum
                            BestDepthUncty = NewUncertainty

                            Best_T0_Value = T0_Local_List[Index]
                            Best_TDur = TDur_Value_Result
                            BestDepth = Delta_Value_Result
                            BestSTD = LocalSTD

                            CBVS_BestBest = NewCBVs
                            BestParameters = ''
                            for val in TempParamName_List[Index]:
                                BestParameters+=str(val)
                            BestParameters = BestParameters.replace("b","").replace("'","")

                            #Save the best basis combination for parameters
                            T0_Int = int(Best_T0_Value)
                            BestBasisLocation = SaveFolder+"/Data/BestBasis"+"_"+str(T0_Int)+".txt"
                            if not(os.path.exists(SaveFolder+"/Data")):
                                os.system("mkdir %s" %(SaveFolder+"/Data"))

                            BasisFile = open(BestBasisLocation, 'w+')
                            BasisFile.write(BestParameters)
                            BasisFile.close()

                            FlattenedLC = FluxChunk - Model
                            LCSaveName = SaveFolder+"/Data/"+"LC_"+str(ChunkCount+1).zfill(4)+".csv"
                            np.savetxt(LCSaveName,np.transpose((TimeChunk, FlattenedLC)), delimiter=',')


                        #Update the parameters
                        ResidualArray[T0_Index] = NewResidual
                        UncertaintyArray[T0_Index] = NewUncertainty
                        TransitDepthArray[T0_Index] = NewCBVs[-1]
                        TDurArray[T0_Index] = NewCBVs[-2]


            #Saving data for the night
            DataLocation = SaveFolder+"/Data"
            if not(os.path.exists(DataLocation)):
                os.system("mkdir %s" %(DataLocation.replace(" ", "\ ")))


            #Remove missed T0 values which are missed
            RemovePoints = ReducedChiSqrArray>1e50                              #1e50 is less than 1e300 which which the array is initialized
            T0_Range = T0_Range[~RemovePoints]
            TransitDepthArray =  TransitDepthArray[~RemovePoints]
            STRSignalArray = STRSignalArray[~RemovePoints]
            ReducedChiSqrArray = ReducedChiSqrArray[~RemovePoints]
            LocalSTDArray = LocalSTDArray[~RemovePoints]
            UncertaintyArray = UncertaintyArray[~RemovePoints]
            TDurArray = TDurArray[~RemovePoints]

            ParamMatrix = ParamMatrix[~RemovePoints]
            ParamNameMatrix = ParamNameMatrix[~RemovePoints]
            BestIndex = BestIndex[~RemovePoints]

            #Save the array
            SaveNamePlot = DataLocation+"/Night"+str(ChunkCount+1)+".csv"
            np.savetxt(SaveNamePlot, np.transpose((T0_Range, TransitDepthArray, STRSignalArray, ReducedChiSqrArray, LocalSTDArray, UncertaintyArray, TDurArray)), delimiter=",", header="T0_Range, TransitDepth, Signal Strength, Reduced Chi Square Value, Local STD Array, Uncertainty in Transit Depth, Transit Duration" )

            SaveNameParam = DataLocation+"/Night"+str(ChunkCount+1)+".param"


            #Write the parameter values to the file
            #remove the zeros
            print("Now writing the parameters to the files:"+SaveNameParam)
            SaveFileParam = open(SaveNameParam,"w+")

            for x,y in zip(ParamMatrix, ParamNameMatrix):
                NumParametersU = int((len(x)-3)/(PolynomialOrder+1))
                #print("You started with:",x)
                x_reversed = x[::-1]
                #print(x_reversed)
                for CheckCounter in range(NumParametersU):
                    Test = max(np.abs(x_reversed[CheckCounter*(PolynomialOrder+1):(CheckCounter+1)*(PolynomialOrder+1)]))<1e-10
                    if Test:
                        NumParametersU-=1

                x=x[:NumParametersU*(PolynomialOrder+1)+3]
                #Store the text to be written in the file
                Text2Write=""
                for ItemCounter,Item in enumerate(x):
                    Text2Write+=str(Item)+","


                NumParameters2Write = int((len(x)-3)/(PolynomialOrder+1))
                y = y[:NumParameters2Write]
                for ItemCounter, Item in enumerate(y):
                    Text2Write+=str(Item)[2:-1]
                    if ItemCounter<len(y)-1:
                        Text2Write+=','
                SaveFileParam.write(Text2Write+"\n")
            SaveFileParam.close()




            print("Updating Figure")

            #Binning the plot

            #Binning size
            BinSize = 5.0 #minutes

            #The time differennce in  days
            TimeDifference = max(TimeChunk) - min(TimeChunk)
            NumBins = int(TimeDifference*86400.0/(BinSize*60.0))

            BinnedTime = binned_statistic(TimeChunk, TimeChunk, statistic='mean', bins=NumBins)[0]
            BinnedFlux = binned_statistic(TimeChunk, FluxChunk, statistic='median', bins=NumBins)[0]

            #Estimating the errorbar
            NumInABin = len(TimeChunk)/len(BinnedTime)
            ScaledSTD = BestSTD*1.0/np.sqrt(NumInABin)
            ErrorSTD = np.ones(len(BinnedTime))*ScaledSTD





            #Plot the best Model
            T0_Plot = int(min(TimeChunk))


            fig, ax = plt.subplots(2,2,figsize=(20,14), sharex=True)
            ax[0,0].plot(TimeChunk - T0_Plot, FluxChunk -np.mean(FluxChunk), color="cyan", marker="o", markersize=3, linestyle="None", label="Data")
            ax[0,0].errorbar(BinnedTime-T0_Plot, BinnedFlux, yerr=ErrorSTD, zorder=10, color="black", ecolor="black", capsize=3, linestyle="None", marker="d")
            Model = BestContinuum + BestTransitModel
            ax[0,0].plot(TimeChunk - T0_Plot, Model - np.mean(Model), "g-", zorder=5, lw=4, label="Combined Model")
            ax[0,0].plot(TimeChunk - T0_Plot, BestTransitModel , "r-", zorder=15, lw=4, label="Transit Model")
            ax[0,0].set_ylabel("Normalized Flux", fontsize = 30)
            ax[0,0].set_xlim(min(T0_Range - T0_Plot)-0.01, max(T0_Range-T0_Plot)+0.01)
            ax[0,0].tick_params(which='both', direction='in')
            ax[0,0].legend(loc=0)
            ax[0,0].set_title("Best $\\chi^2$ Model")


            #SNR
            #plt.subplot(223)
            if MinimizeMethod==2 or MinimizeMethod==3:
                ax[0,1].plot(T0_Range - T0_Plot, TransitDepthArray/UncertaintyArray, marker="3",  markersize=10, linestyle=':', color="navy", )
                ax[0,1].plot(T0_Range[BestIndex] - T0_Plot, TransitDepthArray[BestIndex]/UncertaintyArray[BestIndex], marker="3", markersize=20, linestyle='None', color="red", )
            else:
                #scipy minimize
                ax[0,1].plot(T0_Range - T0_Plot, STRSignalArray/STD, marker="+",  markersize=10, linestyle=':', color="navy", )
                ax[0,1].plot(T0_Range[BestIndex] - T0_Plot, STRSignalArray[BestIndex]/STD, marker="+", markersize=20, linestyle='None', color="red", )

            ax[0,1].set_xlabel("Time (JD) --" +str(T0_Plot), fontsize = 20)
            ax[0,1].set_ylabel("SNR", fontsize = 20)
            ax[0,1].tick_params(which='both', direction='in')
            ax[0,1].set_xlim(min(T0_Range - T0_Plot)-0.01, max(T0_Range-T0_Plot)+0.01)

            #Transit Depth
            if MinimizeMethod==2 or MinimizeMethod==3:
                ax[1,0].errorbar(T0_Range - T0_Plot, TransitDepthArray, yerr=UncertaintyArray, marker="2", capsize=3, elinewidth=2, linestyle=':', color="navy", ecolor="navy")
                ax[1,0].errorbar(T0_Range[BestIndex] - T0_Plot, TransitDepthArray[BestIndex], yerr=UncertaintyArray[BestIndex], marker="2", markersize=15, capsize=3, elinewidth=2, linestyle='None', color="red", ecolor="red", label="Best Model")
            else:
                ax[1,0].plot(T0_Range - T0_Plot, TransitDepthArray, marker="2", linestyle=':', color="navy")
                ax[1,0].plot(T0_Range[BestIndex] - T0_Plot, TransitDepthArray[BestIndex],  marker="2", markersize=20, linestyle=':', color="red", label="Best Model")
            ax[1,0].set_xlabel("Time (JD) --" +str(T0_Plot), fontsize = 40)
            ax[1,0].set_ylabel("Transit Depth", fontsize = 30)
            ax[1,0].tick_params(which='both', direction='in')
            ax[1,0].legend(loc=1)
            ax[1,0].set_xlim(min(T0_Range - T0_Plot)-0.01, max(T0_Range-T0_Plot)+0.01)
            ax[1,0].set_ylim(min(TransitDepthArray), max([0.01, 1.3*max(TransitDepthArray)]))


            #ChiSquare
            ax[1,1].plot(T0_Range - T0_Plot, ReducedChiSqrArray,  marker="1", markersize=10,  linestyle=':', color="navy", )
            ax[1,1].plot(T0_Range[BestIndex] - T0_Plot, ReducedChiSqrArray[BestIndex], marker="1", markersize=20, linestyle='None', color="red", label="Best Model")
            ax[1,1].set_xlabel("Time (JD) --" +str(T0_Plot), fontsize = 40)
            ax[1,1].set_ylabel("$\\chi^2_\\nu$", fontsize = 30)
            ax[1,1].tick_params(which='both', direction='in')
            ax[1,1].set_xlim(min(T0_Range - T0_Plot)-0.01, max(T0_Range-T0_Plot)+0.01)
            ax[1,1].set_ylim(min(ReducedChiSqrArray)-0.1, min([max(ReducedChiSqrArray)+0.1,3.5]))
            plt.subplots_adjust(hspace=0, wspace=0.15)


            TitleText = "T0: "+str(round(Best_T0_Value,5))+"\nTDur: "+str(round(Best_TDur*24,2))+" Hours \n"+BestParameters

            plt.suptitle(TitleText, fontsize=35)

            #SaveName
            SaveFolderPerNight = SaveFolder+"/DailyModels"
            if not(os.path.exists(SaveFolderPerNight)):
                os.system("mkdir %s" %SaveFolderPerNight.replace(" ", "\ "))

            DateString = Time(2450000+int(min(T0_Range)),format='jd',scale='utc').iso.split(" ")[0]
            plt.savefig(SaveFolderPerNight+"/Night"+"_"+str(ChunkCount+1).zfill(4)+"Date_"+DateString+".png")
            plt.close('all')

            #Save the time series for all the data
            np.savetxt(SaveFolderPerNight+"/NightArray"+"_"+str(ChunkCount+1).zfill(4)+".txt", np.transpose((T0_Range, TransitDepthArray, UncertaintyArray, ReducedChiSqrArray)), header="Time, TransitDepth, Uncertainty, ReducedChiSqr" )

            if ChunkCount<len(Locations):
                print("\n\nStarting new night data: %s" %(str(ChunkCount+2)))
