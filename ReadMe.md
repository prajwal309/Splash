Code maintained by: Prajwal Niraula
Insitute: MIT
Contact: pniraula@mit.edu/prajwalniraula@gmail.com

####################################################################################################
####################################################################################################

This is a code repository for planet search in the SPECULOOS data. Version 0.2

Uses multi-cores if available.

Written in python v3.5.2
Note python v2.7.x will fail with this code.

Library Dependencies:
numpy: version used --- 1.16.2
scipy: version used --- 1.3.0.
##corner: version used --- 2.0.1 for corner plot diagnositics
emcee: version used --- 2.1.0 for MCMC
batman: version used --- 2.4.6 for transit injection
fpdf: version used --- 1.7.2 for creating the report
####################################################################################################
####################################################################################################

Run by following command
python3 -W ignore Launch.py -m 1/2/3 -o OutputFolder -i Location_of_LC

For batch run, I run the following:
python3 main.py, which looks for the data in data folder. The data is expected to be in the same format as the

####################################################################################################
####################################################################################################

The input file are the light curve products for ARTEMIS:
JD_UTC, Flux, Err, XShift, YShift, FWHM_X,  FWHM_Y, FWHM, SKY, AIRMASS, ExpTime = np.loadtxt(LC_Location, unpack=True)

There is a folder crawler that generates the best combination of light curve.
####################################################################################################
####################################################################################################


The data file is standard

-W ignore any warnings that are part of MCMC in latter half.

Note these arguments are optional and overwrite any redundant information provided in SearchParams.ini
##-m 1--> Scipy minimize --- Do not use this. Solution usually does not converge
-m 2--> SVD
##-m 3--> MCMC  --> Usable but not preferable

SearchParams.ini allows to optimize the more parameters which are follows:

TransitDuration:0.4,1.5                      #Transit duration in hours. Assign values between 20 minutes to 300 minutes
TStepNorm:5                                  #Take step of 1/Nth of Transit duration where N is the number
Params:t,xy,fwhm,sky,airmass                 #Possible options: t,xy,fwhm,sky,airmass. The parameters to consider for fitting. t is time,xy is shift in the distance of the centroid, fwhm, sky,wv is water vapor, a is the airmass
Combination:2                                #Maximum combination of parameter to consider. Using 2 parameters is strongly suggested.
PolynomialOrder:2                            #Order of polynomial to be used. Higher than 3 does not seem to be working well.
Method:2                                     #Method - 1 is scipy minimize, 2 is SVD least square implementation, 3 is MCMC. Using SVD is highly encouraged
NumTrials:2000                               #Number of function evaluation for scipy, Number of trials for MCMC.
NCPUs:-1                                     #-1 is default to use all the number of cores --> multiprocessing.cpu_count; 1-32 is to that many number of cores.
FancyPlot:1                                  #Use better fonts. 1 is turn on, 0 is turn off.
FolderName:SVD_Results                       #Name of the run ID.
NumPromisingCases:50                         #Number of plot to be made for the promising case
LC_Location:data/global_LC.txt               #Location of the light curve folder
FinalMCMCSteps:500                          #Number of MCMC steps in finding the most significant cases. 500 in most cases is enough. Using higher valu
NumCombination:50                            #Number of top transit pair to consider

####################################################################################################
####################################################################################################

Currently the promise factor, which ranks how promising a transit is calculated as following:

LocalSignificance = (ReducedChiSq - np.mean(ReducedChiSq))/np.std(ReducedChiSq) --->Performed for each night chunk


Each potential transit atre rated by metric called Promise Factor
SNRArray = TransitDepth/TransitDepthUncertainty
PromiseFactor = (SNRArray/(LocalSignificance)**2.0)

Two transits are combined using the following combination:

Combined Promise Factor:
SimilarityValue = PromiseValue1**0.3+PromiseValue2**0.6



####################################################################################################
####################################################################################################


The OutFolder has the following structure:
GlobalAnalysis-->This if folder that has the best potential planetary transit candidates that is found that combines two transits.
Data---> Where data for each nights are stored
LeastChiSqModels--> Figure representing least chi square models each night.
Per Night Analysis--> Contains the actual data and model for promising models
PerNightPdfSummary--> Contains pdf file with summary for each night with summary and models arranged in order of their promise factor.


####################################################################################################
####################################################################################################
