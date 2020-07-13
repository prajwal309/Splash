import numpy as np
import matplotlib.pyplot as plt
import batman


def Residual(NumParam="11", BatmanParam):
    '''
    Residual function
    '''

    if NumParam='11':
        theta = np.array([params['a11'].value,params['a21'].value])
    elif NumParam='21':
        theta = np.array([params['a11'].value,params['a21'].value])
    if NumParam='11':
        theta = np.array([params['a11'].value,params['a21'].value])

    #calculate the inclination
    Inc = np.rad2deg(np.arccos(LMparams['b']/LMparams['a_Rs']))

    params = batman.TransitParams()
    params.t0 = LMparams['T0']                        #time of inferior conjunction
    params.per = LMparams['Period']                   #orbital period
    params.rp = LMparams['Rp_Rs']                     #planet radius (in units of stellar radii)
    params.a = LMparams['a_Rs']                       #semi-major axis (in units of stellar radii)
    params.inc = Inc                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.u = [LMparams['a_Rs'] ]                 #limb darkening coefficients [u1, u2]
    params.limb_dark = "linear"

    #Model the Transit
    m = batman.TransitModel(params, self.SelectedTime)    #initializes model
    TransitModelFlux = m.light_curve(params)          #calculates light curve

    #Subtract the offset
    StartIndex = 0
    for CurrentNight in range(self.NumNights):
        if CurrentNight == len(self.BreakLocation):
            StopIndex = len(self.SelectedTime)
        else:
            StopIndex = self.BreakLocation[CurrentNight]+1
    pass

def Model1_11(Period_Orig, T0_Orig, Basis):
    '''
    One night and single parameter for each night
    '''

    #    assert the dimensions are right

    LMparams = Parameters()
    LMparams.add(name='Period', value=1.2, min=2.0, max=5.0)
    LMparams.add(name='T0', value=1.0, min=0, max=PeriodOrig/2.0)
    LMparams.add(name='a_Rs', value=10.0, min=2.0, max=100000.0)
    LMparams.add(name='Rp_Rs', value=10.0, min=2.0, max=100000.0)
    LMparams.add(name='b', value=0.5, min=0, max=1.2)
    LMparams.add(name='u', value=0.5, min=0, max=1.0)
    LMparams.add(name='a11', value=0.0, min=-1e6, max=1e6)
    LMparams.add(name='a21', value=0.0, min=-1e6, max=1e6)

    BatmanParam = batman.TransitModel()

    pass


def Model2_12(Params):
    pass


def Model2_21(Params):
    pass


def Model2_22(Params):
    pass


def Model3_111(Params):
    pass


def Model3_112(Params):
    pass

def Model3_121(Params):
    pass

def Model3_211(Params):
    pass

def Model3_122(Parama):
    pass

def Model3_212():
    pass


def Model3_221():
    pass


def Model3_222():
    pass


















pass
