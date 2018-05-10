"""
Library for simulating scattering onto a detector according to Klein-Nishina equation
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.constants import *
from scipy.interpolate import interp1d





#I know global variables are evil, but these are constants which i really want to keep consistent
mu = electron_mass*c*c/electron_volt
ro = physical_constants['classical electron radius'][0]


class interpolator(dict):
    def __init__(self, inFN):
        self.inFN = inFN
        data = np.loadtxt(inFN, skiprows=1, delimiter=',')
        keys = map(int, open(inFN).readline().strip().split(',')[1:])
        for k,v in zip(keys,data[:,1:].T):
            self[k] = interp1d(data[:,0], v, kind='quadratic')

scatteringfunction = interpolator('scatteringfunction.txt')
formfactor         = interpolator('formfactor.txt')

    


def transform_spherical(x, y, z, xpos=None, ypos=None):
    """
    transform_spherical(x, y, z, **kw) converts coordinates to the scattering and azimuthal angles, phi and theta. assumes polarization in y plane

    Parameters
    ----------
    x : float or array
    y : float or array
    z : float or array
    xpos : float (optional)  beam center x position in same units as x,y,z. Defaults to zero
    ypos : float (optional)  beam center y position in same units as x,y,z. Defaults to zero

    Returns
    -------
    theta : float or array
    phi   : float or array
    """
    xpos = 0. if xpos is None else xpos
    ypos = 0. if ypos is None else ypos
    theta = np.pi - np.arctan(np.sqrt(np.square(x - xpos) + np.square(y - ypos))/z)
    #Polarization along Y
    #phi = np.arctan2(x-xpos , y - ypos)
    #Polarization along X
    phi = np.arctan2(y-ypos , x - xpos)
    return theta, phi

def compton(theta, phi, ko, Z=None):
    """
    compute the compton portion of the differential intensity of scattering for a given theta and phi according to the Klein Nishina equation

    Parameters
    ----------
    theta : float or array
    phi   : float or array
    ko    : float or array 
        The energy of the incident photon in eV 
    element : int (optional)
        Atomic number or the element scattering. If none is supplied, this function returns scattering from a free electron

    Returns
    -------
    d : float or array
        The differential scattering element for a given phi and theta
    """
    k = ko*mu / (mu + ko*(1 - np.cos(theta)))
    c = 0.5*ro*ro*np.square(k/ko)*(k/ko + ko/k - 2.*np.square(np.sin(theta)*np.cos(phi)))
    if Z is not None:
        wavelength = Planck*speed_of_light/(ko*electron_volt)*1e10
        x = np.sin(theta/2.)/wavelength
        c = c*scatteringfunction[Z](theta)
    return c



def thompson(theta, phi, ko, Z=None):
    """
    compute the thompson portion differential intensity of scattering for a given theta and phi according to the Klein Nishina equation

    Parameters
    ----------
    theta : float or array
    phi   : float or array
    ko    : float or array 
        The energy of the incident photon in eV 
    element : int (optional)
        Atomic number or the element scattering. If none is supplied, this function returns scattering from a free electron

    Returns
    -------
    d : float or array
        The differential scattering element for a given phi and theta
    """
    t = ro*ro*(1. - np.square(np.sin(theta)*np.cos(phi)))
    if Z is not None:
        wavelength = Planck*speed_of_light/(ko*electron_volt)*1e10
        x = np.sin(theta/2.)/wavelength
        t = t*np.square(formfactor[Z](theta))
    return t



def differential_intensity(theta, phi, ko, Z=None):
    """
    compute the total differential intensity of scattering for a given theta and phi according to the Klein Nishina equation

    Parameters
    ----------
    theta : float or array
    phi   : float or array
    ko    : float or array 
        The energy of the incident photon in eV 
    element : int (optional)
        Atomic number or the element scattering. If none is supplied, this function returns scattering from a free electron

    Returns
    -------
    d : float or array
        The differential scattering element for a given phi and theta
    """
    k = ko*mu / (mu + ko*(1 - np.cos(theta)))
    c = compton(theta, phi, ko, Z)
    t = thompson(theta, phi, ko, Z)
    return c + t


