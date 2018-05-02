"""
Library for simulating scattering onto a detector according to Klein-Nishina equation
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.constants import *


#I know global variables are evil, but these are constants which i really want to keep consistent
mu = electron_mass*c*c/electron_volt
ro = physical_constants['classical electron radius'][0]



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
    xpos = xpos or 0.
    ypos = ypos or 0.
    theta = np.pi - np.arctan(np.sqrt(np.square(x - xpos) + np.square(y - ypos))/z)
    phi = np.arctan((y-ypos) / (x - xpos))
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
    compton = 0.5*ro*ro*np.square(k/ko)*(k/ko + ko/k - 2.*np.square(np.sin(theta)*np.cos(phi)))
    return compton 



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
    thompson= ro*ro*(1. - np.square(np.sin(theta)*np.cos(phi)))
    return thompson



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


