"""
Simulate IPM detector readings
"""

import scatter
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


#Panel dimensions in mm (xmin, xmax, ymin, ymax) 
panels = {
    'T': (-5., 5., 10., 20.),
    'R': (10., 20., -5., 5.),
    'B': (-5., 5., -10., -20.),
    'L': (-10., -20., -5., 5.),
}

#Distance from the diode array center to the Si3N4 film in mm
film_distance = 20.


def mesh2d(bounds, points):
    return np.meshgrid(np.linspace(bounds[0], bounds[1], points), np.linspace(bounds[2], bounds[3], points))
    


def plot_detector(readings, cmap=None, xpos=None, ypos=None):
    """
    Parameters
    ----------
    readings : dict
        dictionary containing the names of panels and their respective readings
    """
    cmap = cmap or plt.get_cmap('viridis')
    X,Y,Z = [],[],[]
    for k,v in readings.items():
        x = np.linspace(panels[k][0], panels[k][1], v.shape[0])
        y = np.linspace(panels[k][2], panels[k][3], v.shape[1])
        x,y = np.meshgrid(x,y)
        X = np.concatenate((X, x.flatten()))
        Y = np.concatenate((Y, y.flatten()))
        Z = np.concatenate((Z, v.flatten()))
    plt.hexbin(X, Y, Z*1e28, cmap=cmap)
    ax = plt.gca()
    ax.set_facecolor(cmap(0.))
    for k,v in panels.items():
        plt.plot(v[:2], [v[2], v[2]], lw=3, c='w')
        plt.plot(v[:2], [v[3], v[3]], lw=3, c='w')
        plt.plot([v[0], v[0]], v[2:], lw=3, c='w')
        plt.plot([v[1], v[1]], v[2:], lw=3, c='w')
    plt.colorbar(label=r"$\frac{d\sigma }{d\Omega }\ (barn\ sr^{-1}\ atom^{-1})$")
    if xpos is not None:
        plt.scatter(xpos, ypos, edgecolors='w', facecolors='none')
    plt.xlabel('mm')
    plt.ylabel('mm')
    plt.show()

def differential_scattering(ko, l=None, Z=None, points=None, xpos=None, ypos=None):
    l = film_distance if l is None else l
    points = points or 100
    readings = {}
    for k,v in panels.items():
        x,y = mesh2d(v, points)
        theta,phi = scatter.transform_spherical(x, y, l, xpos, ypos)
        readings[k] = scatter.differential_intensity(theta, phi, ko, Z)
    return readings

def differential_compton_scattering(ko, l=None, Z=None, points=None, xpos=None, ypos=None):
    l = film_distance if l is None else l
    points = points or 100
    readings = {}
    for k,v in panels.items():
        x,y = mesh2d(v, points)
        theta,phi = scatter.transform_spherical(x, y, l, xpos, ypos)
        readings[k] = scatter.compton(theta, phi, ko, Z)
    return readings

def differential_thompson_scattering(ko, l=None, Z=None, points=None, xpos=None, ypos=None):
    l = film_distance if l is None else l
    points = points or 100
    readings = {}
    for k,v in panels.items():
        x,y = mesh2d(v, points)
        theta,phi = scatter.transform_spherical(x, y, l, xpos, ypos)
        readings[k] = scatter.thompson(theta, phi, ko, Z)
    return readings

def ipm_readings(ko, xpos, ypos, l=None, points = None, keys = None):
    """
    make up some ipm readings to go with a given photon energy and beam position. 

    Parameters
    ----------
    ko : float
        photon energy in electron volts
    xpos : float
        beam x-position in microns
    ypos : float
        beam y-position in microns
    """
    xpos,ypos = 1000.*xpos,1000.*ypos
    l = film_distance if l is None else l
    keys = keys or ['T', 'R', 'B', 'L']
    si = {k:v.sum() for k,v in differential_scattering(ko, l, 14, points, xpos, ypos).items()}
    n  = {k:v.sum() for k,v in differential_scattering(ko, l, 7 , points, xpos, ypos).items()}
    t  = {k:3.*si[k]/7. + 4.*n[k]/7. for k in si}
    return np.array([t[k] for k in keys])
