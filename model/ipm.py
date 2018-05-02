"""
Simulate IPM detector readings
"""

import scatter
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


#Panel dimensions (xmin, xmax, ymin, ymax)
panels = {
    'T': (-5., 5., 5., 15.),
    'R': (5., 15., -5., 5.),
    'B': (-5., 5., -5., -15.),
    'L': (-5., -15., -5., 5.),
}


def mesh2d(bounds, points):
    return np.meshgrid(np.linspace(bounds[0], bounds[1], points), np.linspace(bounds[2], bounds[3], points))
    


def plot_detector(readings, cmap=None):
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
    plt.hexbin(X, Y, Z, cmap=cmap)
    ax = plt.gca()
    ax.set_facecolor(cmap(0.))
    for k,v in panels.items():
        plt.plot(v[:2], [v[2], v[2]], lw=3, c='w')
        plt.plot(v[:2], [v[3], v[3]], lw=3, c='w')
        plt.plot([v[0], v[0]], v[2:], lw=3, c='w')
        plt.plot([v[1], v[1]], v[2:], lw=3, c='w')
    plt.show()

def differential_scattering(l, ko, Z=None, points=None, xpos=None, ypos=None):
    points = points or 100
    readings = {}
    for k,v in panels.items():
        x,y = mesh2d(v, points)
        theta,phi = scatter.transform_spherical(x, y, l, xpos, ypos)
        readings[k] = scatter.differential_intensity(theta, phi, ko, Z)
    return readings

def differential_compton_scattering(l, ko, Z=None, points=None, xpos=None, ypos=None):
    points = points or 100
    readings = {}
    for k,v in panels.items():
        x,y = mesh2d(v, points)
        theta,phi = scatter.transform_spherical(x, y, l, xpos, ypos)
        readings[k] = scatter.compton(theta, phi, ko, Z)
    return readings

def differential_thompson_scattering(l, ko, Z=None, points=None, xpos=None, ypos=None):
    points = points or 100
    readings = {}
    for k,v in panels.items():
        x,y = mesh2d(v, points)
        theta,phi = scatter.transform_spherical(x, y, l, xpos, ypos)
        readings[k] = scatter.thompson(theta, phi, ko, Z)
    return readings
