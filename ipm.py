
import scatter
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


#Panel dimensions in um (xmin, xmax, ymin, ymax) 
panels = {
    'T': (-5000., 5000., 5000., 15000.),
    'R': (5000., 15000., -5000., 5000.),
    'B': (-5000., 5000., -5000., -15000.),
    'L': (-5000., -15000., -5000., 5000.),
}

#Distance from the diode array center to the Si3N4 film in um
film_distance = -10000.


def mesh2d(bounds, points):
    return np.meshgrid(np.linspace(bounds[0], bounds[1], points), np.linspace(bounds[2], bounds[3], points))
    


def plot_detector(readings, cmap=None, xpos=None, ypos=None, ax=None, norm=None):
    """
    Parameters
    ----------
    readings : dict
        dictionary containing the names of panels and their respective readings
    """
    ax = ax if ax is not None else plt.gca()
    cmap = cmap if cmap is not None else plt.get_cmap('viridis')
    X,Y,Z = [],[],[]
    for k,v in readings.items():
        x = np.linspace(panels[k][0], panels[k][1], v.shape[0])
        y = np.linspace(panels[k][2], panels[k][3], v.shape[1])
        x,y = np.meshgrid(x,y)
        X = np.concatenate((X, x.flatten()))
        Y = np.concatenate((Y, y.flatten()))
        Z = np.concatenate((Z, v.flatten()))
    Z = Z*1e28 #Barns
    norm = norm if norm is not None else mpl.colors.Normalize(Z.min(), Z.max())
    ax.hexbin(X, Y, Z, cmap=cmap, norm=norm)
    ax.set_facecolor(cmap(0.))
    for k,v in panels.items():
        ax.plot(v[:2], [v[2], v[2]], lw=3, c='w')
        ax.plot(v[:2], [v[3], v[3]], lw=3, c='w')
        ax.plot([v[0], v[0]], v[2:], lw=3, c='w')
        ax.plot([v[1], v[1]], v[2:], lw=3, c='w')
    sm = mpl.cm.ScalarMappable(norm, cmap)
    sm.set_array(Z)
    plt.colorbar(sm, ax=ax, label=r"$\frac{d\sigma }{d\Omega }\ (barn\ sr^{-1}\ atom^{-1})$")
    if xpos is not None:
        ax.scatter(xpos, ypos, edgecolors='w', facecolors='none')
    ax.set_xlabel(r'$\mu m$')
    ax.set_ylabel(r'$\mu m$')
    return ax

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
    xpos,ypos = xpos,ypos
    l = film_distance if l is None else l
    keys = keys or ['T', 'R', 'B', 'L']
    si = {k:v.sum() for k,v in differential_scattering(ko, l, 14, points, xpos, ypos).items()}
    n  = {k:v.sum() for k,v in differential_scattering(ko, l, 7 , points, xpos, ypos).items()}
    t  = {k:3.*si[k]/7. + 4.*n[k]/7. for k in si}
    return np.array([t[k] for k in keys])
