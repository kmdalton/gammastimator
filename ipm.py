import scatter
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.integrate import nquad


#Panel dimensions in um (xmin, xmax, ymin, ymax) 
panels = {
    'T': (-5000., 5000., 5000., 15000.),
    'R': (5000., 15000., -5000., 5000.),
    'B': (-5000., 5000., -15000., -5000.),
    'L': (-15000., -5000., -5000., 5000.),
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
    points = 100 if points is None else points
    readings = {}
    for k,bounds in panels.items():
        x,y = mesh2d(bounds, points)
        x = x if xpos is None else x-xpos
        y = y if ypos is None else y-ypos
        theta,phi = scatter.transform_spherical(x, y, l)
        readings[k] = scatter.differential_intensity(theta, phi, ko, Z)
    return readings

def differential_compton_scattering(ko, l=None, Z=None, points=None, xpos=None, ypos=None):
    l = film_distance if l is None else l
    points = 100 if points is None else points
    readings = {}
    for k,bounds in panels.items():
        x,y = mesh2d(bounds, points)
        x = x if xpos is None else x-xpos
        y = y if ypos is None else y-ypos
        theta,phi = scatter.transform_spherical(x, y, l)
        readings[k] = scatter.compton(theta, phi, ko, Z)
    return readings

def differential_thompson_scattering(ko, l=None, Z=None, points=None, xpos=None, ypos=None):
    l = film_distance if l is None else l
    points = 100 if points is None else points
    readings = {}
    for k,bounds in panels.items():
        x,y = mesh2d(bounds, points)
        x = x if xpos is None else x-xpos
        y = y if ypos is None else y-ypos
        theta,phi = scatter.transform_spherical(x, y, l)
        readings[k] = scatter.thompson(theta, phi, ko, Z)
    return readings

def ipm_readings(ko, xpos, ypos, l=None, points=None, keys=None):
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
    bounds = {k: (panels[k][0] - xpos ,
                  panels[k][1] - xpos ,
                  panels[k][2] - ypos ,
                  panels[k][3] - ypos
              ) for k in keys}
    si = {k:integrate_panel(ko, bounds[k], l, 14, points) for k in keys}
    n  = {k:integrate_panel(ko, bounds[k], l, 7, points)  for k in keys}
    t  = {k:3.*si[k]/7. + 4.*n[k]/7. for k in keys}
    return np.array([t[k] for k in keys])

def transform_cartesian(theta, phi, l=None):
    l = film_distance if l is None else l
    x = -np.sign(np.cos(phi))*l*np.tan(theta)/np.sqrt(1+np.tan(phi)**2)
    y = x*np.tan(phi)
    return x,y

def integrate_panel(ko, bounds, l=None, Z=None, points=None):
    border = 0.1 #add a border around the area to integrate
    l = film_distance if l is None else l
    points = 100 if points is None else points
    x,y = mesh2d(bounds, points)

    theta, phi = scatter.transform_spherical(x, y, l)
    #Setting the integration ranges
    thetarange = theta.max() - theta.min()
    theta = np.linspace(0.-thetarange*border, thetarange*(1.+border), points) + theta.min()

    #Subsample points compute integration parameters
    #dtheta,dphi = (thetamax - thetamin)/float(points), 2*np.pi/float(points)
    #theta = np.linspace(0., thetamax-thetamin, points)+thetamin
    phi = np.linspace(-np.pi, np.pi, points)
    theta,phi = np.meshgrid(theta, phi)
    dtheta,dphi = np.pi/float(points), 2*np.pi/float(points)
    #print(bounds,theta,phi)

    def indicator(theta, phi):
        x,y = transform_cartesian(theta, phi)
        return (x >= bounds[0]) & (x <= bounds[1]) & (y >= bounds[2]) & (y <= bounds[3])
        #indicator = (x >= bounds[0]) & (x <= bounds[1]) & (y >= bounds[2]) & (y <= bounds[3])
        #d = scatter.differential_intensity(theta, phi, ko, Z)
        #return indicator#*d
    return np.sum(scatter.differential_intensity(theta,phi,ko,Z)*indicator(theta,phi)*np.sin(theta)*dtheta*dphi)
    
