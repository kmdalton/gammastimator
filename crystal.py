from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import symop
import numpy as np
import pandas as pd


def cellvol(a, b, c, alpha, beta, gamma):
    alpha = np.deg2rad(alpha)
    beta  = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    V = a*b*c*(1. - np.cos(alpha)*np.cos(alpha) - np.cos(beta)*np.cos(beta) - np.cos(gamma)*np.cos(gamma) + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))**(0.5)
    return V

def orthogonalization(a, b, c, alpha, beta, gamma):
    V = cellvol(a, b, c, alpha, beta, gamma)
    alpha = np.deg2rad(alpha)
    beta  = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    O = np.array([
        [a, b*np.cos(gamma), c*np.cos(beta)],
        [0, b*np.sin(gamma), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)],
        [0., 0., V/(np.sin(gamma)*a*b)]
    ])
    return O

def deorthogonalization(a, b, c, alpha, beta, gamma):
    O = orthogonalization(a, b, c, alpha, beta, gamma)
    return np.linalg.inv(O)

def dhkl(h, k, l, a, b, c, alpha, beta, gamma):
    hkl = np.vstack((h, k, l))
    Oinv = deorthogonalization(a, b, c, alpha, beta, gamma)
    d = 1./np.sqrt(np.sum(np.square(np.matmul(Oinv.T, hkl)), 0))
    return d

def lattice_constants(inFN):
    a = b = c = alpha = beta = gamma = None
    with open(inFN) as f:
        header = f.readline()
        a = float(header.split("a=")[1].split()[0])
        b = float(header.split("b=")[1].split()[0])
        c = float(header.split("c=")[1].split()[0])
        alpha = float(header.split("alpha=")[1].split()[0])
        beta  = float(header.split("beta=")[1].split()[0])
        gamma = float(header.split("gamma=")[1].split()[0])
    return np.array([a,b,c,alpha,beta,gamma])

def parsehkl(inFN):
    F = pd.read_csv(inFN, 
        delim_whitespace=True, 
        names=['H','K','L','F'],
        usecols=[1,2,3,5], 
        skiprows=4,
    )
    a,b,c,alpha,beta,gamma = lattice_constants(inFN)
    F['D'] = dhkl(F['H'], F['K'], F['L'], a, b, c, alpha, beta, gamma)
    F = F.set_index(['H', 'K', 'L'])
    return F

def spacegroup(hklFN):
    line = open(hklFN).readline()
    spacegroupname = line.split()[1].split('=')[1]
    spacegroupname = ''.join(spacegroupname.split('('))
    spacegroupname = ' '.join(spacegroupname.split(')')).strip()
    spacegroupname = spacegroupname[0] + ' ' + spacegroupname[1:]
    return symop.spacegroupnums[spacegroupname]

class crystal():
    def __init__(self, hklFN=None):
        self.spacegroup = None
        self.cell = None
        self.A = None
        self.F = None
        if hklFN is not None:
            self._parse(hklFN)

    def _parse(self, hklFN):
        self.spacegroup = spacegroup(hklFN)
        self.cell = lattice_constants(hklFN)
        #By default, A is initialized to +x
        self.A = orthogonalization(*self.cell).T
        F = parsehkl(hklFN).reset_index()
        F['MERGEDH'] = F['H']
        F['MERGEDK'] = F['K']
        F['MERGEDL'] = F['L']
        self.F = None
        for k,op in symop.symops[self.spacegroup].items():
            f = F.copy()
            f[['H', 'K', 'L']] = np.array(op(f[['H', 'K', 'L']].T).T, int)
            self.F = pd.concat((self.F, f))
        Friedel = self.F.copy()
        Friedel[['H', 'K', 'L']] = -Friedel[['H', 'K', 'L']]
        self.F = pd.concat((self.F, Friedel)).set_index(['H', 'K', 'L'])
        self.F = self.F[~self.F.index.duplicated(keep='first')]

    def copy(self):
        x = crystal()
        x.cell = self.cell.copy()
        x.spacegroup = self.spacegroup
        x.A = self.A.copy()
        x.F = self.F.copy()
        return x

    def rotate(self, phistep, axis=None):
        phistep = np.deg2rad(phistep)
        if axis is None:
            axis = np.array([0., 1, 0.])
        #Formula for arbitrary rotation about an axis
        R = np.identity(3)*np.cos(phistep) + np.sin(phistep)*np.cross(axis, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T + (1 - np.cos(phistep))*np.outer(axis, axis)
        self.A = np.matmul(self.A, R)
        return self

    def reflections(self, wavelength=None, tol=None, detector_distance=None):
        detector_distance= 100. if detector_distance is None else detector_distance
        wavelength = 1. if wavelength is None else wavelength
        tol = 0.001 if tol is None else tol
        Ainv = np.linalg.inv(self.A)
        So = np.array([0, 0, 1./wavelength])
        def err(x):
            h = np.array(x.name)
            S = np.matmul(Ainv, h)
            return 0.5*np.dot(S, S) - np.dot(S, So)
        F = self.F[np.abs(self.F.apply(err, 1)) <= tol]
        def coordinate(x):
            h = np.array(x.name)
            S = np.matmul(Ainv, h)
            S1 = S+So
            S1 = S1/(wavelength*np.linalg.norm(S1)) #Map to Ewald Sphere
            XYZ = detector_distance*S1/S1[2]
            return pd.Series(XYZ)
        F['A'] = [self.A[:,0]]*len(F)
        F['B'] = [self.A[:,1]]*len(F)
        F['C'] = [self.A[:,2]]*len(F)
        return F.join(F.apply(coordinate, 1).rename(columns={0:'X', 1:'Y', 2:'Z'}))

    def orientrandom(self):
        self.rotate(360.*np.random.random(), axis=[1., 0., 0.])
        self.rotate(360.*np.random.random(), axis=[0., 1., 0.])
        self.rotate(360.*np.random.random(), axis=[0., 0., 1.])
        return self

    def phiseries(self, phistep, nsteps, reflections_kwargs=None, axis=None, nprocs=None):
        axis = [0,1,0] if axis is None else axis
        reflections_kwargs = {} if reflections_kwargs is None else reflections_kwargs
        iterable = [(self.copy().rotate(i*phistep, axis=axis), reflections_kwargs, i) for i in range(nsteps)]
        #iterable = [i*phistep for i in range(nsteps)]
        nprocs = cpu_count() if nprocs is None else nprocs
        p = Pool(nprocs)
        F = p.map(_phihelper, iterable)
        p.terminate()
        return pd.concat(F)

def _phihelper(X):
    cryst, kw, i = X
    f = cryst.reflections(**kw)
    f['PHINUMBER'] = i
    return f

