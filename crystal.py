from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from io import StringIO
import symop, re
import numpy as np
import pandas as pd


def cellvol(a, b, c, alpha, beta, gamma):
    """
    Compute the volume of a crystallographic unit cell from the lattice constants.

    Parameters
    ----------
    a : float
        Length of the unit cell A axis in angstroms
    b : float
        Length of the unit cell B axis in angstroms
    c : float
        Length of the unit cell C axis in angstroms
    alpha : float
        Unit cell alpha angle in degrees
    beta  : float
        Unit cell beta angle in degrees
    gamma : float
        Unit cell gamma angle in degrees

    Returns
    -------
    float
        The volume of the unit cell in cubic angstroms
    """
    alpha = np.deg2rad(alpha)
    beta  = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    V = a*b*c*(1. - np.cos(alpha)*np.cos(alpha) - np.cos(beta)*np.cos(beta) - np.cos(gamma)*np.cos(gamma) + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))**(0.5)
    return V

def orthogonalization(a, b, c, alpha, beta, gamma):
    """
    Compute the orthogonalization matrix from the unit cell constants

    Parameters
    ----------
    a : float
        Length of the unit cell A axis in angstroms
    b : float
        Length of the unit cell B axis in angstroms
    c : float
        Length of the unit cell C axis in angstroms
    alpha : float
        Unit cell alpha angle in degrees
    beta  : float
        Unit cell beta angle in degrees
    gamma : float
        Unit cell gamma angle in degrees

    Returns
    -------
    np.ndarry
        A 3x3 array of the orthogonalization matrix corresponding to the supplied cell parameters
    """
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
    """
    Compute the deorthogonalization matrix from the unit cell constants

    Parameters
    ----------
    a : float
        Length of the unit cell A axis in angstroms
    b : float
        Length of the unit cell B axis in angstroms
    c : float
        Length of the unit cell C axis in angstroms
    alpha : float
        Unit cell alpha angle in degrees
    beta  : float
        Unit cell beta angle in degrees
    gamma : float
        Unit cell gamma angle in degrees

    Returns
    -------
    np.ndarry
        A 3x3 array of the deorthogonalization matrix corresponding to the supplied cell parameters
    """
    O = orthogonalization(a, b, c, alpha, beta, gamma)
    return np.linalg.inv(O)

def dhkl(h, k, l, a, b, c, alpha, beta, gamma):
    """
    Compute the real space lattice plane spacing, "d", associated with a given hkl.

    Parameters
    ----------
    h : int or np.ndarray
        h-index or indices for which you wish to calculate lattice spacing
    k : int or np.ndarray
        k-index or indices for which you wish to calculate lattice spacing
    l : int or np.ndarray
        l-index or indices for which you wish to calculate lattice spacing
    a : float
        Length of the unit cell A axis in angstroms
    b : float
        Length of the unit cell B axis in angstroms
    c : float
        Length of the unit cell C axis in angstroms
    alpha : float
        Unit cell alpha angle in degrees
    beta  : float
        Unit cell beta angle in degrees
    gamma : float
        Unit cell gamma angle in degrees

    Returns
    -------
    float or array_like
    """
    hkl = np.vstack((h, k, l))
    Oinv = deorthogonalization(a, b, c, alpha, beta, gamma)
    d = 1./np.sqrt(np.sum(np.square(np.matmul(Oinv.T, hkl)), 0))
    return d

def lattice_constants(inFN):
    """
    Parse a CNS file and return the crystal lattice constants

    Parameters
    ----------
    inFN : string
        Name of CNS file you wish to parse

    Returns
    -------
    np.ndarray
        A numpy array with the lattice constants [a, b, c, alpha, beta, gamma]
    """
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

class hklfile(pd.DataFrame):
    def __init__(self, hklFN=None):
        pd.DataFrame.__init__(self)
        self.header=None
        self.dataname = None #Store name like FOBS or IOBS
        self.filename = hklFN
        self._parse()

    def _parse(self):
        if self.filename is not None:
            self.header  = [i for i in open(self.filename) if i[:4] != 'INDE']
            declare = [i for i in self.header if i[:4] == 'DECL'][0]
            lines   = [i for i in open(self.filename) if i[:4] == 'INDE']

            colnames = ['H', 'K', 'L']
            colnames.append(re.search(r"(?<=NAME=)[^\s]*", declare).group())
            self.dataname = colnames[-1]

            usecols  = [1, 2, 3, 5]
            #Determine if there is phase information in the file
            if len(lines[0].split()) == 7:
                colnames.append('PHASE')
                usecols.append(6)

            f = StringIO(''.join(lines))
            F = pd.read_csv(f, 
                delim_whitespace=True, 
                names=colnames,
                usecols=usecols,
            )
            a,b,c,alpha,beta,gamma = lattice_constants(self.filename)
            F['D'] = dhkl(F['H'], F['K'], F['L'], a, b, c, alpha, beta, gamma)
            F = F.set_index(['H', 'K', 'L'])
            pd.DataFrame.__init__(self, F)

    def write(self, outfile):
        """
        Parameters
        ----------
        outfile : file or string
            Output filename or file object
        """
        if isinstance(outfile, str):
            outfile = open(outfile, 'w')
        outfile.write(''.join(self.header))
        for (h,k,l),F in self.iterrows():
            line = "INDE {} {} {} {}= {:0.4f}".format(h, k, l, self.dataname, F[self.dataname])
            if 'PHASE' in F:
                line += " {}\n".format(F['PHASE'])
            else:
                line += "\n"
            outfile.write(line)
        outfile.close()

def parsehkl(inFN):
    """
    Parse a CNS file and return structure factors

    Parameters
    ----------
    inFN : string
        Name of a CNS file you wish to parse

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing reflection info 
    """
    lines = [i for i in open(inFN) if i[:4] == 'INDE']

    colnames = ['H', 'K', 'L', 'F']
    usecols  = [1, 2, 3, 5]
    #Determine if there is phase information in the file
    if len(lines[0].split()) == 7:
        colnames.append('PHASE')
        usecols.append(6)

    f = StringIO(''.join(lines))
    F = pd.read_csv(f, 
        delim_whitespace=True, 
        names=['H','K','L','F'],
        usecols=[1,2,3,5], 
    )
    a,b,c,alpha,beta,gamma = lattice_constants(inFN)
    F['D'] = dhkl(F['H'], F['K'], F['L'], a, b, c, alpha, beta, gamma)
    F = F.set_index(['H', 'K', 'L'])
    return F

def spacegroup(hklFN):
    """
    Parse a CNS file and return the space group number

    Parameters
    ----------
    inFN : string
        Name of a CNS file you wish to parse

    Returns
    -------
    int
        The space group number
    """
    line = open(hklFN).readline()
    spacegroupname = line.split()[1].split('=')[1]
    spacegroupname = ''.join(spacegroupname.split('('))
    spacegroupname = ' '.join(spacegroupname.split(')')).strip()
    spacegroupname = spacegroupname[0] + ' ' + spacegroupname[1:]
    return symop.spacegroupnums[spacegroupname]

class crystal():
    """
    Representation of a crystal

    Attributes
    ----------
    spacegroup : int
        Number corresponding to the crystal space group
    cell : np.ndarray
        Unit cell constants of crystal (A, B, C, alpha, beta, gamma)
    A : np.ndarray
        Matrix of unit cell vectors
    F : pd.DataFrame
        Dataframe containing the structure factors for the crystal
    """
    def __init__(self, hklFN=None):
        """
        Parameters
        ----------
        hklFN : str
            CNS input filename
        """
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
        """
        Rotate the crystal unit cell by phistep about an axis. Update the A matrix of the crystal accordingly. 

        Parameters
        ----------
        phistep : float
            The number of degrees to rotate the crystal
        axis : np.ndarray
            The cartesian axis on which to rotate the crystal

        Returns
        -------
        crystal
            This method returns self for easy chaining. 
        """
        phistep = np.deg2rad(phistep)
        if axis is None:
            axis = np.array([0., 1, 0.])
        #Formula for arbitrary rotation about an axis
        R = np.identity(3)*np.cos(phistep) + np.sin(phistep)*np.cross(axis, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T + (1 - np.cos(phistep))*np.outer(axis, axis)
        self.A = np.matmul(self.A, R)
        return self

    def reflections(self, wavelength=None, tol=None, detector_distance=None):
        """
        Parameters
        ----------
        wavelength : float
            The wavelength of the x-ray beam in Angstroms
        tol : float, optional
            Allowed error in the Bragg condition to accept a reflection. The defalt value is 0.001.
        detector_distance : float
            The distance of the simulated "detector" from the crystal position. The default value is 100 mm.

        Returns
        -------
        pd.Dataframe
            Dataframe object with reflections, structure factors. 
        """
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
        """
        Randomly rotate the unit cell and update the A-matrix correspondingly. 

        Returns
        -------
        crystal
            This method resturns self for easy chaining. 
        """
        self.rotate(360.*np.random.random(), axis=[1., 0., 0.])
        self.rotate(360.*np.random.random(), axis=[0., 1., 0.])
        self.rotate(360.*np.random.random(), axis=[0., 0., 1.])
        return self

    def phiseries(self, phistep, nsteps, reflections_kwargs=None, axis=None, nprocs=None):
        """
        Compute a series of images by rotating the crystal. This method uses multiprocessing for parallelization. 

        Parameters
        ----------
        phistep : float
            Phi angle step in degrees between frames
        nsteps : int
            Number of images to simulate
        reflections_kwargs : dict
            Keword arguments to pass to crystal.reflections in case you want to override the defaults. Default is None.
        axis : np.ndarray
            Axis about which to rotate the crystal. 
        nprocs : int
            Number of processors to use for this calculation. 

        Returns
        -------
        pd.DataFrame
            Datframe containing the accepted reflections from the rotation series. 
        """
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

