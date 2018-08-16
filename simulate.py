import argparse
import symop
from scatter import ev2angstrom,angstrom2ev
from scipy.special import erf
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import ipm
import numpy as np
import pandas as pd

p = Pool(cpu_count())

argDict = {
    "Fon"                      : "CNS file containing pumped structure factor amplitudes.",
    "Foff"                     : "CNS file containing un-pumped structure factor amplitudes.",
    "out"                      : "CSV intensity file to output.", 
    "--multiplicity"           : "The multiplicity to which the dataset is sampled. Internally simulate.py makes draws from the input data with replacement to build up the dataset of reflections. Multiplicity says for the program to make multiplicity * the number of reflections in the input data.",
    "--missing"                : "The fraction of missing reflections in the dataset. A given reflection cannot always be integrated across all images at a given phi angle. This feature is meant to emulate the simple reality that reflections are often missing from some images. This will simply subsample the data before output. Users may wish to do this themselves in post. Subsampling will speed up data generation. If you plan to subsample for testing anyway, you may want to use this option.",
    "--intensityscale"         : "Scale parameter for the gamma distribution of IPM intensities",
    "--intensityshape"         : "Shape parameter for the gamma distribution of intensities",
    "--reflectionsperimage"    : "Average number of reflections per image (default 150)",
    "--reflectionsperimagestd" : "Std deviation of reflectiosn per image (default 50)",
    "--minreflectionsperimage" : "Minimum reflections to generate per image",
    "--minimages"              : "Minimum images per run/crystal",
    "--meanimages"            : "Mean images per run/crystal",
    "--stdimages"              : "Standard deviation images per run/crystal",
    "--onreps"                 : "Number of on images per phi angle.",
    "--offreps"                : "Number of off images per phi angle.",
    "--sigintercept"           : "Sigma(I) = m*I + b. b",
    "--sigslope"               : "Sigma(I) = m*I + b. m",
    "--partialitymean"         : "Average partiality of reflections",
    "--partialitystd"          : "Average partiality of reflections",
    "--partialitymin"          : "Minimum partiality of reflections",
    "--energy"                 : "X-Ray energy in electron volts",

    #IPM parameters
    "--ipmslope"               : "Relation between IPM reading and photon flux", 
    "--ipmintercept"           : "Relation between IPM reading and photon flux", 

    #Beam geometry parameters
    "--sigx"                   : "Standard deviation in x pos in 'microns'",
    "--sigy"                   : "Standard deviation in y pos in 'microns'",
    "--divx"                   : "Beam size in x in 'microns' (standard deviation of bivariate gaussian)",
    "--divy"                   : "Beam size in y in 'microns' (standard deviation of bivariate gaussian)",

    #Crystal dimensions and alignment parameters
    "--sigheight"              : "Standard deviation of crystal dimension in microns",
    "--sigwidth"               : "Standard deviation of crystal dimension in microns",
    "--height"                 : "Mean of crystal dimension in microns",
    "--width"                  : "Mean of crystal dimension in microns",
    "--sigalign"               : "Standard deviation of crystal alignment dimension in microns",
}

datatypes = {
    "--multiplicity"           : float ,
    "--missing"                : float , 
    "--intensityscale"         : float ,
    "--intensityshape"           : float , 
    "--reflectionsperimage"    : int,
    "--reflectionsperimagestd" : int,
    "--minreflectionsperimage" : int,
    "--minimages"              : int,
    "--meanimages"            : int,
    "--stdimages"              : int, 
    "--onreps"                 : int,
    "--offreps"                : int,
    "--sigintercept"           : float,
    "--sigslope"               : float, 
    "--partialitymean"         : float,
    "--partialitystd"          : float,
    "--partialitymin"          : float,
    "--energy"                 : float,

    #IPM parameters
    "--ipmslope"               : float, 
    "--ipmintercept"           : float, 

    #Beam geometry parameters
    "--sigx"                   : float,
    "--sigy"                   : float,
    "--divx"                   : float,
    "--divy"                   : float,

    #Crystal dimensions and alignment parameters
    "--sigheight"              : float,
    "--sigwidth"               : float,
    "--height"                 : float,
    "--width"                  : float,
    "--sigalign"               : float,
}

defaults = {
    "--multiplicity"           : 10.,
    "--missing"                : 0.0, 
    "--intensityscale"         : 1.0,
    "--intensityshape"           : 2.0, 
    "--reflectionsperimage"    : 150,
    "--reflectionsperimagestd" : 50,
    "--minreflectionsperimage" : 50,
    "--minimages"              : 10,
    "--meanimages"            : 10,
    "--stdimages"              : 10, 
    "--onreps"                 : 4, 
    "--offreps"                : 4,
    "--sigintercept"           : 5.,
    "--sigslope"               : 0.03, 
    "--partialitymean"         : 0.6,
    "--partialitystd"          : 0.2, 
    "--partialitymin"          : 0.1, 
    "--energy"                 : 12398., 

    #IPM parameters
    "--ipmslope"               : 1., 
    "--ipmintercept"           : 0., 

    #Beam geometry parameters
    "--sigx"                   : 10.,
    "--sigy"                   : 5.,
    "--divx"                   : 100.,
    "--divy"                   : 50.,

    #Crystal dimensions and alignment parameters
    "--sigheight"              : 10.,
    "--sigwidth"               : 10.,
    "--height"                 : 100.,
    "--width"                  : 100.,
    "--sigalign"               : 10.,
}


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
    return a,b,c,alpha,beta,gamma

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
    def __init__(self, hklFN):
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
        tol = 0.005 if tol is None else tol
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

    def phiseries(self, phistep, nsteps, reflections_kwargs=None, axis=None):
        axis = [0,1,0] if axis is None else axis
        reflections_kwargs = {} if reflections_kwargs is None else reflections_kwargs
        F = None
        for i in range(1, nsteps+1):
            self.rotate(phistep, axis)
            f = self.reflections(**reflections_kwargs)
            f['PHINUMBER'] = i
            F = pd.concat((F, f))
        return F

def better_model(offFN, onFN, **kw):
    #Crystal orientation
    sigalign = kw.get("sigalign", 10.)
    energy = kw.get('energy', 12398.)
    wavelength = ev2angstrom(energy)
    #Crystal dimensions 
    height = kw.get("height", 50.)
    width  = kw.get("width", 100.)
    sigh   = kw.get("sigheight", 10.)
    sigw   = kw.get("sigwidth", 10.)

    ewald_tol  = kw.get('ewald_tol', .005)
    detector_distance  = kw.get('detector_distance', 100.0)
    phistep = kw.get('phistep', 0.25)
    nruns = kw.get('runs', 10)
    runlength = kw.get('runlength', 30)
    reflections_kwargs = {
        'wavelength' : wavelength , 
        'tol'        : ewald_tol, 
        'detector_distance' : detector_distance,
    }

    Fon = parsehkl(onFN)
    X = crystal(offFN)
    unitcellvolume = cellvol(*X.cell)
    model = None
    for i in range(nruns):
        #randomize the phi angle
        X.rotate(360.*np.random.random())
        run = X.phiseries(phistep, runlength)
        run['RUN'] = i+1
        x = np.random.normal(0., sigalign)
        y = np.random.normal(0., sigalign)
        h = np.random.normal(height, sigh)
        w = np.random.normal(width , sigw)
        run['CRYSTBOTTOM'] = y - h/2.
        run['CRYSTTOP']    = y + h/2.
        run['CRYSTLEFT']   = x - w/2.
        run['CRYSTRIGHT']  = x + w/2.
        run['CRYSTVOL'] = np.pi*0.25*w*w*h*(1e12) #The crystal is just modeled as a cylinder to make things easy
        run['CELLVOL'] = unitcellvolume
        model = pd.concat((model, run))
    model['Fon'] = Fon.loc[model.index]['F']
    model.rename({"F": "Foff"}, axis=1, inplace=True)
    model['gamma'] = (model['Fon']/model['Foff'])**2


    model['SERIES'] = 'off1'
    m = model.copy()
    for i in range(kw.get('offreps', 4)-1):
        n = m.copy()
        n['SERIES'] = 'off{}'.format(i+2)
        model = pd.concat((model, n))
 
    for i in range(kw.get('onreps', 4)):
        n = m.copy()
        n['SERIES'] = 'on{}'.format(i+1)
        model = pd.concat((model, n))

    model = populate_ipm_data(model, **kw)
    model['Io'] = model['IPM']*kw.get('ipmslope', 1.) + kw.get('ipmintercept', 0.)
#    model['Icryst'] = 0.25*model['Io']*(
#            erf((model['CRYSTRIGHT']  - model['BEAMX'])/divx) - 
#            erf((model['CRYSTLEFT']   - model['BEAMX'])/divx) 
#            ) * (
#            erf((model['CRYSTTOP']    - model['BEAMY'])/divy) - 
#            erf((model['CRYSTBOTTOM'] - model['BEAMY'])/divy)
#        )
#
    #model['I'] = (wavelength**3*model['CRYSTVOL']/np.square(model['CELLVOL']))*model.P*(model.Fon**2*model.SERIES.str.contains('on') + model.Foff**2*model.SERIES.str.contains('off'))
    #model['SIGMA(IOBS)'] = kw.get("sigintercept", 5.0) + kw.get("sigslope", 0.03)*model['I']
    #model['IOBS']  = np.random.normal(model['I'], model['SIGMA(IOBS)'])
    return model

def ipmhelper(args):
    return ipm.ipm_readings(*args)

def populate_ipm_data(model, **kw):
    g = model.groupby(['RUN', 'PHINUMBER', 'SERIES'])
    model = model.set_index(['RUN', 'PHINUMBER', 'SERIES'])
    n = len(g)

    #Things we need to populate: IPM, Icryst, BEAMX, BEAMY, IPM_0, IPM_1, IPM_2, IPM_3, IPM_X, IPM_Y
    #Note that IPM == sum(IPM_0,1,2,3)
    sigx,sigy = kw.get('sigx', 10.),kw.get('sigy', 5.)
    divx,divy = kw.get('divx', 100.),kw.get('divy', 50.)
    divx,divy = np.sqrt(2)*divx,np.sqrt(2)*divy

    d = np.zeros((n, 9)) 
    d[:,0],d[:,1] = np.random.normal(0., sigx, n), np.random.normal(0., sigy, n)
    #d[:,2:6] = np.vstack([ipm.ipm_readings(kw.get('energy', 12398.), i, j, points=kw.get('points', 500)) for i,j in zip(d[:,0], d[:,1])])
    e = kw.get('energy', 12398.) * np.ones(n)
    points=kw.get('points', 500) * np.ones(n)
    z = ipm.film_distance * np.ones(n)
    d[:,2:6] = p.map(ipmhelper, zip(e*np.ones(n), d[:,0], d[:,1], z, points*np.ones(n)))
    #d[:,2:6] = list(map(ipmhelper, zip(e*np.ones(n), d[:,0], d[:,1], z, points*np.ones(n))))

    d[:,8] = np.random.gamma(kw.get('intensityshape', 2.0), kw.get('intensityscale', 1.), n)
    d[:,6] = (d[:,3] - d[:,5]) / (d[:,3] + d[:,5])
    d[:,7] = (d[:,2] - d[:,4]) / (d[:,2] + d[:,4])
    d[:,2:6] = d[:,-1,None]*d[:,2:6]/d[:,2:6].sum(1)[:,None]
    d = np.hstack((d, np.arange(n)[:, None] + 1))
    for key in ['BEAMX', 'BEAMY', 'IPM_0', 'IPM_1', 'IPM_2', 'IPM_3', 'IPM_X', 'IPM_Y', 'IPM', 'IMAGENUMBER']:
        model[key] = 0.

    for i,idx in enumerate(g.groups):
        model.loc[idx, ['BEAMX', 'BEAMY', 'IPM_0', 'IPM_1', 'IPM_2', 'IPM_3', 'IPM_X', 'IPM_Y', 'IPM', 'IMAGENUMBER']] = d[i]
    model = model.reset_index()
    return model


def build_model(offFN, onFN, **kw):
    Fon,Foff = parsehkl(onFN),parsehkl(offFN)
    Foff['gamma'] = np.square(Fon['F']/Foff['F'])
    model = Foff.sample(frac=kw.get('multiplicity', 10.), replace=True)
    model['Foff'] = model['F']
    model['Fon'] = Fon.loc[model.index]['F']
    del model['F']
    model.reset_index(inplace=True)

    #you could do this with np.repeat in a couple of lines but it is unreadable
    minref  = kw.get('minreflections', 50)
    meanref = kw.get('reflectionsperimage', 150)
    stdref  = kw.get('reflectionsperimagestd', 50)
    nrefs = int(np.random.normal(meanref, stdref))
    nrefs = max(nrefs, minref)
    imagenumber = [0]*nrefs

    while len(imagenumber) < len(model):
        nrefs = int(np.random.normal(meanref, stdref))
        nrefs = max(nrefs, minref)
        imagenumber += [imagenumber[-1] + 1]*nrefs
        #Stop criterion
        if len(imagenumber) > len(model):
            imagenumber = imagenumber[:len(model)]
        
    model["PHINUMBER"] = imagenumber

    meanimages = kw.get('meanimages', 50)
    stdimages = kw.get('stdimages', 20)
    minimages = kw.get('minimages', 10)
    runs = []
    images = list(model['PHINUMBER'].unique())
    while len(images) > 0:
        ct = max(minimages, int(np.random.normal(meanimages, stdimages)))
        if len(images) < ct:
            ct = len(images)
        runs.append([images.pop(0) for i in range(ct)])

    #Crystal orientation
    sigalign = kw.get("sigalign", 10.)

    #Crystal dimensions 
    height = kw.get("height", 50.)
    width  = kw.get("width", 100.)
    sigh   = kw.get("sigheight", 10.)
    sigw   = kw.get("sigwidth", 10.)

    model['RUN'] = 0
    for n,run in enumerate(runs, 1):
        x = np.random.normal(0., sigalign)
        y = np.random.normal(0., sigalign)
        h = np.random.normal(height, sigh)
        w = np.random.normal(width , sigw)
        model.loc[model['PHINUMBER'].isin(run), 'RUN'] = n
        model.loc[model['PHINUMBER'].isin(run), 'CRYSTBOTTOM'] = y - h/2.
        model.loc[model['PHINUMBER'].isin(run), 'CRYSTTOP']    = y + h/2.
        model.loc[model['PHINUMBER'].isin(run), 'CRYSTLEFT']   = x - w/2.
        model.loc[model['PHINUMBER'].isin(run), 'CRYSTRIGHT']  = x + w/2.

    partiality = np.random.normal(kw.get("partialitymean", 0.6), kw.get("partialitystd", 0.2), len(model))
    partiality = np.minimum(partiality, 1.)
    partiality = np.maximum(partiality, kw.get("partialitymin", 0.1))
    model['P'] = partiality
    I = None

    #pd.concat((model

    model['SERIES'] = 'off1'
    m = model.copy()
    for i in range(kw.get('offreps', 4)-1):
        n = m.copy()
        n['SERIES'] = 'off{}'.format(i+2)
        model = pd.concat((model, n))
 
    for i in range(kw.get('onreps', 4)):
        n = m.copy()
        n['SERIES'] = 'on{}'.format(i+1)
        model = pd.concat((model, n))


    keys = [
            'BEAMX' ,
            'BEAMY' ,
            'IPM'    ,
            'Icryst',
            'IPM_0' ,
            'IPM_1' ,
            'IPM_2' ,
            'IPM_3' ,
            'IPM_X' ,
            'IPM_Y' ,
            'IMAGENUMBER' ,
        ]


    for k in keys:
        model[k] = 0.

    #Subsample the data before the time consuming step of integrating ipm panels
    model = model.sample(frac = 1. - kw.get('missing', 0.), replace=False)

    g = model.groupby(['RUN', 'PHINUMBER', 'SERIES'])
    model = model.set_index(['RUN', 'PHINUMBER', 'SERIES'])
    n = len(g)

    #Things we need to populate: IPM, Icryst, BEAMX, BEAMY, IPM_0, IPM_1, IPM_2, IPM_3, IPM_X, IPM_Y
    #Note that IPM == sum(IPM_0,1,2,3)
    sigx,sigy = kw.get('sigx', 10.),kw.get('sigy', 5.)
    divx,divy = kw.get('divx', 100.),kw.get('divy', 50.)
    divx,divy = np.sqrt(2)*divx,np.sqrt(2)*divy

    d = np.zeros((n, 9)) 
    d[:,0],d[:,1] = np.random.normal(0., sigx, n), np.random.normal(0., sigy, n)
    d[:,2:6] = np.vstack([ipm.ipm_readings(kw.get('energy', 12398.), i, j, points=kw.get('points', 500)) for i,j in zip(d[:,0], d[:,1])])
    d[:,8] = np.random.gamma(kw.get('intensityshape', 2.0), kw.get('intensityscale', 1.), n)
    d[:,6] = (d[:,3] - d[:,5]) / (d[:,3] + d[:,5])
    d[:,7] = (d[:,2] - d[:,4]) / (d[:,2] + d[:,4])
    d[:,2:6] = d[:,-1,None]*d[:,2:6]/d[:,2:6].sum(1)[:,None]
    d = np.hstack((d, np.arange(n)[:, None] + 1))

    for i,idx in enumerate(g.groups):
        model.loc[idx, ['BEAMX', 'BEAMY', 'IPM_0', 'IPM_1', 'IPM_2', 'IPM_3', 'IPM_X', 'IPM_Y', 'IPM', 'IMAGENUMBER']] = d[i]
    model = model.reset_index()
    model['Io'] = model['IPM']*kw.get('ipmslope', 1.) + kw.get('ipmintercept', 0.)

    model['Icryst'] = 0.25*model['Io']*(
            erf((model['CRYSTRIGHT']  - model['BEAMX'])/divx) - 
            erf((model['CRYSTLEFT']   - model['BEAMX'])/divx) 
            ) * (
            erf((model['CRYSTTOP']    - model['BEAMY'])/divy) - 
            erf((model['CRYSTBOTTOM'] - model['BEAMY'])/divy)
        )

    model['I'] = model.Icryst*model.P*(model.Fon**2*model.SERIES.str.contains('on') + model.Foff**2*model.SERIES.str.contains('off'))
    model['SIGMA(IOBS)'] = kw.get("sigintercept", 5.0) + kw.get("sigslope", 0.03)*model['I']
    model['IOBS']  = np.random.normal(model['I'], model['SIGMA(IOBS)'])
    model['PHINUMBER'] = model[['RUN', 'PHINUMBER']].groupby('RUN').transform(lambda x: x - np.min(x) +1)
    model['IMAGENUMBER'] = model['IMAGENUMBER'].astype(int)
    return model



def main():
    parser = argparse.ArgumentParser()
    for k,v in argDict.items():
        parser.add_argument(k, help=v, default=defaults.get(k, None), type=datatypes.get(k, None))

    #TODO: parser.add_argument("--signal_decay", default=None, help="Exponential decay constant to model how I/sig(I) decays with resolution. Default is 0 (no decay).")

    parser = parser.parse_args()
    model  = build_model(parser.Fon, parser.Foff, **vars(parser))
    model.to_csv(parser.out)


if __name__=="__main__":
    main()
