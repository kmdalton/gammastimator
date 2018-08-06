import argparse
from scipy.special import erf
import ipm
import numpy as np
import pandas as pd

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
    "--intensityscale"         : 2.,
    "--intensityshape"           : 0.2, 
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

def parsehkl(inFN):
    F = pd.read_csv(inFN, 
        delim_whitespace=True, 
        names=['H','K','L','F'],
        usecols=[1,2,3,5], 
        skiprows=4,
        index_col=['H','K','L']
    )
    return F


def build_model(offFN, onFN, **kw):
    Fon,Foff = parsehkl(onFN),parsehkl(offFN)
    Foff['gamma'] = np.square(Fon['F']/Foff['F'])
    model = Foff.sample(frac=kw.get('multiplicity', 10.), replace=True)
    model['Foff'] = model['F']
    model['Fon'] = Fon.loc[model.index]
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
    d[:,8] = np.random.gamma(kw.get('intensityshape', 0.2), kw.get('intensityscale', 2.), n)
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
