import crystal
import argparse
import symop
from scatter import ev2angstrom,angstrom2ev
from scipy.special import erf
from multiprocessing import cpu_count
import ipm
import numpy as np
import pandas as pd


argDict = {
    "Fon"                      : "CNS file containing pumped structure factor amplitudes.",
    "Foff"                     : "CNS file containing un-pumped structure factor amplitudes.",
    "out"                      : "CSV intensity file to output.", 
    "--crystals"               : "Number of crystals to simulate. Default is 20.", 
    "--procs"                  : "Number of processors to use for calculations.",
    #"--multiplicity"           : "The multiplicity to which the dataset is sampled. Internally simulate.py makes draws from the input data with replacement to build up the dataset of reflections. Multiplicity says for the program to make multiplicity * the number of reflections in the input data.",
    "--missing"                : "The fraction of missing reflections in the dataset. A given reflection cannot always be integrated across all images at a given phi angle. This feature is meant to emulate the simple reality that reflections are often missing from some images. This will simply subsample the data before output. Users may wish to do this themselves in post. Subsampling will speed up data generation. If you plan to subsample for testing anyway, you may want to use this option.",
    "--intensityscale"         : "Scale parameter for the gamma distribution of IPM intensities",
    "--intensityshape"         : "Shape parameter for the gamma distribution of intensities",
    "--phistep"                : "Phi angle rotation in degrees between successive images. Default 0.25 degrees.",
    #"--reflectionsperimage"    : "Average number of reflections per image (default 150)",
    #"--reflectionsperimagestd" : "Std deviation of reflectiosn per image (default 50)",
    #"--minreflectionsperimage" : "Minimum reflections to generate per image",
    "--braggtol"               : "Maximum deviation from Bragg condition to accept reflection. Default 0.001 inverse angstrom squared.",
    "--minimages"              : "Minimum images per run/crystal",
    "--meanimages"             : "Mean images per run/crystal",
    "--stdimages"              : "Standard deviation images per run/crystal",
    "--onreps"                 : "Number of on images per phi angle.",
    "--offreps"                : "Number of off images per phi angle.",
    "--sigintercept"           : "Sigma(I) = m*I + b. b",
    "--sigslope"               : "Sigma(I) = m*I + b. m",
    "--partialitymean"         : "Average partiality of reflections",
    "--partialitystd"          : "Average partiality of reflections",
    "--partialitymin"          : "Minimum partiality of reflections",
    "--energy"                 : "X-Ray energy in electron volts",
    "--bfactor"                : "Overall temperature factor of the off data. Default is 0 inverse angstroms squared.", 
    "--deltab"                 : "Overall temperature factor difference between on and off shots (Bon - Boff) in inverse angstroms squared. Default is 0.", 
    "--excited"                : "Fraction of the molecules excited in the crystal. Defaults 1.", 

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
    #"--multiplicity"           : float ,
    "--crystals"               : int, 
    "--procs"                  : int,
    "--missing"                : float , 
    "--intensityscale"         : float ,
    "--intensityshape"           : float , 
#    "--reflectionsperimage"    : int,
#    "--reflectionsperimagestd" : int,
#    "--minreflectionsperimage" : int,
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
    "--bfactor"                : float,
    "--deltab"                 : float,
    "--excited"                : float,


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
    "--phistep"                : float, 
}

defaults = {
    "--crystals"               : 20, 
    "--procs"                  : cpu_count() - 1,
    "--phistep"                : 0.25, 
    #"--multiplicity"           : 10.,
    "--missing"                : 0.0, 
    "--intensityscale"         : 1.0,
    "--intensityshape"           : 2.0, 
#    "--reflectionsperimage"    : 150,
#    "--reflectionsperimagestd" : 50,
#    "--minreflectionsperimage" : 50,
    "--minimages"              : 10,
    "--meanimages"             : 50,
    "--stdimages"              : 10, 
    "--onreps"                 : 4, 
    "--offreps"                : 4,
    "--sigintercept"           : 5.,
    "--sigslope"               : 0.03, 
    "--partialitymean"         : 0.6,
    "--partialitystd"          : 0.2, 
    "--partialitymin"          : 0.1, 
    "--energy"                 : 12398., 
    "--bfactor"                : 0.,
    "--deltab"                 : 0.,
    "--excited"                : 1.,

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


def shoot_crystal(offFN, onFN, **kw):
    #Crystal orientation
    sigalign = kw.get("sigalign", 10.)
    energy = kw.get('energy', 12398.)
    wavelength = ev2angstrom(energy)
    nprocs = kw.get("procs", cpu_count()-1)
    #Crystal dimensions 
    height = kw.get("height", 50.)
    width  = kw.get("width", 100.)
    sigh   = kw.get("sigheight", 10.)
    sigw   = kw.get("sigwidth", 10.)

    ewald_tol  = kw.get('ewald_tol', .001)
    detector_distance  = kw.get('detector_distance', 100.0)
    phistep = kw.get('phistep', 0.25)
    nruns = kw.get('crystals', 20)
    runlength = kw.get('runlength', 30)
    reflections_kwargs = {
        'wavelength' : wavelength , 
        'tol'        : ewald_tol, 
        'detector_distance' : detector_distance,
    }

    #Fon = parsehkl(onFN)
    Fon = crystal.crystal().read_hkl(onFN).unmerge()
    X = crystal.crystal().read_hkl(offFN).unmerge()
    unitcellvolume = crystal.cellvol(*X.cell)
    model = None
    meanimages = kw.get('meanimages', 50)
    stdimages = kw.get('stdimages', 20)
    minimages = kw.get('minimages', 10)
    #runlength = map(int, np.maximum(minimages, np.random.normal(meanimages, stdimages, nruns)))

    #randomize the phi angle
    X.rotate(360.*np.random.random())
    runlength =  int(np.maximum(minimages, np.random.normal(meanimages, stdimages)))
    model = X.phiseries(phistep, runlength, nprocs=nprocs)
    model['RUN'] = 1
    x = np.random.normal(0., sigalign)
    y = np.random.normal(0., sigalign)
    h = np.random.normal(height, sigh)
    w = np.random.normal(width , sigw)
    model['CRYSTBOTTOM'] = y - h/2.
    model['CRYSTTOP']    = y + h/2.
    model['CRYSTLEFT']   = x - w/2.
    model['CRYSTRIGHT']  = x + w/2.
    model['CRYSTVOL'] = np.pi*0.25*w*w*h*(1e12) #The crystal is just modeled as a cylinder to make things easy
    model['CELLVOL'] = unitcellvolume

    #TODO add support for different column names besides FOBS
    model['Fon'] = Fon.loc[model.index]['FOBS']
    model.rename({"FOBS": "Foff"}, axis=1, inplace=True)
    model['gamma'] = (model['Fon']/model['Foff'])**2
    partiality = np.random.normal(kw.get("partialitymean", 0.6), kw.get("partialitystd", 0.2), len(model))
    partiality = np.minimum(partiality, 1.)
    partiality = np.maximum(partiality, kw.get("partialitymin", 0.1))
    model['P'] = partiality

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

    #Things we need to populate: IPM, Icryst, BEAMX, BEAMY, IPM_0, IPM_1, IPM_2, IPM_3, IPM_X, IPM_Y
    #Note that IPM == sum(IPM_0,1,2,3)
    divx,divy = kw.get('divx', 100.),kw.get('divy', 50.)
    divx,divy = np.sqrt(2)*divx,np.sqrt(2)*divy

    model = model.sample(frac = 1. - kw.get('missing', 0.), replace=False)
    model = populate_ipm_data(model, **kw)
    model['Io'] = model['IPM']*kw.get('ipmslope', 1.) + kw.get('ipmintercept', 0.)
    model['Icryst'] = 0.25*model['Io']*(
            erf((model['CRYSTRIGHT']  - model['BEAMX'])/divx) - 
            erf((model['CRYSTLEFT']   - model['BEAMX'])/divx) 
            ) * (
            erf((model['CRYSTTOP']    - model['BEAMY'])/divy) - 
            erf((model['CRYSTBOTTOM'] - model['BEAMY'])/divy)
        )

    #model['I'] = (wavelength**3*model['CRYSTVOL']/np.square(model['CELLVOL']))*model.P*(model.Fon**2*model.SERIES.str.contains('on') + model.Foff**2*model.SERIES.str.contains('off'))
    bfactor = kw.get("bfactor", 0.)
    deltab  = kw.get("deltab", 0.)
    excited = kw.get("excited", 1.)

    #Compute idealized intensities per reflection observation
    model['I'] = (np.exp(-(bfactor + deltab*model.SERIES.str.contains('on'))/np.square(model.D)/4.)) * (
        #Crystal dimensions and partiality:
        (wavelength**3)*model['CRYSTVOL']/np.square(model['CELLVOL'])) * model.P * ( 
        #On images:
        excited*model.Fon**2*model.SERIES.str.contains('on') + (1 - excited)*model.Foff**2*model.SERIES.str.contains('on') + \
        #Off images:
        model.Foff**2*model.SERIES.str.contains('on')\
        ) 
    model['SIGMA(IOBS)'] = kw.get("sigintercept", np.percentile(model['I'][model['I'] > 0], 20) + kw.get("sigslope", 0.03)*model['I'])
    model['IOBS']  = np.random.normal(model['I'], model['SIGMA(IOBS)'])
    model = model[[i for i in model if 'unnamed' not in i.lower()]]
    return model


def populate_ipm_data(model, **kw):
    g = model.groupby(['RUN', 'PHINUMBER', 'SERIES'])
    model = model.reset_index().set_index(['RUN', 'PHINUMBER', 'SERIES'])
    n = len(g)
    nprocs = kw.get('procs', cpu_count() -1)

    #Things we need to populate: IPM, Icryst, BEAMX, BEAMY, IPM_0, IPM_1, IPM_2, IPM_3, IPM_X, IPM_Y
    #Note that IPM == sum(IPM_0,1,2,3)
    sigx,sigy = kw.get('sigx', 10.),kw.get('sigy', 5.)
    divx,divy = kw.get('divx', 100.),kw.get('divy', 50.)
    divx,divy = np.sqrt(2)*divx,np.sqrt(2)*divy

    d = np.zeros((n, 9)) 
    d[:,0],d[:,1] = np.random.normal(0., sigx, n), np.random.normal(0., sigy, n)
    #d[:,2:6] = np.vstack([ipm.ipm_readings(kw.get('energy', 12398.), i, j, points=kw.get('points', 500)) for i,j in zip(d[:,0], d[:,1])])
    e = kw.get('energy', 12398.) 
    points=kw.get('points', 500)
    d[:,2:6] = np.vstack(ipm.parallel_ipm_readings(e, d[:,0], d[:,1], ipm.film_distance, points, nprocs=nprocs))
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


def main():
    parser = argparse.ArgumentParser()
    for k,v in argDict.items():
        parser.add_argument(k, help=v, default=defaults.get(k, None), type=datatypes.get(k, None))
    parser = parser.parse_args()

    nruns = parser.crystals
    outFN = parser.out

    model = None
    for runnumber in range(1, nruns+1):
        crystal = shoot_crystal(parser.Foff, parser.Fon, **vars(parser))
        crystal['RUN'] = runnumber
        if runnumber == 1:
            with open(outFN, 'w') as out:
                crystal.to_csv(out, header=True)
        else:
            with open(outFN, 'a') as out:
                crystal.to_csv(out, header=False)

if __name__=="__main__":
    main()
