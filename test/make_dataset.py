import argparse
import numpy as np
import pandas as pd

argDict = {
    "Fon"                      : "CNS file containing pumped structure factor amplitudes.",
    "Foff"                     : "CNS file containing un-pumped structure factor amplitudes.",
    "out"                      : "CSV intensity file to output.", 
    "--multiplicity"           : "Multiplicity of the dataset. Default is 10.",
    "--missing"                : "Fraction of missing reflections.",
    "--intensityshape"         : "Shape parameter for the distribution of intensities",
    "--intensityloc"           : "Location parameter for the distribution of intensities",
    "--intensityslope"         : "Relation between intensity metadata and reflection intensities",
    "--intensityoffset"        : "Intercept of the intensity measurements",
    "--reflectionsperimage"    : "Average number of reflections per image (default 150)",
    "--reflectionsperimagestd" : "Std deviation of reflectiosn per image (default 50)",
    "--minreflectionsperimage" : "Minimum reflections to generate per image",
    "--minimages"              : "Minimum images per run/crystal",
    "--meannimages"            : "Mean images per run/crystal",
    "--stdimages"              : "Standard deviation images per run/crystal",
    "--onreps"                 : "Number of on images per phi angle.",
    "--offreps"                : "Number of off images per phi angle.",
    "--sigintercept"           : "Sigma(I) = m*I + b. b",
    "--sigslope"               : "Sigma(I) = m*I + b. m",
    "--partialitymean"         : "Average partiality of reflections",
    "--partialitystd"          : "Average partiality of reflections",
    "--partialitymin"          : "Minimum partiality of reflections",
    "--sigInoise"              : "Fractional error. SigI=error*I + u",
    "--runlength"              : "Approximate number of phi angles per crystal",
}

defaults = {
    "--multiplicity"           : 10.,
    "--missing"                : 0.0, 
    "--intensityshape"         : 2.,
    "--intensityloc"           : 0.2, 
    "--intensityslope"         : 1.0,
    "--intensityoffset"        : 0.0,
    "--reflectionsperimage"    : 150,
    "--reflectionsperimagestd" : 50,
    "--minreflectionsperimage" : 50,
    "--minimages"              : 10,
    "--meannimages"            : 10,
    "--stdimages"              : 10, 
    "--onreps"                 : 4, 
    "--offreps"                : 4,
    "--sigintercept"           : 5.,
    "--sigslope"               : 0.03, 
    "--partialitymean"         : 0.6,
    "--partialitystd"          : 0.2, 
    "--partialitymin"          : 0.1, 
    "--sigInoise"              : 0.03,
    "--runlength"              : 50, 
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


offFN = '1ubq.pdb.hkl'
onFN  = '1ubq-flip.pdb.hkl'


Foff = parsehkl(offFN)
Fon  = parsehkl(onFN)

def build_model(offFN, onFN, **kw):
    Fon,Foff = parsehkl(onFN),parsehkl(offFN)
    Foff['gamma'] = np.square(Fon['F']/Foff['F'])
    model = Foff.sample(frac=kw.get('multiplicity', 10.), replace=True)
    model['Foff'] = model['F']
    model['Fon'] = Fon.loc[model.index]
    del model['F']

    #you could do this with np.repeat in a couple of lines but it is unreadable
    minref  = kw.get('minrefelctions', 50)
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
    model["IMAGENUMBER"] = imagenumber

    meanimages = kw.get('meanimages', 50)
    stdimages = kw.get('stdimages', 20)
    minimages = kw.get('minimages', 10)
    runs = []
    images = list(model['IMAGENUMBER'].unique())
    while len(images) > 0:
        ct = max(minimages, int(np.random.normal(meanimages, stdimages)))
        if len(images) < ct:
            ct = len(images)
        runs.append([images.pop(0) for i in range(ct)])

    model['RUN'] = 0
    for n,run in enumerate(runs, 1):
        model.loc[model['IMAGENUMBER'].isin(run), 'RUN'] = n

    partiality = np.random.normal(kw.get("partialitymean", 0.6), kw.get("partialitystd", 0.2), len(model))
    partiality = np.minimum(partiality, 1.)
    partiality = np.maximum(partiality, kw.get("partialitymin", 0.1))
    model['P'] = partiality
    I = None
    for i in range(kw.get("onreps", 4)):
        iobs = model.copy()
        iobs['SERIES'] = 'on{}'.format(i + 1)
        iobs['IOBS'] = model['P']*model['Fon']**2
        iobs['SIGMA(IOBS)'] = kw.get("sigintercept", 5.0) + kw.get("sigslope", 0.03)*iobs['IOBS']
        iobs['IOBS'] = [np.random.normal(i,j) for i,j in zip(iobs['IOBS'], iobs['SIGMA(IOBS)'])]
        I = pd.concat((I, iobs))

    for i in range(kw.get("offreps", 4)):
        iobs = model.copy()
        iobs['SERIES'] = 'off{}'.format(i + 1)
        iobs['IOBS'] = model['P']*model['Foff']**2
        iobs['SIGMA(IOBS)'] = kw.get("sigintercept", 5.0) + kw.get("sigslope", 0.03)*iobs['IOBS']
        iobs['IOBS'] = [np.random.normal(i,j) for i,j in zip(iobs['IOBS'], iobs['SIGMA(IOBS)'])]
        I = pd.concat((I, iobs))

    return I.sample(frac = 1. - kw.get('missing', 0.), replace=False)



def main():
    parser = argparse.ArgumentParser()
    for k,v in argDict.items():
        parser.add_argument(k, help=v, default=defaults.get(k, None))

    #TODO: parser.add_argument("--signal_decay", default=None, help="Exponential decay constant to model how I/sig(I) decays with resolution. Default is 0 (no decay).")

    parser = parser.parse_args()
    model  = build_model(parser.Fon, parser.Foff, **vars(parser))
    model.to_csv(parser.out)


if __name__=="__main__":
    main()
