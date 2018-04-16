import argparse
import numpy as np
import pandas as pd



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

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Fon", help="CNS file containing pumped structure factor amplitudes.")
    parser.add_argument("Foff", help="CNS file containing un-pumped structure factor amplitudes.")
    parser.add_argument("out.csv", help="CSV intensity file to output.")
    #TODO: parser.add_argument("--signal_decay", default=None, help="Exponential decay constant to model how I/sig(I) decays with resolution. Default is 0 (no decay).")
    parser.add_argument("--multiplicity", default=10., help="Multiplicity of the dataset. Default is 10.")
    parser.add_argument("--missing", default=0.0, help="Fraction of missing reflections.")
    parser.add_argument("--intensityshape", default=2., help="Shape parameter for the distribution of intensities")
    parser.add_argument("--intensityloc", default=0.2, help="Location parameter for the distribution of intensities")
    parser.add_argument("--intensityslope", default=1.0, help="Relation between intensity metadata and reflection intensities")
    parser.add_argument("--intensityoffset", default=0.0, help="Intercept of the intensity measurements")
    parser.add_argument("--reflectionsperimage", default=150, help="Average number of reflections per image (default 150)")
    parser.add_argument("--reflectionsperimagestd", default=50, help="Std deviation of reflectiosn per image (default 50)")
    parser.add_argument("--minreflectionsperimage", default=50, help="Minimum reflections to generate per image")
    parser.add_argument("--minimages", default=10, help="Minimum images per run/crystal")
    parser.add_argument("--meannimages", default=10, help="Mean images per run/crystal")
    parser.add_argument("--stdimages", default=10, help="Standard deviation images per run/crystal")
    parser.add_argument("--onreps", default=4, help="Number of on images per phi angle.")
    parser.add_argument("--offreps", default=4, help="Number of off images per phi angle.")
    parser.add_argument("--partialitymean", default=0.6, help="Average partiality of reflections")
    parser.add_argument("--partialitystd", default=0.2, help="Average partiality of reflections")
    parser.add_argument("--runlength", default=50, help="Approximate number of phi angles per crystal")
    parser = parser.parse_args()
    model  = build_model(parser.Fon, parser.Foff, **vars(parser))


if __name__=="__main__":
    main()
