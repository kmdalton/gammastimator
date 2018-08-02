import tensorflow as tf
import pandas as pd
import numpy as np
import argparse


argDict = {
    "inFN"  : "csv file containing ratiometic TR-X data", 
}


def pare_data(dataframe):
    """Remove reflection observations from which gammas cannot be estimated due to missing on or off reflections."""
    indexnames = dataframe.index.names
    df = dataframe.reset_index().set_index(['H', 'K', 'L', 'RUN', 'PHINUMBER'])
    df = df.loc[df[df.SERIES.str.contains('on')].index.intersection(df[df.SERIES.str.contains('off')].index)]
    if None not in indexnames:
        return df.reset_index().set_index(indexnames)
    else:
        return df.reset_index()

def image_metadata(dataframe, keys = None):
    """Aggregate the image metadata from an integration run into a separate dataframe"""
    if keys is None:
        keys = [k for k in dataframe.keys() if 'ipm' in k.lower()]
        specifically_check = ['Io', 'BEAMX', 'BEAMY', 'Icryst', 'SERIES', 'RUN']
        for k in specifically_check:
            if k in dataframe:
                keys.append(k)
    return dataframe[['IMAGENUMBER'] + list(keys)].groupby('IMAGENUMBER').mean()

def raw_gammas(dataframe):
    """Compute uncorrected intensity ratios, return (raw gamma array, indexing array)"""

    I = pare_data(dataframe)
    iobs        = I.pivot_table(values='IOBS', index=['H', 'K', 'L', 'RUN', 'PHINUMBER'], columns='SERIES', fill_value=np.NaN) 
    imagenumber = I.pivot_table(values='IMAGENUMBER', index=['H', 'K', 'L', 'RUN', 'PHINUMBER'], columns='SERIES', fill_value=-1)
    gammas = iobs[[i for i in iobs if 'on' in i]].sum(1) / iobs[[i for i in iobs if 'off' in i]].sum(1)
    return gammas, imagenumber


def main():
    parser = argparse.ArgumentParser()
    for k,v in argDict.items():
        parser.add_argument(k, help=v, default=defaults.get(k, None), type=datatypes.get(k, None))
    parser = parser.parse_args()
    data = pare_data(pd.read_csv(parser.inFN))
    gammas,indices = raw_gammas(data)

    #h is a dataframe that maps each h,k,l to a unique integer
    h = g.reset_index()[['H','K','L']].drop_duplicates().reset_index(drop=True).reset_index().pivot_table(index=['H','K','L'], values='index')
    gammaidx = h.loc[gammas.reset_index().set_index(['H', 'K', 'L'])

    r = len(gammas.reset_index().groupby('RUN'))
    runidx = gammas.reset_index()['RUN'] - 1
    M = image_metadata(data)

    tf.reset_default_graph()
    #Constants 
    raw_gammas = tf.constant(gammas)
    ipm        = tf.constant(M['IPM'])
    ipm_x      = tf.constant(M['IPM_X'])
    ipm_y      = tf.constant(M['IPM_Y'])

    #Regularization strength
    rho = tf.placeholder(tf.float32)

    #LCs for scaling IPM data
    x_intercept   = tf.Variable(0.)
    x_slope       = tf.Variable(1.)
    y_intercept   = tf.Variable(0.)
    y_slope       = tf.Variable(1.)
    ipm_intercept = tf.Variable(0.)
    ipm_slope     = tf.Variable(1.)

    #Beam shape
    sigx          = tf.Variable(10.)
    sigy          = tf.Variable(10.)

    #Crystal dimensions
    xmin          = tf.Variable(-50.*np.ones(r))
    xmax          = tf.Variable( 50.*np.ones(r))
    ymin          = tf.Variable(-50.*np.ones(r))
    ymax          = tf.Variable( 50.*np.ones(r))

    #gammastimates
    gamma         = tf.Variable(np.ones(len(h)))

    beamx  = ipm_x * x_slope + x_intercept
    beamy  = ipm_y * y_slope + y_intercept

    Icryst = 0.25*(ipm_slope*ipm + ipm_intercept) * (
        tf.erf((tf.gather(xmin, runidx) - beamx)/sigx) - tf.erf((tf.gather(xmax, runidx) - beamx)/sigx)
        ) * (
        tf.erf((tf.gather(ymin, runidx) - beamy)/sigy) - tf.erf((tf.gather(ymax, runidx) - beamy)/sigy)
        )
        
    tf.gather(gamma, gammaidx) - rawgammas

if __name__=="__main__":
    main()
