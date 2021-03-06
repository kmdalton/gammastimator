import crystal
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from tensorflow.contrib.opt import ScipyOptimizerInterface


argDict = {
    "inFN"  : "csv file containing ratiometic TR-X data", 
}


def pare_data(dataframe, columns=None):
    """Remove reflection observations from which gammas cannot be estimated due to missing on or off reflections. Remove columns which won't be used."""
    if columns is None:
        columns = {
            'RUN' : int,
            'PHINUMBER': int,
            'SERIES': str,
            'H': int,
            'K': int,
            'L': int,
            'MERGEDH': int,
            'MERGEDK': int,
            'MERGEDL': int,
            'IOBS': float,
            'SIGMA(IOBS)': float,
            'D': float, 
        }
        columns.update({i: float for i in dataframe.keys() if 'ipm' in i.lower()})
    dataframe = dataframe[[i for i in columns if i in dataframe]]
    for k in dataframe:
        if k in columns:
            dataframe[k] = dataframe[k].astype(columns[k])
        else:
            del dataframe[k]

    #print("Number of reflection observations: {}".format(len(dataframe)))
    #print("Multiplicity: {}".format(len(dataframe)/len(dataframe.groupby(['H', 'K', 'L']))))

    #This removes reflections which were not observed in the 'on' and 'off' datasets at a given rotation
    #TODO: This line could use some serious optimization. It seems inocuous but runs very slow
    #dataframe = dataframe.groupby(['H', 'K', 'L', 'RUN', 'PHINUMBER']).filter(lambda x: x.SERIES.str.contains('on').max() and x.SERIES.str.contains('off').max())

    dataframe['on'] = dataframe.SERIES.str.contains('on')
    #dataframe['off'] = dataframe.SERIES.str.contains('off')
    dataframe = dataframe.groupby(['H', 'K', 'L', 'RUN', 'PHINUMBER']).filter(lambda x: x.on.max() and ~x.on.min())

    del dataframe['on']
    #gammaobs = len(dataframe.groupby(['H', 'K', 'L', 'RUN', 'PHINUMBER']))
    #gammamult = gammaobs / len(dataframe.groupby(['H', 'K', 'L']))
    #print("Number of ratio observations: {}".format(gammaobs))
    #print("Ratio multiplicity: {}".format(gammamult)) 
    return dataframe

def add_index_cols(dataframe):
    """
    Parameters
    ----------
    dataframe : pd.DataFrame
        A dataframe containing integrated diffraction data

    Returns
    -------
    dataframe : pd.DataFrame
        Modified dataframe with new columns for addressing individual observations of intensity ratios
    

    It is important to establish some numerical indices in
    order to index arrays in the optimization problem. To this end, 
    this function will define the following new columns to simplify 
    indexing ops.

    "GAMMAINDEX"
        A unique numeric index is assigned to each combination
        of H, K, and L. This index uses the "MERGEDH/K/L"
        attributes in the dataframe. This way we don't estimate
        more gammas than are truly necessary. This should not be
        used for grouping observations to make ratiometric
        observations, because it will group together equivalent
        observations in a single image should they exist. That
        would be problematic, because equivalent observations will
        certainly not have the same partiality.

    "RUNINDEX"
        A unique, sequential identifier for each run/crystal.
        Use this for indexing per crystal parameters.

    "IMAGEINDEX"
        A unique, sequential identifier for each image in the
        dataset. Use this for adding per shot parameters.

    "PHIINDEX"
        A unique, sequential identifier for each group of shots
        on the same crystal at the same rotation angle.
    """

    indices = {
        'GAMMAINDEX' : ['MERGEDH', 'MERGEDK', 'MERGEDL'],
        'RUNINDEX'   : 'RUN',
        'IMAGEINDEX' : ['RUN', 'PHINUMBER', 'SERIES'],
        'PHIINDEX'   : ['RUN', 'PHINUMBER'],
    }

    for k,v in indices.items():
        dataframe[k] = dataframe.groupby(v).ngroup()
    return dataframe

def scramble_labels(*args):
    #TODO: can this be done easier with indices?
    idx1,idx2 = np.where(~np.isnan(args[0]))
    tmp = -np.ones(args[0].shape, dtype=int)
    tmp[idx1, idx2] = idx2
    for i in tmp:
        i[i > -1] = np.random.permutation(i[i > -1])
    permuted_idx2 = tmp[tmp > -1]
    for i,df in enumerate(args):
        data = df.values
        data[idx1, idx2] = data[idx1, permuted_idx2]
        df[df.keys()] = data
from time import time

def sparsedeltaFestimate(dataframe, xposkey=None, yposkey=None, intensitykey=None, scramble=None, learning_rate=None, tolerance=None, maxiter=None, rho=None):
    start = time()
    maxiter = 2000 if maxiter is None else maxiter
    tolerance = 1e-3 if tolerance is None else tolerance
    learning_rate = 20 if learning_rate is None else learning_rate
    scramble = False if scramble is None else bool(scramble)
    dataframe = add_index_cols(dataframe)
    xposkey = 'ipm2_xpos' if xposkey is None else xposkey
    yposkey = 'ipm2_ypos' if yposkey is None else yposkey
    intensitykey = 'ipm2' if intensitykey is None else intensitykey

    xposkey = 'IPM_X' if xposkey not in dataframe else xposkey
    yposkey = 'IPM_Y' if yposkey not in dataframe else yposkey
    intensitykey = 'IPM' if intensitykey not in dataframe else intensitykey

    #print("1: {}".format(time() - start))
    #Prepare the per image metadata
    k = [i for i in dataframe if 'ipm' in i.lower()]
    k += ['RUNINDEX']
    imagemetadata = dataframe[k + ['IMAGEINDEX']].groupby('IMAGEINDEX').mean()

    #print("2: {}".format(time() - start))
    #Construct pivot tables of intensities, errors, metadatakeys
    iobs        = dataframe.pivot_table(values='IOBS', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=np.NaN)
    imagenumber = dataframe.pivot_table(values='IMAGEINDEX', index=['H', 'K', 'L', 'RUNINDEX', 'PHIINDEX'], columns='SERIES', fill_value=-1)
    sigma       = dataframe.pivot_table(values='SIGMA(IOBS)', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=np.NaN)

    #print("3: {}".format(time() - start))
    #Optionally permute the series labels in order to scramble assignments of on/off intensities (negative control)
    if scramble:
        #I sure hope this works inplace
        scramble_labels(iobs, imagenumber, sigma)

    #print("4: {}".format(time() - start))
    #Compute raw gammas without beam intensity adjustments
    ion    = iobs[[i for i in iobs if  'on' in i]].sum(1)
    ioff   = iobs[[i for i in iobs if 'off' in i]].sum(1)
    gammas = ion / ioff
    gammaidx = dataframe.pivot_table(values='GAMMAINDEX', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'])
    gammaidx = np.array(gammaidx).flatten()

    #print("5: {}".format(time() - start))
    #We want to use the error estimates from the integration to weight the merging
    sigmaion    = np.sqrt(np.square(iobs[[i for i in iobs if  'on' in i]]).sum(1))
    sigmaioff   = np.sqrt(np.square(iobs[[i for i in iobs if 'off' in i]]).sum(1))
    sigmagamma  = np.abs(gammas)*np.sqrt(np.square(sigmaion / ion) + np.square(sigmaioff / ioff))
    mergingweights = (1. / np.array(sigmagamma, dtype=np.float32)) * ((1./np.array([(1./sigmagamma.iloc[gammaidx == i]).sum() for i in range(gammaidx.max() + 1)], dtype=np.float32))[gammaidx])

    #print("6: {}".format(time() - start))
    H = np.array(dataframe.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
    K = np.array(dataframe.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
    L = np.array(dataframe.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

    #return iobs, imagenumber, sigma, gammas, sigmagamma

    """
    This block is where we construct the tensorflow graph. 
    """
    tf.reset_default_graph()
    h = gammaidx.max() + 1
    r = len(gammas.reset_index().groupby('RUNINDEX'))
    runidx = np.array(imagemetadata['RUNINDEX'])

    #We need two sparse tensors to map from Icryst estimates into the liklihood function. 
    #First the 'on' shots
    tmp = np.array(imagenumber[[i for i in imagenumber if 'on' in i]])
    idx = np.vstack((np.indices(tmp.shape)[0][tmp >= 0], tmp[tmp >= 0])).T
    onimageidx = tf.SparseTensor(idx, np.ones(len(idx), dtype=np.float32), (len(imagenumber), len(imagemetadata)))
    onimageidx = tf.sparse_reorder(onimageidx)

    #Now the 'off' shots
    tmp = np.array(imagenumber[[i for i in imagenumber if 'off' in i]])
    idx = np.vstack((np.indices(tmp.shape)[0][tmp >= 0], tmp[tmp >= 0])).T
    offimageidx = tf.SparseTensor(idx, np.ones(len(idx), dtype=np.float32), (len(imagenumber), len(imagemetadata)))
    offimageidx = tf.sparse_reorder(offimageidx)

    #Let us have a sparse matrix for averaging hkl observations with weights
    idx = np.vstack((
        gammaidx,
        np.arange(len(gammaidx), dtype=int), 
    )).T
    tshape = (
        gammaidx.max()+1,
        len(gammaidx), 
    )
    vals = mergingweights
    mergingtensor = tf.SparseTensor(idx, vals, tshape)
    mergingtensor = tf.sparse_reorder(mergingtensor)


    #Constants 
    raw_gammas   = tf.constant(np.float32(gammas))
    sigma_gammas = tf.constant(np.float32(sigmagamma))
    ipm          = tf.constant(np.float32(imagemetadata[intensitykey]))
    ipm_x        = tf.constant(np.float32(imagemetadata[xposkey]))
    ipm_y        = tf.constant(np.float32(imagemetadata[yposkey]))

    #LCs for scaling IPM data
    x_intercept   = tf.constant(imagemetadata[xposkey].mean())
    x_slope       = tf.constant(1./imagemetadata[xposkey].std())
    y_intercept   = tf.constant(imagemetadata[yposkey].mean())
    y_slope       = tf.constant(1./imagemetadata[yposkey].std())
    ipm_slope     = tf.constant(1.)
    ipm_intercept = tf.Variable(0.)

    #Beam shape is fixed for this model
    sigx          = tf.constant(50.)
    sigy          = tf.constant(50.)

    #Crystal dimensions are the real optimization parameters
    xmin          = tf.Variable(-50.*np.ones(r, dtype=np.float32))
    xmax          = tf.Variable( 50.*np.ones(r, dtype=np.float32))
    ymin          = tf.Variable(-50.*np.ones(r, dtype=np.float32))
    ymax          = tf.Variable( 50.*np.ones(r, dtype=np.float32))

    #gammastimates
    gamma         = tf.Variable(np.ones(h), dtype=np.float32)
    beamx  = ipm_x * x_slope + x_intercept
    beamy  = ipm_y * y_slope + y_intercept


    #This represents the actual intensity on the crystal
    Icryst = 0.25*(ipm_slope*ipm + ipm_intercept) * (
        tf.erf((tf.gather(xmin, runidx) - beamx)/sigx) - tf.erf((tf.gather(xmax, runidx) - beamx)/sigx)
        ) * (
        tf.erf((tf.gather(ymin, runidx) - beamy)/sigy) - tf.erf((tf.gather(ymax, runidx) - beamy)/sigy)
        )

    if rho is not None:
        err = tf.Variable(np.zeros(len(runidx), dtype=np.float32))
        Icryst = Icryst + err

    Bon  = tf.squeeze(tf.sparse_tensor_dense_matmul( onimageidx, tf.expand_dims(Icryst, 1)))
    Boff = tf.squeeze(tf.sparse_tensor_dense_matmul(offimageidx, tf.expand_dims(Icryst, 1)))


    g = tf.squeeze(tf.sparse_tensor_dense_matmul(mergingtensor, tf.expand_dims(raw_gammas*Boff/(Bon), 1)))
    variance_g = tf.squeeze(tf.sparse_tensor_dense_matmul(mergingtensor, tf.square(tf.expand_dims(raw_gammas*Boff/(Bon), 1))))
    deltaFoverF = (tf.sqrt(g) - 1.)
    variance_deltaFestimate  = variance_g*(deltaFoverF/g)**2

    if rho is not None:
        f_ = int(deltaFoverF.get_shape()[0])
        e_ = int(err.get_shape()[0])
        loss = (1/f_)*(1. - rho)*tf.norm(deltaFoverF, 1) + (1/e_)*rho*tf.norm(err, 1)
    else:
        f_ = int(deltaFoverF.get_shape()[0])
        loss = (1/f_)*tf.norm(deltaFoverF, 1)


    """
    optimizer = ScipyOptimizerInterface(loss, method='CG', options={'ftol': 1e-5*np.finfo(float).eps})
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        deltaFestimate = sess.run(deltaFoverF)
        optimizer.minimize(sess)
        loss_ = sess.run(loss)
        if np.isnan(loss_):
            print("Desired error not achieved due to precision loss.")
        else:
            deltaFestimate = sess.run(deltaFoverF)


    """
    #TODO: Implement custom optimizers
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate, 0.1).minimize(loss)

    #print("7: {}".format(time() - start))
    deltaFestimate = None
    variance = None
    with tf.Session() as sess:
        #print("8: {}".format(time() - start))
        sess.run(tf.global_variables_initializer())
        loss__ = sess.run(loss)
        if np.isnan(loss__):
            print("Initial loss is NaN, aborting.")
        _, loss_ = sess.run((optimizer, loss))
        if np.isnan(loss__):
            print("Desired error not achieved due to precision loss. Exiting after 0 iterations")
        #loss_ = sess.run((optimizer, loss))
        #print("loss: {}".format(loss_))
        #print("9: {}".format(time() - start))
        for i in range(maxiter):
            deltaFestimate = sess.run(deltaFoverF)
            variance = sess.run(variance_deltaFestimate)
            _, loss__ = sess.run((optimizer, loss))
            #print("loss: {}".format(loss__))
            if np.isnan(loss__):
                print("Desired error not achieved due to precision loss. Exiting after {} iterations".format(i+1))
                break
            #Absolute fractional change
            if np.abs(loss_ - loss__)/loss_ <= tolerance: 
                print("Converged to tol={} after {} iterations.".format(tolerance, i))
                deltaFestimate = sess.run(deltaFoverF)
                variance = sess.run(variance_deltaFestimate)
                break
            loss_ = loss__

    #print("9: {}".format(time() - start))
    result = pd.DataFrame()
    result['H'] = H
    result['K'] = K
    result['L'] = L
    result['DeltaFoverF'] = deltaFestimate
    result['Var(DeltaFoverF)'] = variance
    result = result.set_index(['H', 'K', 'L'])
    return result

def split(dataframe, groupingkeys = None, frac=None):
    groupingkeys = ['H', 'K', 'L', 'RUN', 'PHINUMBER'] if groupingkeys is None else groupingkeys
    frac = 0.5 if frac is None else frac
    g = dataframe.groupby(groupingkeys)
    labels = np.random.binomial(1, frac, g.ngroups).astype(bool)
    return dataframe[labels[g.ngroup()]] , dataframe[~labels[g.ngroup()]]

def cchalf(dataframe, function, bins):
    """
    Compute cchalf of a function over resolution bins
    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe containing the ratiometric TR-X data. Must have a column 'D' for reflection resolution. make sure to pare the dataframe first with gammastimator.pare_data. otherwise, there will be no way to estimate gammas. also truncate very small / negative intensities unless you are defining your own estimators. 
    function : callable
        a function that returns a dataframe of, for instance, sparce estimates of delta f. this is the thing from which correlation coefficients will be computed. 
    bins : int
        number of resolution bins
    Returns
    -------
    cchalf : np.ndarray
    binedges : np.ndarray
    """
    dist = dataframe.set_index(['H', 'K', 'L'])['D'].drop_duplicates()
    dmin = dist.min()
    dmax = dist.max()
    binedges = np.linspace(dmin**-2, dmax**-2, bins+1)**-0.5
    binedges = list(zip(binedges[:-1], binedges[1:]))
    a,b = split(dataframe)
    xval_a, xval_b  = function(a), function(b)
#TODO: Fix this awful hack
    key = [i for i in xval_a if i!='D'][0]
    xval_a, xval_b  = xval_a.join(dist),xval_b.join(dist)
    idx = xval_a.index.intersection(xval_b.index)
    xval_a,xval_b = xval_a.loc[idx],xval_b.loc[idx]
    cchalf = []
    for dmin,dmax in binedges:
        idx = (xval_a['D'] > dmin) & (xval_a['D'] < dmax)
        a = np.array(xval_a[idx][key]).flatten()
        b = np.array(xval_b[idx][key]).flatten()
        cchalf.append(np.corrcoef(a,b)[0, 1])
    return cchalf, binedges

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

def append_reference_data(dataframe, crystal):
    crystal.unmerge()
    for k in crystal:
        if k in dataframe:
            del dataframe[k]
    return dataframe.join(crystal, ['H', 'K', 'L']).dropna()

def main():
    parser = argparse.ArgumentParser()
    for k,v in argDict.items():
        parser.add_argument(k, help=v, default=defaults.get(k, None), type=datatypes.get(k, None))
    parser = parser.parse_args()
    data = pare_data(pd.read_csv(parser.inFN))
    gammas,indices = raw_gammas(data)

if __name__=="__main__":
    main()
