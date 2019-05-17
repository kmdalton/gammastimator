import tensorflow as tf
import pandas as pd
import numpy as np
from sys import stdout
import tensorflow_probability as tfp
from scipy import sparse
from tensorflow.python.ops.parallel_for.gradients import jacobian
from tensorflow.contrib.layers import dense_to_sparse
pd.options.mode.chained_assignment = None


"""
Implementation of delta F estimation using the Coppens ratio method with the random diffuse model. 
This implementation will individually weight reflection intensities by their corresponding Io estimates.
I will also seek to implement dose corrections for radiation damage. 

TODO:
1) Implement without weights or radiation damage
2) Add weights
3) Add radiation Damage
"""

    


def fit_model(df, rho, l, intensitykey='ipm2', referencekey='FCALC', optimizer=None, config=None, maxiter=1000, tolerance=1e-5, verbose=False):
    tf.reset_default_graph()

    rho = np.array(rho).flatten()
    l = np.array(l).flatten()

    indices = {
        'GAMMAINDEX'        : ['MERGEDH', 'MERGEDK', 'MERGEDL'],
        'RUNINDEX'          : 'RUN',
        'IMAGEINDEX'        : ['RUN', 'PHINUMBER', 'SERIES'],
        'PHIINDEX'          : ['RUN', 'PHINUMBER'],
    }

    for k,v in indices.items():
        df.loc[:,k] = df.groupby(v).ngroup()
    df

    #Prepare the per image metadata
    k = [i for i in df if 'ipm' in i.lower()]
    k += ['RUNINDEX']
    imagemetadata = df[k + ['IMAGEINDEX']].groupby('IMAGEINDEX').first()

    #Construct pivot tables of intensities, errors, metadatakeys
    iobs        = df.pivot_table(values='IOBS', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)
    imagenumber = df.pivot_table(values='IMAGEINDEX', index=['H', 'K', 'L', 'RUNINDEX', 'PHIINDEX'], columns='SERIES', fill_value=0)
    sigma       = df.pivot_table(values='SIGMA(IOBS)', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)

    ion    = iobs[[i for i in iobs if  'on' in i]].values
    ioff   = iobs[[i for i in iobs if 'off' in i]].values
    sigon  = sigma[[i for i in sigma if  'on' in i]].values
    sigoff = sigma[[i for i in sigma if 'off' in i]].values
    imon   = imagenumber[[i for i in imagenumber if  'on' in i]].values
    imoff  = imagenumber[[i for i in imagenumber if 'off' in i]].values
    cardon  = (ion  > 0).sum(1)
    cardoff = (ioff > 0).sum(1)

    ion     = tf.convert_to_tensor(ion, dtype=tf.float32, name='ion')
    ioff    = tf.convert_to_tensor(ioff, dtype=tf.float32, name='ioff')
    sigon   = tf.convert_to_tensor(sigon, dtype=tf.float32, name='sigon')
    sigoff  = tf.convert_to_tensor(sigoff, dtype=tf.float32, name='sigoff')
    cardon  = tf.convert_to_tensor(cardon, dtype=tf.float32, name='cardon')
    cardoff = tf.convert_to_tensor(cardoff, dtype=tf.float32, name='cardoff')

    #Problem Variables
    gammaidx = df.pivot_table(values='GAMMAINDEX', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'])
    gammaidx = np.array(gammaidx).flatten()
    Foff = tf.constant(df[['GAMMAINDEX', referencekey]].groupby('GAMMAINDEX').first().values.flatten(), tf.float32)
    h = gammaidx.max() + 1
    ipm = np.float32(imagemetadata[intensitykey])
    variables = tf.Variable(np.concatenate((np.ones(h, dtype=np.float32), ipm)))
    ipm = tf.constant(ipm)
    gammas = tf.nn.softplus(variables[:h])
    Icryst = tf.nn.softplus(variables[h:])

    Bon  = tf.gather(Icryst, imon)
    Boff = tf.gather(Icryst, imoff)

    n = int(Icryst.shape[0]) #Is this variable ever referenced? 

    variance = tf.math.reduce_sum((tf.gather(gammas, gammaidx)[:,None]*sigoff/Boff/cardoff[:,None])**2, 1) + \
               tf.math.reduce_sum((sigon/Bon/cardon[:,None])**2, 1) 


    likelihood = tf.losses.mean_squared_error(
        tf.reduce_sum(ioff/Boff/cardoff[:,None], 1)*tf.gather(gammas, gammaidx), 
        tf.reduce_sum(ion/Bon/cardon[:,None], 1), 
        weights=variance**-1
    )

    rhop = tf.placeholder(tf.float32)
    lp   = tf.placeholder(tf.float32)

    regularizer = rhop*tf.losses.mean_squared_error(ipm, Icryst)
    sparsifier  = lp*tf.losses.mean_squared_error(tf.zeros(gammas.shape), Foff*(gammas - 1))

    loss = likelihood + regularizer + sparsifier
    #Foff = np.ones(len(gammaidx))


    #print("6: {}".format(time() - start))
    H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
    K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
    L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import tensor_array_ops
    from tensorflow.python.ops import control_flow_ops
    from tensorflow import dtypes

    # Declare an iterator and tensor array loop variables for the gradients.
    n = array_ops.size(gammas)
    loop_vars = [
        array_ops.constant(0, dtypes.int32),
        tensor_array_ops.TensorArray(gammas.dtype, n)
    ]    
    # Iterate over all elements of the gradient and compute second order
    # derivatives.
    gradients = tf.gradients(loss, gammas)[0]
    gradients = array_ops.reshape(gradients, [-1])
    _, diag_A = control_flow_ops.while_loop(
        lambda j, _: j < n, 
        lambda j, result: (j + 1, 
                           result.write(j, tf.gather(tf.gradients(gradients[j], gammas)[0], j))),
        loop_vars
    )    
    diag_A = array_ops.reshape(diag_A.stack(), [n])

    #diag_A = tf.diag_part(tf.hessians(loss, gammas)[0])
    #diag_A = tfp.math.diag_jacobian(gammas, tfp.math.diag_jacobian(gammas, loss))
    #diag_A = [tf.gradients(tf.gradients(loss, gammas[i]), gammas[i]) for i in range(h)]
    #B = dense_to_sparse(jacobian(tf.gradients(loss, gammas)[0], Icryst, use_pfor=False))
    #C = dense_to_sparse(tf.hessians(loss, Icryst)[0])

    result = pd.DataFrame()
    intensities = pd.DataFrame()
    lossdf = pd.DataFrame()
    optimizer = tf.train.AdamOptimizer(0.05) if optimizer is None else optimizer
    optimizer = optimizer.minimize(loss)
    for iternumber,(rho_,l_) in enumerate(zip(rho, l)):
        losses = [[], [], [], []]
        feed_dict = {lp: l_, rhop: rho_}
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

            # Loop and a half problem, thanks mehran
            _, loss__, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
            losses[0].append(loss__)
            losses[1].append(like)
            losses[2].append(reg)
            losses[3].append(sp)
            if np.isnan(loss__) or np.isinf(loss__):
                print("Initial Loss is NaN!!")
                break

            for i in range(maxiter):
                _, loss_ = sess.run((optimizer, loss), feed_dict=feed_dict)
                _, loss_, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
                losses[0].append(loss_)
                losses[1].append(like)
                losses[2].append(reg)
                losses[3].append(sp)

                if np.isnan(loss__) or np.isinf(loss__):
                    print("Desired error not achieved due to precision loss. Exiting after {} iterations".format(i+1))
                    break

                #Absolute fractional change
                if np.abs(loss_ - loss__)/loss_ < tolerance:
                    if verbose:
                        percent = 100*iternumber/len(rho)
                        message = "Converged to tol={} after {} iterations. {} % complete...".format(tolerance, i, percent)
                        print(message, end='\r')
                    break

                loss__ = loss_

            #print(sess.run((loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict))
            gammas_, Icryst_, ipm_ = sess.run((gammas, Icryst, ipm), feed_dict=feed_dict)

            # These are the parts of the Hessian needed to compute error estimates
            diag_A_ = sess.run(diag_A, feed_dict=feed_dict)
        F = pd.DataFrame()
        F.loc[:,'H'] = H
        F.loc[:,'K'] = K
        F.loc[:,'L'] = L
        F.loc[:,'GAMMA'] = gammas_
        F = F.set_index(['H', 'K', 'L'])

        F.loc[:,'FCALC'] = np.array(df.groupby('GAMMAINDEX').first()['FCALC'], dtype=float)
        F.loc[:,'PHIC'] = np.array(df.groupby('GAMMAINDEX').first()['PHIC'], dtype=float)
        F.loc[:,'FOBS'] = np.array(df.groupby('GAMMAINDEX').first()['FOBS'], dtype=float)
        F.loc[:,'SIGMA(FOBS)'] = np.array(df.groupby('GAMMAINDEX').first()['SIGMA(FOBS)'], dtype=float)
        F.loc[:,'D'] = np.array(df.groupby('GAMMAINDEX').first()['D'], dtype=float)
        F.loc[:,'DeltaF'] = F[referencekey]*(F['GAMMA'] - 1)

        I = pd.DataFrame()
        I.loc[:,intensitykey] = ipm_
        I.loc[:,'Icryst'] = Icryst_

        sigmagamma = np.array(diag_A_)
        sigmagamma[sigmagamma > 0] = 1./sigmagamma[sigmagamma > 0]
        F.loc[:,'SIGMA(GAMMA)'] = sigmagamma
        F.loc[:,'SIGMA(DeltaF)'] = np.abs(0.5*sigmagamma*F[referencekey]*F['GAMMA']**-0.5)

        F.loc[:,'LAMBDA'] = l_
        F.loc[:,'RHO'] = rho_
        I.loc[:,'LAMBDA'] = l_
        I.loc[:,'RHO'] = rho_

        lossdf_ = pd.DataFrame()
        lossdf_.loc[:,'STEP']        = range(len(losses[0]))
        lossdf_.loc[:,'LOSS']        = losses[0]
        lossdf_.loc[:,'LIKELIHOOD']  = losses[1]
        lossdf_.loc[:,'REGULARIZER'] = losses[2]
        lossdf_.loc[:,'SPARSIFIER']  = losses[3]
        lossdf_.loc[:,'RHO']         = rho_
        lossdf_.loc[:,'LAMBDA']      = l_

        result = pd.concat((result, F))
        intensities = pd.concat((intensities, I))
        lossdf = pd.concat((lossdf, lossdf_))
    #from IPython import embed
    #embed()

    return result, intensities, lossdf





if __name__=="__main__":
    from sys import argv
    import pandas as pd
    inFN = argv[1]
    df = pd.read_csv(inFN)
    F,Icryst,losses = fit_model(
        df, 
        #optimizer=optim, 
        maxiter=10000,
        compute_full_hessian=False, 
        rho=0.005, 
        l=1e-5, 
        tolerance=1e-5, 
        intensitykey='ipm2',
        referencekey='FCALC',
        use_weights=True,
    )



