#Start by importing some useful modules
import numpy as np
import emcee
from astropy.table import Table,join
import astroroutines as AS
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.spatial
import random
import time

# Define the SED fitter function to be called by the muliprocessing task
def SED_Fitter(ARR):

    # Unpack the array
    obj   = ARR[0]
    imag  = ARR[1]
    ierr  = ARR[2]
    zmag  = ARR[3]
    zerr  = ARR[4]
    jmag  = ARR[5]
    jerr  = ARR[6]
    hmag  = ARR[7]
    herr  = ARR[8]
    kmag  = ARR[9]
    kerr  = ARR[10]
    w1mag = ARR[11]
    w1err = ARR[12]

    # Grab the list of model parameters and fluxes
    t1 = Table.read('Photometric_BTSETTL_Models_Scaled.fits')
    # Take just a subsample that lies well within the bounds that you care about
    # This step doesn't matter that much, just simplifies things a little
    t = t1[np.where( (t1['Temp'] >= 0) & (t1['Temp'] <= 5000) )] 
 

    # Shoe-horn existing data for entry into KDTree routines
    # I'm only fitting for Temp and Logg (and the scaling factor), but
    # you could easily add in metallicity and alpha abundance (and/or clouds)
    combined_x_y_arrays = np.dstack([t['Temp'].data.ravel(), t['Logg'].data.ravel()])[0]

    # Create the function to query the KDTree for the nearest model parameters
    def do_kdtree(combined_x_y_arrays, points):
        mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
        dist, indexes = mytree.query(points, distance_upper_bound=100) # I put a bound on this to allow the interpolation to go a little farther that the bounds of the parameter space
        try:
            return combined_x_y_arrays[indexes]
        except:
            return (0,0) # The model doesn't exist because you went far away from the endpoints
        
    # Check if the magnitude is available (in order of bands)
    # Basically creating a mask for the array of fluxes that will
    # be compared to the model
    binary = []
    # iz should always be available
    binary.append(0)
    binary.append(0)
    if jmag != -9999 and jerr != -9999: binary.append(0)
    else: binary.append(1)
    if hmag != -9999 and herr != -9999: binary.append(0)
    else: binary.append(1)
    if kmag != -9999 and kerr != -9999: binary.append(0)
    else: binary.append(1)
    if w1mag != -9999 and w1err != -9999: binary.append(0)
    else: binary.append(1)

    # Convert magnitudes into flux values and uncertanties
    #a1, a1un = AS.magtoflux(umag, 'asinh', 'ang', sdss='u', mag_uncert = uerr)
    #a2, a2un = AS.magtoflux(gmag, 'asinh', 'ang', sdss='g', mag_uncert = gerr)
    #a3, a3un = AS.magtoflux(rmag, 'asinh', 'ang', sdss='r', mag_uncert = rerr)
    a4, a4un = AS.magtoflux(imag, 'asinh', 'ang', sdss='i', mag_uncert = ierr)
    a5, a5un = AS.magtoflux(zmag, 'asinh', 'ang', sdss='z', mag_uncert = zerr)
    a6, a6un = AS.magtoflux(jmag, 'ang', mass='j', mag_uncert = jerr)
    a7, a7un = AS.magtoflux(hmag, 'ang', mass='h', mag_uncert = herr)
    a8, a8un = AS.magtoflux(kmag, 'ang', mass='k', mag_uncert = kerr)
    a9, a9un = AS.magtoflux(w1mag, 'ang', wise=1, powerlaw=-2, mag_uncert = w1err)
    #a10, a10un = AS.magtoflux(w2mag, 'ang', wise=2, powerlaw=-2, mag_uncert = w2err)
    #a11, a11un = AS.magtoflux(w3mag, 'ang', wise=3, powerlaw=-2, mag_uncert = w3err)
    #a12, a12un = AS.magtoflux(w4mag, 'ang', wise=4, powerlaw=-2, mag_uncert = w4err)

    # Put them into an array
    Fluxes = np.array([a4, a5, a6, a7, a8, a9])
    FluxesUncert = np.array([a4un, a5un, a6un, a7un, a8un, a9un])

    # Create the masked arrays
    Fluxes       = np.ma.array(Fluxes, mask=binary)
    FluxesUncert = np.ma.array(FluxesUncert, mask=binary)
 
    # Start with the likelihood function. This is the probability that the data resulted from
    #  the model, which is a function of your parameters. We're going to assume the likelihood
    #  is simply a Gaussian with mean and standard deviation set by the model and the error on 
    #  the data. This is usually an OK assumption (cf. Central Limit Theorem), but it may not
    #  always be appropriate.
    
    def ln_likelihood(parameters, fluxes, fluxerrs, t, binary):
        # Unpack your parameters and make your model
        Teff, logg, scale = parameters
            
        index = np.where( (t['Temp'] == Teff) & (t['Logg'] == logg) )[0][0]
            
        # The model is the model flux based off the index
        model_fluxes = np.ma.array( [t['Isum'][index], t['Zsum'][index],
                                     t['Jsum'][index], t['Hsum'][index], t['Ksum'][index],
                                     t['W1sum'][index] ], mask = binary ) * scale
         
        # Now we can return the ln of the likelihood function (which is a Gaussian). 
        # The sigma in a normal Gaussian is just the error on our data points and the mu is 
        # just our model. (It may require doing out the algebra to understand the next line)
        return -0.5*np.sum( ( (fluxes.compressed() - model_fluxes.compressed() ) / fluxerrs.compressed() )**2 + np.log( 2 * np.pi * fluxerrs.compressed()**2 ) )



    def ln_prior(parameters):
        # Define the priors. Since the scaling factor covers a lot of ground I used a
        # Jefferies prior.

        deltaT, deltaG, Scale = 5000-800, 6-0.5, 1e-19/1e-25
        
        Teff, logg, scale = parameters

        if Teff != 0 and logg != 0:
            return -np.log(deltaT) - np.log(deltaG) - np.log(scale * np.log(Scale))
        else:
            return -np.inf
    
    

    # Now you're ready to find the full ln probability function, which is just the sum of the 
    #  two previous functions with a check to make sure we don't have a -inf
    def ln_probability(parameters, fluxes, fluxerrs, t, binary):

        # Get the parameters
        Temp, logg, scale = parameters

        # Interpolate to the resolution of the model parameter space
        Temp2, logg2 = do_kdtree(combined_x_y_arrays, (Temp, logg))
        parameters2 = [Temp2, logg2, scale]
   
        # Call the prior function, check that it doesn't return -inf then create the log-
        #  probability function
        priors = ln_prior(parameters2)
        
        if not np.isfinite(priors):
            return -np.inf
        else:
            return priors + ln_likelihood(parameters2, fluxes, fluxerrs, t, binary)

    
    # Almost there! Now we must initialize our walkers. Remember that emcee uses a bunch of 
    # walkers, and we define their starting distribution. 
    n_dim, n_walkers, n_steps = 3, 100, 4000

    Temp_rand = []
    logg_rand = []
    min_val, max_val = 1e-25, 1e-19
    scale_rand = []
    for i in range(n_walkers):
        Temp_rand.append( random.choice(t['Temp']) )
        logg_rand.append( random.choice(t['Logg']) )
        scale_rand.append(10 ** random.uniform(np.log10(min_val), np.log10(max_val)))
    

    # positions should be a list of N-dimensional arrays where N is the number of parameters 
    # you have. The length of the list should match n_walkers.
    positions = []
    for param in range(n_walkers):
        positions.append(np.array([Temp_rand[param], logg_rand[param], scale_rand[param]]))
        
    #Finally, you're ready to set up and run the emcee sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_probability, args=(Fluxes, FluxesUncert, t, binary) )
    sampler.run_mcmc(positions, n_steps)

    # You may notice that the walkers took a few steps to find the local probability well.
    #  Before we calculate numbers and uncertainties, we want to remove that burn in.
    #burn = float(input('Enter the burn_in length: '))
    burn = 2000 

    # Now apply the burn by cutting those steps out of the chain. 
    chain_burnt = sampler.chain[:, burn:, :] 
    # Also flatten the chain to just a list of samples
    samples = chain_burnt.reshape((-1, n_dim))
    
    # So that's all well and good, but what are my best-fit parameter values and uncertainties?

    # This rather 'pythonic' line came straight from the emcee documentation. It sets up a 
    #  function with lambda and applies that functions to each percentile to get the best-
    #  fit values.
    T_mcmc, g_mcmc, scale_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples,[16, 50, 84], axis=0)))
    print T_mcmc, g_mcmc, scale_mcmc

    # And you're done! The above variables print out the parameter values and asymmetric errors.
    
    f = open("%s_TEST2.csv"%obj, "w")
    f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%('SDSS_OBJID', 'TEMP', 'TEMP+', 'TEMP-', 'LOGG', 'LOGG+', 'LOGG-', 'SCALE', 'SCALE+', 'SCALE-'))
    f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(obj, T_mcmc[0], T_mcmc[1], T_mcmc[2], g_mcmc[0], g_mcmc[1], g_mcmc[2], scale_mcmc[0], scale_mcmc[1], scale_mcmc[2]))
    f.close()
    
    return 0


################################

# Setup a list of processes that we want to run
# Limit it to run multiple process at any given time

data = Table.read('Catalogs5/LaTeMoVeRS_v0_9_2.hdf5') # open an HDF5 file

# Get what you want
objs   = data['SDSS_OBJID']
imags  = data['IMAG']
ierrs  = data['IMAG_ERR']
zmags  = data['ZMAG']
zerrs  = data['ZMAG_ERR']
jmags  = data['JMAG']
jerrs  = data['JMAG_ERR']
hmags  = data['HMAG']
herrs  = data['HMAG_ERR']
kmags  = data['KMAG']
kerrs  = data['KMAG_ERR']
w1mags = data['W1MPRO']
w1errs = data['W1SIGMPRO']

# Zip it all up to multiprocess it
ARR = zip(objs, imags, ierrs, zmags, zerrs, jmags, jerrs, hmags, herrs, kmags, kerrs, w1mags, w1errs)

# Multiprocessing part
cpus = 16 # Number of cores
pool = mp.Pool(processes = cpus)
pool.map(SED_Fitter, ARR)
pool.close()

    
