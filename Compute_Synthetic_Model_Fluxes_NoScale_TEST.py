#!/usr/bin/env python
import numpy as np
import sys, os, os.path, time
import glob
import astroroutines3 as AS
from astropy.table import Table

# Start timing counter to see how fast it runs
start = time.time()

# Define constants
light_speed = 2.99792458e5  # speed of light (km/s)

# Get the filter response curves
jcurve, hcurve, kcurve = AS.transCurves('2mass', o='ang')
ucurve, gcurve, rcurve, icurve, zcurve = AS.transCurves('sdss_doi', norm='normalize')
w1curve, w2curve, w3curve, w4curve = AS.transCurves('wise', o='ang')

#DataFile1 = open('Photometric_BTSETTL_Models2.txt', 'w')
#DataFile0 = open('Photometric_Models2.txt', 'w')

FileList = []
Temps = []
Loggs = []
Metals = []
Alphas = []
Umags = []
Gmags = []
Rmags = []
Imags = []
Zmags = []
Jmags = []
Hmags = []
Kmags = []
W1mags = []
W2mags = []
W3mags = []
W4mags = []
Luminosities = []

#####################
# Check each Pheonix model
path = '/mnt/Resources/immortal/Model_Photospheres/BT-Settl/'
#path = '/mnt/Resources/immortal/Model_Photospheres/BT-Settl_M-0.0a+0.0/'

Tstart, Tend = 800, 6000
#Tstart, Tend = 1000, 1050

for infile in glob.glob( os.path.join(path, '*.spec.7') ):
    print(infile)

    #Teff = float(infile.split('/')[3].split('-')[0].lstrip('lte'))*100

    #print infile
    newline = infile.split('/')[6]
    #print newline
    newline2 = newline.lstrip('lte').rstrip('.BT-Settl.spec.7')
    #print newline2
    Teff = float(newline2[:-13])*100
    Logg = float(newline2[-12:-9])
    Metal = float(newline2[-9:-5])
    Alpha = float(newline2[-4:])
        
    # Only look at models that have a reasonable temperature
    if Teff < Tstart or Teff > Tend: continue

    # Read the file
    #Model_wave, Model_Flux = np.loadtxt(infile, usecols=(0,1), unpack=True)
    Model_wave, Model_Flux, Model_BB_Flux = np.loadtxt(infile, usecols=(0,1,2), unpack=True)
    # Sort by wavelength
    order = np.argsort(Model_wave)
    modelWaves = Model_wave[order]
    ModelFlux = Model_Flux[order]
    ModelBBFlux = Model_BB_Flux[order]
    newModelFlux = 10**(ModelFlux - 8.)  # ergs/s/cm^2/Ang
    newModelBBFlux = 10**(ModelBBFlux - 8.)  # ergs/s/cm^2/Ang

    # Interpolate the flux at bandpass wavelengths
    interU = np.interp(modelWaves, ucurve[0], ucurve[1])
    Usum = np.trapz(interU * newModelFlux, x=Model_wave) / np.trapz(ucurve[1], x=ucurve[0])
    interG = np.interp(modelWaves, gcurve[0], gcurve[1])
    Gsum = np.trapz(interG * newModelFlux, x=Model_wave) / np.trapz(gcurve[1], x=gcurve[0])
    interR = np.interp(modelWaves, rcurve[0], rcurve[1])
    Rsum = np.trapz(interR * newModelFlux, x=Model_wave) / np.trapz(rcurve[1], x=rcurve[0])
    interI = np.interp(modelWaves, icurve[0], icurve[1])
    Isum = np.trapz(interI * newModelFlux, x=Model_wave) / np.trapz(icurve[1], x=icurve[0])
    interZ = np.interp(modelWaves, zcurve[0], zcurve[1])
    Zsum = np.trapz(interZ * newModelFlux, x=Model_wave) / np.trapz(zcurve[1], x=zcurve[0])
    
    interJ = np.interp(modelWaves, jcurve[0], jcurve[1])
    Jsum = np.trapz(interJ * newModelFlux, x=Model_wave) / np.trapz(jcurve[1], x=jcurve[0])
    interH = np.interp(modelWaves, hcurve[0], hcurve[1])
    Hsum = np.trapz(interH * newModelFlux, x=Model_wave) / np.trapz(hcurve[1], x=hcurve[0])
    interK = np.interp(modelWaves, kcurve[0], kcurve[1])
    Ksum = np.trapz(interK * newModelFlux, x=Model_wave) / np.trapz(kcurve[1], x=kcurve[0])
    
    interW1 = np.interp(modelWaves, w1curve[0], w1curve[1])
    W1sum = np.trapz(interW1 * newModelFlux, x=Model_wave) / np.trapz(w1curve[1], x=w1curve[0])
    interW2 = np.interp(modelWaves, w2curve[0], w2curve[1])
    W2sum = np.trapz(interW2 * newModelFlux, x=Model_wave) / np.trapz(w2curve[1], x=w2curve[0])
    interW3 = np.interp(modelWaves, w3curve[0], w3curve[1])
    W3sum = np.trapz(interW3 * newModelFlux, x=Model_wave) / np.trapz(w3curve[1], x=w3curve[0])
    interW4 = np.interp(modelWaves, w4curve[0], w4curve[1])
    W4sum = np.trapz(interW4 * newModelFlux, x=Model_wave) / np.trapz(w4curve[1], x=w4curve[0])
     
    #NormS = np.array([Usum,Gsum,Rsum,Isum,Zsum,Jsum,Hsum,Ksum,W1sum,W2sum,W3sum,W4sum])

    Lstar = np.trapz(newModelBBFlux, x=modelWaves)

    FileList.append(infile.split('/')[6])
    Temps.append(Teff)
    Loggs.append(Logg)
    Metals.append(Metal)
    Alphas.append(Alpha)
    Umags.append(Usum)
    Gmags.append(Gsum)
    Rmags.append(Rsum)
    Imags.append(Isum)
    Zmags.append(Zsum)
    Jmags.append(Jsum)
    Hmags.append(Hsum)
    Kmags.append(Ksum)
    W1mags.append(W1sum)
    W2mags.append(W2sum)
    W3mags.append(W3sum)
    W4mags.append(W4sum)
    Luminosities.append(Lstar)

    print(infile, 'DONE')

#DataFile0.close()
#DataFile1.close()

t = Table([FileList, Temps, Loggs, Metals, Alphas, Umags, Gmags, Rmags, Imags, Zmags, Jmags, Hmags, Kmags, W1mags, W2mags, W3mags, W4mags, Luminosities],
          names=('File', 'Temp', 'Logg', 'Metal', 'Alpha', 'Usum', 'Gsum', 'Rsum', 'Isum', 'Zsum', 'Jsum', 'Hsum', 'Ksum', 'W1sum', 'W2sum', 'W3sum', 'W4sum', 'Lstar') )
t.write('Photometric_BTSETTL_Models_NoScale_AllParams.fits')
    
# Let me know it's done
print("Done in %2.2f seconds"% (time.time() - start))
print("Done in %2.2f minutes"% ((time.time() - start)/60.0))
print("Done in %2.2f hours"% ((time.time() - start)/3600.0))
