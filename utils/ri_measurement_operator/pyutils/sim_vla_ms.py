# scripts to create Fourier sampling patterns in .mat files containing the fields:
# `u`,`v`,`w` (all in units of meter), `frequency` (MHz), `nominal_pixelsize` (arcsec)
# Author: A. Dabbech
import warnings
import matplotlib.cbook
try:
    from matplotlib import MatplotlibDeprecationWarning
except ImportError:
    pass 

if not hasattr(matplotlib.cbook, "MatplotlibDeprecationWarning"):
    matplotlib.cbook.MatplotlibDeprecationWarning = DeprecationWarning


import simms
from casatools import table
from casatasks import concat
import numpy as np
import scipy.io as sio
import os
import argparse
import math
import timeit
import matplotlib.pyplot as plt

# constants
c = 299792458  # Speed of light

# set dirs
# capture the folder where this script lives to find antenna config files
script_dir = os.path.dirname(os.path.abspath(__file__))
## user-input
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--npatterns', default=1)
parser.add_argument('--start_id', default=0, help="Starting ID for the generated files")
# custom output directory (defaults to current working directory)
parser.add_argument('--outdir', default=os.getcwd(), help="Output directory")
args = parser.parse_args()

# use the specified outdir for all data generation
mydir = args.outdir
almasimsdir = mydir + "/alma_sims/"
os.system("mkdir %s" % almasimsdir)
msdir = almasimsdir + "ms/"
os.system("mkdir %s" % msdir)
uvdir = almasimsdir + "uvw/"
os.system("mkdir %s" % uvdir)
pngdir = almasimsdir + "png/"
os.system("mkdir %s" % pngdir)

def main():
    # number of sampling patterns & MS tables to generate
    print('############# User-input:')
    npatterns = int(args.npatterns)
    print('Number of requested Fourier sampling patterns: %s' % npatterns)
    start_id = int(args.start_id)
    # generate sampling patterns aka uvw-coverages & related info
    for i in range(start_id, start_id + npatterns):
        start = timeit.default_timer()
        print('############# Fourier sampling pattern id: %s' % i)

        while True:
            try:
                print('##### observation settings:')
                ## number of ALMA antennas
                na = 40
                ## pointing direction: Right Ascension
                ra = np.random.uniform() * 23  # [0h,23h]
                ra_h = int(ra)
                ra_min = abs(int((ra - ra_h) * 59))
                msra = str(ra_h) + "h" + str(ra_min) + "m0s"
                print("info: (param) RA: %s" % msra)
                ## pointing direction: Declination
                dec_deg = int(np.random.uniform() * -80 + 20)  #  [-60,+20]
                dec_min = int(np.random.uniform() * 59)
                msdec = str(dec_deg) + "d" + str(dec_min) + "m0s"
                print("info: (param) DEC: %s" % msdec)

               
                # DSHARP-like conditions (sparse to full quality, for R2D2 training):
                dt5 = np.random.uniform(0.03, 0.25)    # C40-5 (compact): ~2-15 min on-source (DSHARP: ~12 min)
                dt8 = np.random.uniform(0.15, 1.17)    # C40-8/9 (extended): ~9-70 min on-source (DSHARP: 35-70 min)
                dt_step = 6                             # ALMA standard dump time (seconds)
            
                print("info: (param) obs. time with config C40-5 of freqs: %.2f h" % dt5)
                print("info: (param) obs. time with config C40-8 of freqs: %.2f h" % dt8)
                print("info: (param) time step: %.2f sec" % dt_step)
                ## freq specs (DSHARP ALMA Band 6 continuum)
                freq0 = 243e9  # ~243 GHz, Band 6 center
                nfreqs = 1
                freq_vect = np.array([freq0])
                msdfreq = "31.25MHz"  # channel width (1.875 GHz / 128 channels)

                msfreq0 = "%sMHz" % (freq0 / 1e6)

                print("info: (param) number of freqs: %s" % nfreqs)

                print('##### Augmentation:')
                ## rotation angle of the uv-coverage/psf
                rot_theta = np.random.uniform(0, 360)  # to switch off, set to 0
                print("info: (param) rotation angle of the sampling pattern: %.2f deg" % rot_theta)
                flag_percentage_max = 0.2  # upper-bound on the flagging percentage

                print('##### Imaging settings (required to generate imaging weights):')
                # briggs weighting: robust param
                briggs = 2  # np.random.uniform(-1, 1)  # B
                print("info: (param) briggs weigting: %.2f" % briggs)

                ## MS filenames
                # config C40-5
                ext5 = 'id_' + str(i) + '_dec_' + str(dec_deg) + '_dt5_' + str("%.2f" % dt5)
                mymsfile_5 = '%salmac40-5_%s.MS' % (msdir, ext5)
                # config C40-8
                extc = 'id_' + str(i) + '_dec_' + str(dec_deg) + '_dt8_' + str("%.2f" % dt8)
                mymsfile_8 = '%salmac40-8_%s.MS' % (msdir, extc)
                # ms file
                ext = f'{i:04d}' # + '_dt_' + str("%.2f" % (dta + dtc)) + "_freqratio_" + str(
                    # "%.2f" % freq_ratio) + '_nfreq_' + str(nfreqs) + '_rotation_' + str("%.2f" % rot_theta)
                mymsfile = '%sdisk_%s.MS' % (msdir, ext)

                print('##### Create empty MS: start')
                # delete old ms just in case
                try:
                    os.system('rm -rf %s' % mymsfile)
                except:
                    pass

                # --------------------------------------
                # Step 1: Generate both MSs
                # --------------------------------------
                print("CASA:start ----------------------------")
        # VLA--------------------------------------------------------
        # print('##### Create empty MS .. A config')
        # A config
        # Use script_dir to locate the antenna configuration regardless of where the script is run from
        # simms.create_empty_ms(msname=mymsfile_a, tel='vla', pos="%s/observatories/vlaa.itrf.txt" % script_dir,
        #                       pos_type='ascii', coords="itrf",
        #                       synthesis=dta, dtime=dt_step, dfreq=msdfreq, freq0=msfreq0, nchan=str(nfreqs),
        #                       ra=msra, dec=msdec, scan_length=[dta + 0.01], scan_lag=0,
        #                       stokes="XX YY", setlimits=True, optimise_start=True)
        # # C config
        # print('##### Create empty MS .. C config')
        # simms.create_empty_ms(msname=mymsfile_c, tel='vla', pos="%s/observatories/vlaa.itrf.txt" % script_dir,
        #                       pos_type='ascii', coords="itrf",
        #                       synthesis=dtc, dtime=dt_step, dfreq=msdfreq, freq0=msfreq0, nchan=str(nfreqs),
        #                       ra=msra, dec=msdec, scan_length=[dtc + 0.01], scan_lag=0,
        #                       stokes="XX YY", setlimits=True, optimise_start=True)

                # ALMA --------------------------------------------------------
                print('##### Create empty MS .. C40-5')
                simms.create_empty_ms(msname=mymsfile_5, tel='ALMA', pos="%s/observatories/ALMA.C40-5.LOC.cfg" % script_dir,
                                      pos_type='ascii', coords="enu",
                                      synthesis=dt5, dtime=dt_step, dfreq=msdfreq, freq0=msfreq0, nchan=str(nfreqs),
                                      ra=msra, dec=msdec, scan_length=[dt5 + 0.01], scan_lag=0,
                                      stokes="XX YY", setlimits=True, optimise_start=True)

                print('##### Create empty MS .. C40-8')
                simms.create_empty_ms(msname=mymsfile_8, tel='ALMA', pos="%s/observatories/ALMA.C40-8.LOC.cfg" % script_dir,
                                      pos_type='ascii', coords="enu",
                                      synthesis=dt8, dtime=dt_step, dfreq=msdfreq, freq0=msfreq0, nchan=str(nfreqs),
                                      ra=msra, dec=msdec, scan_length=[dt8 + 0.01], scan_lag=0,
                                      stokes="XX YY", setlimits=True, optimise_start=True)
                
                print('##### Concat  MSs ..')
                concat(vis=[mymsfile_5, mymsfile_8], concatvis=mymsfile)
                os.system('rm -rf %s' % mymsfile_5)  # delete C40-5 MS
                os.system('rm -rf %s' % mymsfile_8)  # delete C40-8 MS
                print("CASA:end ----------------------------")
                # --------------------------------------
                # Step 2: Apply (random) flags & extract (final) uvw
                # --------------------------------------
                tb = table()
                tb.open(mymsfile, nomodify=False)
                print("MS table columns:", *(tb.colnames()))
                print('##### Random flagging & data extraction: start')

                # check number of scans
                scans = tb.getcol('SCAN_NUMBER')
                nscans = len(np.unique(scans))
                # get number of meas. per freq.
                nmeas = len(scans)
                print("info: number of scans %s" % nscans)

                # get uvw col
                uvw = tb.getcol('UVW')
                # apply rotation to the Fourier sampling pattern (aka uv-coverage)
                uvw_rot = np.zeros(uvw.shape)
                uvw_rot[0,:] = math.cos(math.radians(rot_theta)) * uvw[0,:] - math.sin(math.radians(rot_theta)) * uvw[1,:]
                uvw_rot[1,:] = math.sin(math.radians(rot_theta)) * uvw[0,:] + math.cos(math.radians(rot_theta)) * uvw[1,:]
                uvw_rot[2,:] = uvw[2,:]
                # overwrite uvw col
                tb.putcol('UVW', uvw_rot)

                # get FLAG col (set to False everywhere)
                flag = tb.getcol('FLAG')
                ## init
                u, v, w = [], [], []
                ## apply random flagging at each frequency & get uvw
                ref_freq = (freq_vect[nfreqs - 1] + freq_vect[0]) / 2

                for ifreq in range(len(freq_vect)):
                    flag_percentage = np.random.uniform() * flag_percentage_max
                    print("info: (param) freq %s: flagging percentage %.4f " % (ifreq, 100 * flag_percentage))
                    if nmeas > 0:
                        flagged_rows = (np.random.choice(np.linspace(0, nmeas - 1, num=nmeas),
                                                         size=math.floor(nmeas * flag_percentage))).astype(int)
                        flag[:, ifreq, flagged_rows] = True
                    select_rows = (flag[0, ifreq, :] == False)

                    # freq ratio to be applied
                    ifreq_ratio = freq_vect[ifreq] / ref_freq
                    # get uvw after flagging (in meter)
                    u.extend(uvw_rot[0, select_rows] / ifreq_ratio)
                    v.extend(uvw_rot[1, select_rows] / ifreq_ratio)
                    w.extend(uvw_rot[2, select_rows] / ifreq_ratio)

                print("info: initial number  meas. %s" % (nfreqs * nmeas))
                print("info: number of meas. after flagging %s" % len(u))
                
                if len(u) == 0:
                    print("Warning: All data flagged or zero measurements. Retrying this disk...")
                    tb.close()
                    continue

                # overwrite FLAG col & update MS
                tb.putcol('FLAG', flag)  # update flag column in the MS
                tb.close()
                print('##### Random flagging & data extraction: end')

                # reshape vars
                u = np.reshape(np.array(u), (len(u), 1))
                v = np.reshape(np.array(v), (len(u), 1))
                w = np.reshape(np.array(w), (len(u), 1))

                # info needed for pixelsize
                wavelength = c / ref_freq
                maxProjBaseline = (np.sqrt(np.max(u ** 2 + v ** 2))).astype(float) / wavelength
                print('info: frequency: %f GHz ' % (ref_freq / 1e9))
                print("info: max. projected baseline in units of the wavelength %s " % maxProjBaseline)

                # ------------------------------
                # Step 3: Save uvw to .mat
                # ------------------------------
                # Set nominal pixelsize randomly in the interval [0.5/64, 4/64] arcsec
                nominal_pixelsize = np.random.uniform(1.5 / 64, 3.0 / 64)
                print("info: nominal pixelsize:  %f arcsec" % nominal_pixelsize)

                # uvw saved in units of meter
                sampling_pattern_uvw = {'u': u, 'v': v, 'w': w, 'frequency': ref_freq,
                                        'nominal_pixelsize': nominal_pixelsize}

                # final  mat file
                uvwmatfile = uvdir + 'uv_' + ext + '.mat'
                print("info: saving .mat file: %s" % uvwmatfile)
                sio.savemat(uvwmatfile, sampling_pattern_uvw)

                # additional plot of the uv-coverage (for info only)
                plt.figure()
                plt.scatter(u, v, color='red', s=0.01)
                plt.scatter(-u, -v, color='blue', s=0.01)
                plt.title("dt:")
                plt.savefig(pngdir + 'uv_' + ext + '.png')
                plt.close()
                print(
                    "info: mat file created successfully, which includes these fields:  `u`,`v`,`w` (all in units of meter), `frequency` (MHz), `nominal_pixelsize` (arcsec)")
                print('##### File saved.')
                break  # Successfully generated, break the retry loop
            except Exception as e:
                print(f"Error occurred: {e}. Retrying this disk...")
                continue

    ## delete tmp dirs
    # os.system("rm -rf %s" % mymsfile)

    stop = timeit.default_timer()

    print('Time to generate sampling patterns & files: %2.f sec ' % (stop - start))
    print('##### END.')

if __name__ == "__main__":
    main()
