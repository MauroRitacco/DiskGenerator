import numpy as np
import os
from casatools import table
from astropy.io import fits

def calculate_pixel_size(ms_path):
    tb = table()
    
    # Get the longest projected baseline in meters (u, v)
    tb.open(ms_path)
    uvw = tb.getcol('UVW')
    tb.close()
    longest_baseline_meters = np.max(np.sqrt(uvw[0]**2 + uvw[1]**2))
    
    # Get the reference frequency and calculate wavelength
    tb.open(ms_path + '/SPECTRAL_WINDOW')
    frequency = tb.getcol('CHAN_FREQ')[0, 0]
    tb.close()
    
    wavelength = 299792458.0 / frequency
    longest_baseline_lambda = longest_baseline_meters / wavelength
    
    # cell(arcsec) = (180 * 3600 / pi) * (1 / (superresolution * n * maxProjBaseline))
    return (180.0 * 3600.0 / np.pi) * (1.0 / (1.5 * longest_baseline_lambda))

def convert_jybeam_to_jypixel(fits_path):
    """Convert a FITS image from Jy/beam to Jy/pixel."""
    data, hdr = fits.getdata(fits_path, header=True)
    # Unit conversion: Jy/beam -> Jy/pixel
    factor = (4 * np.log(2) * hdr['CDELT1']**2) / (np.pi * hdr['BMAJ'] * hdr['BMIN'])
    data = data * factor
    if 'BUNIT' in hdr:
        hdr['BUNIT'] = 'Jy/pixel'
    fits.writeto(fits_path, data, hdr, overwrite=True)

def smoothimage(reconstruction_path, ground_truth_path, ms_path):
    recon_hdr = fits.getheader(reconstruction_path)
    
    BMAJ=str(recon_hdr['BMAJ']) + 'deg'
    BMIN=str(recon_hdr['BMIN']) + 'deg'
    PA=str(recon_hdr.get('BPA', 0))+ 'deg'

    gt_dir = os.path.dirname(ground_truth_path)
    recon_name = os.path.basename(reconstruction_path)
    
    # Delete output file if it exists to avoid CASA collision issues
    out_file = os.path.join(gt_dir, recon_name.replace('.fits', '_smoothed.fits'))
    if os.path.exists(out_file):
        try:
            import shutil
            if os.path.isdir(out_file):
                shutil.rmtree(out_file)
            else:
                os.remove(out_file)
        except Exception:
            pass

    # imsmooth outputs a CASA image directory. We write to a temporary .image first.
    temp_casa_image = os.path.join(gt_dir, recon_name.replace('.fits', '_temp_smoothed.image'))
    if os.path.exists(temp_casa_image):
        import shutil
        shutil.rmtree(temp_casa_image, ignore_errors=True)

    from casatasks import imsmooth
    imsmooth(
    imagename=ground_truth_path, # Smooth the ground truth (already has valid WCS)
    kernel="gaussian",
    beam={"major": BMAJ, "minor": BMIN, "pa": PA},
    targetres=True,
    outfile=temp_casa_image,
    overwrite=True,
    )

    from casatasks import exportfits
    exportfits(
        imagename=temp_casa_image,
        fitsimage=out_file,
        overwrite=True
    )
    # exportfits sometimes drops the beam keywords. We inject them explicitly 
    # so convert_jybeam_to_jypixel can perform the Jy/beam -> Jy/pixel scaling.
    out_data, out_hdr = fits.getdata(out_file, header=True)
    out_hdr['BMAJ'] = recon_hdr['BMAJ']
    out_hdr['BMIN'] = recon_hdr['BMIN']
    out_hdr['BPA'] = recon_hdr.get('BPA', 0)
    if out_hdr['BUNIT'] != 'Jy/beam':
        out_hdr['BUNIT'] = 'Jy/pixel'

    fits.writeto(out_file, out_data, out_hdr, overwrite=True)

    
    # Cleanup temporary files
    if os.path.exists(temp_casa_image):
        import shutil
        shutil.rmtree(temp_casa_image, ignore_errors=True)
    
def formatfits(fits_path, ms_path):
    imsize = 64 
    data = fits.getdata(fits_path)

    hdr = fits.Header()
    hdr['SIMPLE'] = True
    hdr['BITPIX'] = -32
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = imsize
    hdr['NAXIS2'] = imsize
    hdr['BUNIT'] = 'Jy/pixel'

    # 2. Convertir cell a grados para el estándar WCS usando el ms_path
    cell_deg = calculate_pixel_size(ms_path) / 3600.0

    # CRUCIAL PARA CARTA: Mismo valor absoluto en ambos ejes
    hdr['CDELT1'] = -cell_deg  # Eje X (negativo por convención de RA)
    hdr['CDELT2'] = cell_deg   # Eje Y (DEC)

    # Puntos de referencia simétricos en el centro de la grilla
    hdr['CRPIX1'] = (imsize / 2) + 0.5
    hdr['CRPIX2'] = (imsize / 2) + 0.5

    # Guardar el FITS limpio
    fits.writeto("salida_cuadrada.fits", data, hdr, overwrite=True)
