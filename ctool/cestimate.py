import copy
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import medfilt, savgol_filter, find_peaks
from astropy.io import fits

# Original one
# def calc_cont(wave,flux, niter=5, boxsize=95, exclude=None, threshold=0.998, offset=0, spike_threshold=None):
#     flux_tmp=copy.deepcopy(flux)

#     #Remove negative spikes from consideration
#     if(spike_threshold is not None):
#         bad_pix, _ = find_peaks(-1*flux_tmp, prominence=spike_threshold)
#         flux_tmp[bad_pix] = np.nan

#     #Exclude regions    
#     if(exclude is not None):
#         for myexclude in exclude:  #Exclude regions from fitting
#             localbool=((wave>myexclude[0]) & (wave<myexclude[1]))
#             flux_tmp[localbool]=np.nan   
            
#     #Perform continuum determination        
#     cont = copy.deepcopy(flux_tmp)            
#     for ii in np.arange(niter):
#         smooth = medfilt(cont.astype(np.float32),boxsize)
#         csubs = np.where(smooth>cont*threshold)   
#         cont = np.interp(wave,wave[csubs],cont[csubs])

#     cont = savgol_filter(cont,boxsize*3,polyorder=2,mode='mirror')
    
#     #Apply offset
#     cont+=offset
    
#     return cont

# Slight modification for CSV 
def calc_cont(wave, flux, niter=5, boxsize=95, exclude=None, threshold=None, offset=0, spike_threshold=None):
    # Convert pandas Series to numpy arrays if needed
    # threshold_number default is 0.998 in the original function though
    wave_arr = wave.to_numpy() if hasattr(wave, 'to_numpy') else np.array(wave)
    flux_arr = flux.to_numpy() if hasattr(flux, 'to_numpy') else np.array(flux)
    
    flux_tmp = copy.deepcopy(flux_arr)
    
    # Remove negative spikes from consideration
    if spike_threshold is not None:
        bad_pix, _ = find_peaks(-1*flux_tmp, prominence=spike_threshold)
        flux_tmp[bad_pix] = np.nan
    
    # Exclude regions
    if exclude is not None:
        for myexclude in exclude:
            localbool = ((wave_arr > myexclude[0]) & (wave_arr < myexclude[1]))
            flux_tmp[localbool] = np.nan
    
    # Perform continuum determination
    cont = copy.deepcopy(flux_tmp)
    for ii in np.arange(niter):
        smooth = medfilt(cont.astype(np.float32), boxsize)
        csubs = np.where(smooth > cont * threshold)[0]  # Get the first element of the tuple
        cont = np.interp(wave_arr, wave_arr[csubs], cont[csubs])
    
    cont = savgol_filter(cont, boxsize*3, polyorder=2, mode='mirror')
    
    # Apply offset
    cont += offset
    
    return cont


# Assuming the columns are named 'wave', 'flux', 'error', and 'continuum'
# Create a new column for flux+continuum
def visualize (df, filename, threshold_number):

    df['flux_plus_continuum'] = df['flux'] + df['cont']
    
    flux = df['flux_plus_continuum']
    spectral_axis = df['wave']
    
    # Define the wavelength ranges for visualization
    visualization_wavelength_ranges = [
        (4.9, 27.5),
        (4.9, 5.5),
        (5.5, 8),
        (8, 12),
        (13, 15.5),
        (15.5, 18.5),
        (19, 22),
        (22, 27.5)
    ]
    
    # Create a figure with 8 subplots
    fig, axs = plt.subplots(8, figsize=(10, 15))
    
    # Loop through each wavelength range and plot the data
    for i, (start, end) in enumerate(visualization_wavelength_ranges):
        # Select the data within the current wavelength range
        mask = (spectral_axis >= start) & (spectral_axis <= end)
        w_subset = spectral_axis[mask]
        f_subset = flux[mask]
    
        # Calculate the continuum for this subset
        continuum = calc_cont(w_subset, f_subset, niter=5, boxsize=95, exclude=None,
                             threshold=threshold_number, offset=0, spike_threshold=None)
    
        # Plot the data and the continuum fit
        axs[i].plot(w_subset, f_subset, label='Original data', color='k')
        axs[i].plot(w_subset, continuum, label='Continuum fit', color='r')
        axs[i].set_yscale('log')
    
        # Only show legend for the first subplot
        if i == 0:
            axs[i].legend()
    
    # Layout so plots do not overlap
    fig.tight_layout()
    
    # Save the plot as a PDF
    plt.savefig(filename+'.pdf', bbox_inches='tight')
    
    plt.show()


# Example: How to use

# Read the CSV file
# df = pd.read_csv('/Users/mk/Desktop/JWST_JDISC/MIRI_data/JDISCS_v8.2/CSUB_v8.2/Oph_1_1d_v8.2_csub.csv')

# visualize (df, 'Oph1', 0.85)