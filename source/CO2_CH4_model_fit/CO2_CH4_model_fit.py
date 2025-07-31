# Functions to generate model fits to CO2 and CH4 data from CAMS EGG4 reanalysis.
# Models will be used to extrapolate CO2 and CH4 time series to the future times 
#   of TIRS scenes in Aux-Met.
# Models have linear and harmonic components.
# Models are developed by applying a separate fit to the zonal mean time series 
#   at each latitude, then fitting a smoothed spline curve to the variation of
#   each model coefficient across latitudes.

# The "model_dev_notebooks" contains additional info on model development,
#   including plots of actual and modelled CO2 and CH4 time series, and examples of 
#   alternative models considered.

import numpy as np
from scipy import interpolate
import xarray
import h5py
import datetime as dt


def CAMS_EGG4_ts_write(data_dir):
    """
    Read in 3-hourly CO2 and CH4 data for 2003-2020, write out daily time array
    and time series of daily CO2 and CH4 zonal mean for all latitudes.

    Parameters
    ----------
    data_dir : str
        Directory containing monthly CAMS EGG4 files.

    Returns
    -------
    None.
    
    """
    
    yrs = np.arange(2003,2021,1)
    mths = np.arange(1,13,1)
    
    t = np.array([], dtype='datetime64')
    
    file_ix = 0
    for yr in yrs:
        for mth in mths:
            fname = dt.datetime(yr,mth,1).strftime('%Y%m.nc')
            ds = xarray.open_dataset(data_dir+fname)
            t_daily = ds['time'].resample(time='1D').mean()
            CO2_daily = ds['tcco2'].resample(time='1D').mean()
            # Convert CH4 from ppb to ppm
            CH4_daily = (ds['tcch4'] / 1000).resample(time='1D').mean()
            
            if file_ix == 0:
                CO2_zonal_mean_ts = np.mean(CO2_daily, axis=2)
                CH4_zonal_mean_ts = np.mean(CH4_daily, axis=2)
            else:
                CO2_zonal_mean_ts = np.concatenate((CO2_zonal_mean_ts, np.mean(CO2_daily, axis=2)), axis=0)
                CH4_zonal_mean_ts = np.concatenate((CH4_zonal_mean_ts, np.mean(CH4_daily, axis=2)), axis=0)
            t = np.append(t, t_daily)
            
            file_ix += 1
            
            print('Finished reading year: '+str(yr)+', month: '+str(mth)+
                  ' at '+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
    np.save('CO2_zonal_mean_ts.npy', CO2_zonal_mean_ts)
    np.save('CH4_zonal_mean_ts.npy', CH4_zonal_mean_ts)
    np.save('t_daily_2003_2020.npy', t)
    

def generate_harmonics(t, nharmonics):
    """
    generate harmonic functions (pairs of sin, cos) for input time array
    The time array should contain fractional years.
    """
    harmonics = np.zeros((nharmonics*2,)+t.shape)
    tphase = t * 2 * np.pi
    for h in range(nharmonics):
        tmps = np.sin((h+1)*tphase)
        tmpc = np.cos((h+1)*tphase)
        harmonics[2*h,  :] = tmps
        harmonics[2*h+1,:] = tmpc
    return harmonics


def compute_model_coefs(t, v, npoly, nharmonics, msk=None):
    """
    create a polynomal/harmonic model of time
    
    Parameters
    ----------
    t : ndarray
        array with times, as fractional years, shaped (t,)
    v : ndarray
        array with data values, shaped (t,n), where the n is the latitude.
        data is assumed to be the zonal average of some data field
    npoly : int
        order of polynomial to fit; 1 is linear.
    nharmonics : int
        number of harmonics. The frequencies are relative to the annual cycle.
        Here, 1 means fit sine/cosine to annual cycle (1 cycle per year), 2
        means fit 1 and 2 cycles per year.
        
    Returns
    -------
    poly_coef : ndarray
        array with shape (n, npoly), with the polynomial coef per latitude.
        ordered with highest power first (same as np.polyfit)
    harmonic_coef : ndarray
        array with shape (n, nharmonic*2), with harmonic fit coefs.
        ordered with lowest frequency first, sine then cosine.
    """
    
    if msk is None:
        msk = np.ones(t.shape, bool)

    tm = t[msk]
    vm = v[msk]

    nlats = vm.shape[1]
    
    harmonics = generate_harmonics(tm, nharmonics)

    poly_coefs = np.zeros((nlats, npoly+1))
    harmonic_coefs = np.zeros((nlats, nharmonics*2))

    for i in range(nlats):
        poly_coef = np.polyfit(tm, vm[:,i], npoly)
        poly_model = np.polyval(poly_coef, tm)
        poly_coefs[i,:] = poly_coef

        # use dot product to get coefficient: this is projecting
        # the residual onto the harmonic basis functions.
        # this only works on residuals that have zero mean, so subtract
        # the linear model first.
        residual = vm[:,i] - poly_model
        residual = residual / (0.5*tm.shape[0])
        harmonic_coefs[i,:] = harmonics @ residual
    
    return poly_coefs, harmonic_coefs


def smooth_model_coefs(lm_coefs, hm_coefs, lats, smoothing_factor=1):
    """
    Smooth the shapes of the linear and harmonic model coefficients across 
    latitudes, using scipy.interpolate.splrep

    Parameters
    ----------
    lm_coefs : ndarray
        Array with shape (n, npoly), with the polynomial coef per latitude.
        ordered with highest power first (same as np.polyfit).
    hm_coefs : ndarray
        Array with shape (n, nharmonic*2), with harmonic fit coefs.
    lats : ndarray
        Latitudes from the CAMS EGG4 data.
    smoothing_factor : int, optional
        Smoothing factor passed to splrep to smooth the B-spline curve.
        The default is 1.

    Returns
    -------
    lm_knots : ndarray
        The knots of the B-spline fit for each coefficient of the linear model
        across latitudes.
    lm_bcoefs : ndarray
        The B-spline coefficients of the B-spline fit for each coefficient of 
        the linear model across latitudes.
    hm_knots : ndarray
        The knots of the B-spline fit for each coefficient of the harmonic model
        across latitudes.
    hm_bcoefs : ndarray
        The B-spline coefficients of the B-spline fit for each coefficient of 
        the harmonic model across latitudes.

    """

    # splrep assumes x is increasing, so we have to flip the lats array.
    x = lats[::-1]

    lm_knots = []
    lm_bcoefs = []

    for n in range(lm_coefs.shape[1]):
        y, scale, offset = scale_y(lm_coefs[::-1,n])
        tck = interpolate.splrep(x, y, s=smoothing_factor)
        # Note: we can actually fold in the scale/offsets at this stage.
        # since the spline is a linear combination of the B-spline coefficients,
        # we can apply the scale/offset to those coefficients right here.
        # this simplifies the downstream application step
        lm_knots.append(tck[0])
        lm_bcoefs.append(tck[1] * scale + offset)
    
    hm_knots = []
    hm_bcoefs = []
    
    for n in range(hm_coefs.shape[1]):
        y, scale, offset = scale_y(hm_coefs[::-1,n])
        tck = interpolate.splrep(x, y, s=smoothing_factor)
        hm_knots.append(tck[0])
        hm_bcoefs.append(tck[1] * scale + offset)
    
    return lm_knots, lm_bcoefs, hm_knots, hm_bcoefs


def scale_y(y):
    offset = y.mean()
    y = y - offset
    scale = y.std()
    y = y / scale
    return y, scale, offset


def write_spline_file(var, lm_knots, lm_bcoefs, hm_knots, hm_bcoefs,
                      npoly, nharmonics, spline_degree=3, start_year=2003):
    """
    Write HDF file with parameters of spline fit across latitudes for a given
    variable.

    Parameters
    ----------
    var : str
        "CO2" or "CH4".
    lm_knots : ndarray
        The knots of the B-spline fit for each coefficient of the linear model
        across latitudes.
    lm_bcoefs : ndarray
        The B-spline coefficients of the B-spline fit for each coefficient of 
        the linear model across latitudes.
    hm_knots : ndarray
        The knots of the B-spline fit for each coefficient of the harmonic model
        across latitudes.
    hm_bcoefs : ndarray
        The B-spline coefficients of the B-spline fit for each coefficient of 
        the harmonic model across latitudes.
    npoly : int
        Order of polynomial fit; 1 is linear.
    nharmonics : int
        The number of harmonics used for the harmonic model fit.
    spline_degree : int
        The degree of the spline fit. The default is 3 (cubic).
    start_yr : int, optional
        Start year of the curve fit. The default is 2003.

    Returns
    -------
    None.

    """
    
    with h5py.File(var+'_model_spline_tck.h5', 'w') as h:
        h['start_year'] = start_year
        h['poly_degree'] = npoly
        h['harmonic_degree'] = nharmonics
        h['spline_degree'] = spline_degree
        
        for n in range(npoly*2):
            h['linear_knots'+str(n)] = lm_knots[n]
            h['linear_bcoef'+str(n)] = lm_bcoefs[n]
        for n in range(nharmonics*2):
            h['harmonic_knots'+str(n)] = hm_knots[n]
            h['harmonic_bcoef'+str(n)] = hm_bcoefs[n]


if __name__ == "__main__":
    # # Write time, CO2, and CO4 daily arrays to disk
    # CAMS_EGG4_dir = '/data/users/k/CAMS_EGG4_CO2_CH4/'
    # CAMS_EGG4_ts_write(CAMS_EGG4_dir)
    
    # Load time, CO2, and CO4 daily arrays if they have already been written
    t = np.load('t_daily_2003_2020.npy')
    CO2_zonal_mean_ts = np.load('CO2_zonal_mean_ts.npy')
    CH4_zonal_mean_ts = np.load('CH4_zonal_mean_ts.npy')    
    
    lats = np.linspace(90,-90,241)
    nyears = (t[-1]-t[0])/np.timedelta64(1,'D')/365.25
    nyears = np.round(nyears)
    tyears = np.linspace(0, nyears, t.shape[0])    
    
    # Compute CO2 model for all latitudes
    npoly_CO2 = 1
    nharmonics_CO2 = 3
    CO2_lm_coefs, CO2_hm_coefs = compute_model_coefs(tyears,
                                                     CO2_zonal_mean_ts,
                                                     npoly_CO2,
                                                     nharmonics_CO2)
    # Smooth coefficients of CO2 model
    CO2_lm_knots, CO2_lm_bcoefs, CO2_hm_knots, CO2_hm_bcoefs = \
        smooth_model_coefs(CO2_lm_coefs, CO2_hm_coefs, lats)
    # Write CO2 model spline fit to file
    write_spline_file('CO2', CO2_lm_knots, CO2_lm_bcoefs, CO2_hm_knots, CO2_hm_bcoefs, 
                      npoly_CO2, nharmonics_CO2)

    # Compute CH4 model for all latitudes, using 1 harmonic instead of 3
    npoly_CH4 = 1
    nharmonics_CH4 = 1
    CH4_lm_coefs, CH4_hm_coefs = compute_model_coefs(tyears,
                                                     CH4_zonal_mean_ts,
                                                     npoly_CH4,
                                                     nharmonics_CH4)
    # Smooth coefficients of CH4 model
    CH4_lm_knots, CH4_lm_bcoefs, CH4_hm_knots, CH4_hm_bcoefs = \
        smooth_model_coefs(CH4_lm_coefs, CH4_hm_coefs, lats)
    # Write CH4 model spline fit to file
    write_spline_file('CH4', CH4_lm_knots, CH4_lm_bcoefs, CH4_hm_knots, CH4_hm_bcoefs, 
                      npoly_CH4, nharmonics_CH4)
