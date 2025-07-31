import numpy as np
import datetime as dt
import h5py
from scipy import interpolate

from PREFIRE_tools.utils.aux import TIRS_L1B_time_array_to_dt


def CO2_CH4_model_load(model_fpath):
    """
    Load the model data, stored in h5, into a python dictionary.
    This reorganizes the data back into a form that can be directly passed to
    scipy.interpolate.splev
    
    Parameters
    ----------
    model_fpath : str
        Path to model data h5 file.

    Returns
    -------
    model_data : dict
        Dictionary with smoothed model coefficients (linear and harmonic) and
        ancillary information needed to evaluate the model.

    """

    model_data = {}
    with h5py.File(model_fpath, 'r') as h:
        # empty tuple as argument is apparently 'correct' way to get a scalar value.
        model_data['start_year'] = h['start_year'][()]
        model_data['poly_degree'] = h['poly_degree'][()]
        model_data['harmonic_degree'] = h['harmonic_degree'][()]
        for n in range(model_data['poly_degree']*2):
            model_data['linear_model_tck'+str(n)] = (
                h['linear_knots'+str(n)][:],
                h['linear_bcoef'+str(n)][:],
                h['spline_degree'][()]
                )
        for n in range(model_data['harmonic_degree']*2):
            model_data['harmonic_model_tck'+str(n)] = (
                h['harmonic_knots'+str(n)][:],
                h['harmonic_bcoef'+str(n)][:],
                h['spline_degree'][()]
                )

    return model_data


def CO2_CH4_model_eval(model_data, TIRS_lat, TIRS_time_array):
    """
    Evaluate model to get CO2 / CH4 value at a given TIRS scene. 
    
    Parameters
    ----------
    model_data : dict
        Dictionary with smoothed model coefficients (linear and harmonic) and
        ancillary information needed to evaluate the model.
    TIRS_lat : float
        Latitude of TIRS scene.
    TIRS_time_array : ndarray
        TIRS observation time represented as as an array with shape (7,). Array
        entries are: year, month, day, hour, minute, second, microsecond.

    Returns
    -------
    model : float
        The modelled value of CO2 or CH4 at the TIRS scene latitude and time.

    """
    
    # Compute linear temporal model (scaled data)
    c0 = interpolate.splev(TIRS_lat, model_data['linear_model_tck0'], der=0)
    c1 = interpolate.splev(TIRS_lat, model_data['linear_model_tck1'], der=0)
    
    # Convert time to format of decimal years elapsed since Jan. 1 of start year
    t = np.datetime64(TIRS_L1B_time_array_to_dt(TIRS_time_array))
    start_t = np.datetime64(dt.datetime(model_data['start_year'], 1, 1))
    t_decyrs = (t-start_t)/np.timedelta64(1,'D')/365.25
    linear_model = c0 * t_decyrs + c1

    hcoefs = np.zeros(model_data['harmonic_degree']*2)
    for n in range(model_data['harmonic_degree']*2):
        hcoefs[n] = interpolate.splev(
                TIRS_lat, model_data['harmonic_model_tck'+str(n)], der=0)

    t_phase = t_decyrs * 2 * np.pi
    
    # CH4 model is expected to have 1 harmonic, CO2 model is expected to have
    # 3 harmonics.
    if model_data['harmonic_degree'] == 1:
        harmonics = np.array([
            np.sin(t_phase),
            np.cos(t_phase)
        ])
    elif model_data['harmonic_degree'] == 3:
        harmonics = np.array([
            np.sin(t_phase),
            np.cos(t_phase),
            np.sin(t_phase*2),
            np.cos(t_phase*2),
            np.sin(t_phase*3),
            np.cos(t_phase*3),
        ])
    else:
        raise Exception('The only expected values for number of harmonics '+\
                        'are 1 (CH4 model) or 3 (CO2 model)')
        
    harmonic_model = hcoefs @ harmonics
    
    model = harmonic_model + linear_model

    return model


def apply_CO2_CH4_model(TIRS_geo_data, model_fpath, FillValue):
    """
    Get CO2 / CH4 value for all TIRS scenes in a given orbit file by evaluating
    model scene-by-scene.

    Parameters
    ----------
    TIRS_geo_data : dict
        Dictionary containing arrays of TIRS data from a given orbit file: 
        lat, lon, surface elevation, and time.
    model_fpath : str
        Filepath of the CO2 or CH4 model file.

    Returns
    -------
    var_allscenes : ndarray
        Modelled CO2 or CH4 values at all TIRS scenes.

    """
    # Load model
    model_data = CO2_CH4_model_load(model_fpath)
    
    # Get latitude and time data at all TIRS scenes
    TIRS_lats = TIRS_geo_data['lats_deg']
    TIRS_time_arrays = TIRS_geo_data['times_UTC']

    # Loop through TIRS scenes, extract lat and time and evaluate model    
    var_allscenes = np.full(TIRS_lats.shape, FillValue)
    for ai, xi in np.ndindex(TIRS_lats.shape):
        var_allscenes[ai, xi] = CO2_CH4_model_eval(model_data,
                                                   TIRS_lats[ai,xi], 
                                                   TIRS_time_arrays[ai,xi])
    
    return var_allscenes
