#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def get_layer_hybrid_coefs(coef_file='geos5_hybrid_sigma_72_double.txt'):
    """
    get_layer_hybrid_coefs

    example of how to convert the GEOS-IT hybrid sigma coefficients, which
    define the pressure levels (e.g. layer boundaries) into the mid-layer
    quantities expected for the surf_correction function.

    note the expected coef arrays have 73 elements: one for each mid-layer
    pressure (which is just the average) and then the surface pressure
    as the bottom level.


    Parameters
    ----------
    coef_file : str
       the file path for the coefficient file


    Returns
    -------
    alayer : ndarray
        ak coefficients defined for mid-layer, shape (73,)
    blayer : ndarray
        bk coefficients defined for mid-layer, shape (73,)
    Note the returned coefficients are defined for Psurf in [Pa].
    """

    # file columns are: level index, Ak, Bk, Pref, DelP - we only need
    # ak, bk, so discard the rest.
    _, ak, bk, _, _ = np.loadtxt(coef_file, skiprows=1).T

    # convert hPa to Pa.
    ak *= 100
    alayer = 0.5 * (ak[1:] + ak[:-1])
    blayer = 0.5 * (bk[1:] + bk[:-1])
    alayer = np.append(alayer, 0.0)
    blayer = np.append(blayer, 1.0)

    return alayer, blayer


def surf_correction(p_surf, t_profile, q_profile,
                    phi_surf, T2m, ak, bk, obs_elev,
                    other_profiles = None, verbose = False):

    """
    surface_correction

    correct a set of profile variables, that are functions of pressure,
    for a measurement surface elevation that is different from
    the analysis surface elevation. This could mean an interpolation
    (if the measurement is at a higher surface elevation) or extrapolation
    (if the measurement is at a lower elevation).

    Parameters
    ----------

    p_surf : float
        surface pressure from NWP model [Pa]
    t_profile : ndarray
        temperature profile from NWP analysis model. Assumed to be layer
        average quantities, with an additional value for the surface level.
        If the shape is (k,) this implies k-1 layers. [K]
    q_profile : ndarray
        water vapor profile, specific humidity, assumed to be layer average
        quantities, with an additional value for the surface. Same shape as
        t_profile [kg/kg]
    phi_surf : float
        surface geopotential from NWP model [m^2 / s^2]
    T2m : float
        Two-meter temperature from NWP model [K]
    ak : ndarray
        hybrid sigma a-coefficients to compute layer average pressures in Pa.
    bk : ndarray
        hybrid sigma b-coefficients to compute layer average pressures in Pa.
        the pressure profile will be computed as: p = ak + bk * psurf
        The ak, bk coefficients should have shape (k,), so that the final
        elements reproduce the surface level: ak[-1] = 0.0, bk[-1] = 1.0
    obs_elev : float
        surface elevation for the sensor's observed scene footprint. [m]

    other_profiles : list
        a list of ndarray, each having the same shape as t_profile. These will
        be interpolated in the same manner as the t and q profiles and returned
        as a list. These can be any quantity, any units, since these are not used
        to determine the interpolation.
        by default, this is None, meaning no other profiles are processed.
    verbose : bool
        set to True to get some diagnostic information printed to console.
        (this was primarily used for debugging/testing during development,
        and could be removed...)

    Returns
    -------
    p_surf_c : float
        the corrected surface pressure after accounting for the observation
        elevation [Pa]
    p_profile_c : ndarray
        the corrected pressure profile
    t_profile_c : ndarray
        corrected temperature profile, same shape as input t_profile [K]
    q_profile_c : ndarray
        corrected specific humidity [kg/kg]
    other_profiles_c : list
        optional - if other_profiles is None, this argument is not returned.
        Otherwise this is a list matching the length of the other_profiles list.
    """

    # hard coded physical constants, in MKS units, along with a few constants
    # derived from these to make the equations simpler.
    g0 = 9.80665
    Rd = 287
    Rv = 461
    epsilon = Rd / Rv
    v = 1 / epsilon - 1
    c = g0 / Rd

    # convert the model GPH to altitude, assuming a fixed value of g.
    model_elev = phi_surf / g0

    # delta temperature from two-meter T - this gets added back after
    # the hypsometric adjustment.
    dT2m = T2m - t_profile[-1]

    # the elevation change we need to make
    dz = obs_elev - model_elev

    # compute pressure profile - remember this should be the mid-layer
    # pressures, with the surface pressure as the last element.
    p = ak + bk * p_surf

    # simple iterative loop to implement hypsometric adjustment.
    # the problem is that we have T defined on P, but the adjustment is
    # defined in Z. We need to do an initial hypsometric adjustment
    # with virtual temperature computed from the surface level. Then we
    # can compute a more correct mean virtual temperature for this near-surface
    # layer using the adjusted vertical position.

    p_mean = p_surf

    for n in range(2):

        if verbose:
            print('iter ', n, ' start p_mean: ', p_mean)

        # locate the vertical layer index, then compute the linear weight
        k = np.searchsorted(p, p_mean) - 1
        if k == (p.shape[0]-1):
            k -= 1
        wt = (p[k+1] - p_mean) / (p[k+1] - p[k])

        if verbose:
            print('weights and bracketing p values: ', wt, 1-wt, k, p[k], p[k+1])

        # interpolate t, q, with linear weight
        t_hyp = t_profile[k+1]*(1-wt) + t_profile[k]*wt
        q_hyp = q_profile[k+1]*(1-wt) + q_profile[k]*wt

        # virtual temperature and hypsometric adjustment.
        Tv = t_hyp * (1 + v*q_hyp)
        p_hyp = p_surf * np.exp(-dz * c / Tv)
        # now that we have an adjusted pressure, 
        # find the more accurate mean pressure for this near-surface
        # layer, and repeat calculation.
        p_mean = 0.5 * (p_hyp + p_surf)

        if verbose:
            print('th, qh, tv, ph, end p_mean:', t_hyp, q_hyp, Tv, p_hyp, p_mean)

    p_surf_c = min(p_hyp, 109999.)  # [Pa], kludge evades values > 1100 hPa

    # tricky steps to finalize things:
    # apply T2m adjustment to surface level - copy the array so we don't
    # modify this in the caller.
    t_profile_adj = t_profile.copy()
    t_profile_adj[-1] += dT2m

    # derive corrected profiles.
    # if we are extrapolating (obs. surface is below nwp surface), make a new
    # make a new pressure array that is a copy of the input pressure array, but
    # contains the new surface pressure. This means the new values will be interpolated
    # between the lowest altitude layer and the new surface pressure; for temperature
    # this will be the temperature at the lowest altitude layer and the T2m-adjusted
    # surface level temp. This is preferred over extrapolating from those two values
    # at the mid-level and (old) surface pressure - if the altitude adjustment is large,
    # and T2m - t_profile[-2] is large (adiabatic or maybe superadiabatic?), then
    # extrapolating could make a large temperature increase. Same thing could happen
    # in reverse if there was a sharp inversion.
    p_adj = p.copy()
    if p_surf_c > p_surf:
        p_adj[-1] = p_surf_c

    # now get the new pressure profile; the new profiles are the linear interps
    # from p_adj to p_c. Note that the adjustment made just above means we never
    # actually extrapolate, so using np.interp is OK.
    p_c = ak + bk * p_surf_c

    t_profile_c = np.interp(p_c, p_adj, t_profile_adj)
    q_profile_c = np.interp(p_c, p_adj, q_profile)
    if other_profiles:
        other_profiles_c = [np.interp(p_c, p_adj, prof) for prof in other_profiles]

    if other_profiles:
        return p_surf_c, p_c, t_profile_c, q_profile_c, other_profiles_c, dz
    else:
        return p_surf_c, p_c, t_profile_c, q_profile_c, dz


def apply_surf_correction(interp_data_uncorr, obs_elev, obs_elev_fillvalue,
                          coef_fpath, data_product_specs,
                          ancsim_vars=None):
    """
    Correct vertical profile met analysis variables to account for the difference
    between the source dataset surface elevation and "true" scene elevation 
    contained in L1B Geometry group.
    
    Parameters
    ----------
    interp_data : dict
        Met analysis data interpolated in space and time to TIRS scene center points.
    obs_elev : numpy.ndarray
        "True" surface altitude for each TIRS scene derived from L1B Geometry group.
    obs_elev_fillvalue : float
        File "_FillValue" (missing value) for the field in 'obs_elev'.
    coef_fpath : str
        Path to text file containing a & b coefficients for GEOS5 model
    data_product_specs : dict
        Dictionary of data product specifications
    ancsim_vars : list
        Extra cloud-related variables output if code is run in ANC-SimTruth
        mode.

    Returns
    -------
    interp_data_sfc_corr : dict
        Met analysis data with surface correction applied to 3D variables.

    """
    
    ak, bk = get_layer_hybrid_coefs(coef_file=coef_fpath)
    
    p_surf = interp_data_uncorr['surface_pressure']
    t_profile = interp_data_uncorr['temp_profile']
    q_profile = interp_data_uncorr['wv_profile']
    o3_profile = interp_data_uncorr['o3_profile']
    phi_surf = interp_data_uncorr['surface_phi']
    T2m = interp_data_uncorr['temp_2m']
    
    # Third dimension contains mid-layer values from met analysis, plus value
    # at surface tacked on
    corr_shape = (np.shape(t_profile)[0],
                  np.shape(t_profile)[1],
                  np.shape(t_profile)[2]+1)
    
    interp_data_corr = {}

    vars_2d = ['surface_pressure', 'elevation_correction']
    for v in vars_2d:
         interp_data_corr[v] = np.full(corr_shape[0:2],
                                data_product_specs["Aux-Met"][v]["fill_value"])

    vars_3d = ['pressure_profile', 'temp_profile', 'wv_profile']
    for v in vars_3d:
         interp_data_corr[v] = np.full(corr_shape,
                                data_product_specs["Aux-Met"][v]["fill_value"])
       
    other_3d_vars = ['o3_profile','altitude_profile','u_profile','v_profile','omega_profile']
    for v in other_3d_vars:
        interp_data_corr[v] = np.full(corr_shape, 
                                data_product_specs["Aux-Met"][v]["fill_value"])
    if ancsim_vars:
        other_3d_vars.extend(ancsim_vars)
        for v in ancsim_vars:
            interp_data_corr[v] = np.full(corr_shape, 
                               data_product_specs["SimTruth"][v]["fill_value"])
    
    for ai, xi in np.ndindex(corr_shape[0:2]):
        # Extrapolate to surface pressure from lowest two altitude layers
        t_inc_sfc = np.append(t_profile[ai,xi],
                              t_profile[ai,xi][-1]*1.5 - t_profile[ai,xi][-2]*0.5)
        # Water vapor and o3 values should always be positive. For these profiles, 
        # if extrapolation to surface pressure results in a negative value,
        # append the value from the lowest model layer instead.
        # - The only known case where this will happen is when there is a
        #   water vapor inversion sharp enough such that 1.5x the value in the first
        #   model layer above the surface is less than 0.5x the value in the second
        #   model layer above the surface.
        if (q_profile[ai,xi][-1]*1.5 - q_profile[ai,xi][-2]*0.5) < 0:
            q_inc_sfc = np.append(q_profile[ai,xi],
                                  q_profile[ai,xi][-1])
        else:
            q_inc_sfc = np.append(q_profile[ai,xi],
                                  q_profile[ai,xi][-1]*1.5 - q_profile[ai,xi][-2]*0.5)
        if (o3_profile[ai,xi][-1]*1.5 - o3_profile[ai,xi][-2]*0.5) < 0:
            o3_inc_sfc = np.append(o3_profile[ai,xi],
                                   o3_profile[ai,xi][-1])
        else:
            o3_inc_sfc = np.append(o3_profile[ai,xi],
                                   o3_profile[ai,xi][-1]*1.5 - o3_profile[ai,xi][-2]*0.5)
        
        other_profs_inc_sfc = [o3_inc_sfc]
        for v in other_3d_vars:
            if v == 'o3_profile':
                pass
            elif v == 'cloud_fraction_profile':
                prof_inc_sfc = np.append(
                    interp_data_uncorr[v][ai,xi],
                    np.clip(
                        interp_data_uncorr[v][ai,xi][-1]*1.5 - interp_data_uncorr[v][ai,xi][-2]*0.5,
                        a_min=0, a_max=1
                        )
                    )
                other_profs_inc_sfc.append(prof_inc_sfc)
            elif v in ['qi_profile','ql_profile']:
                v_prof = interp_data_uncorr[v][ai,xi]
                if (v_prof[-1]*1.5 - v_prof[-2]*0.5) < 0:
                    prof_inc_sfc = np.append(v_prof, v_prof[-1])
                else:
                    prof_inc_sfc = np.append(
                        v_prof,
                        v_prof[-1]*1.5 - v_prof[-2]*0.5
                        )
                other_profs_inc_sfc.append(prof_inc_sfc)
            elif v == 'altitude_profile':
                prof_inc_sfc = np.append(
                    interp_data_uncorr['altitude_profile'][ai,xi],
                    obs_elev[ai,xi]
                    )

                other_profs_inc_sfc.append(prof_inc_sfc)
            else:
                prof_inc_sfc = np.append(interp_data_uncorr[v][ai,xi],
                                         interp_data_uncorr[v][ai,xi][-1]*1.5 - interp_data_uncorr[v][ai,xi][-2]*0.5)
                other_profs_inc_sfc.append(prof_inc_sfc)
       
        # TIRS scenes with masked elevation values: don't change the existing _FillValue
        # in surface pressure, obs_minus_model_altitude, and vertical profile arrays.
        if not np.allclose([obs_elev[ai,xi]], [obs_elev_fillvalue]):
            p_surf_c_Pa, p_c_Pa, t_profile_c, q_profile_c, other_profiles_c, dz = \
                surf_correction(p_surf[ai,xi],
                                t_inc_sfc,
                                q_inc_sfc,
                                phi_surf[ai,xi],
                                T2m[ai,xi],
                                ak,
                                bk,
                                obs_elev[ai,xi],
                                other_profiles=other_profs_inc_sfc)
            
            # Convert pressure to hPa to match file spec and expected input for 
            # pressure_interp function
            p_surf_c = p_surf_c_Pa / 100
            p_c = p_c_Pa / 100
            
            interp_data_corr['surface_pressure'][ai,xi] = p_surf_c
            interp_data_corr['elevation_correction'][ai,xi] = dz
            interp_data_corr['pressure_profile'][ai,xi] = p_c
            interp_data_corr['temp_profile'][ai,xi] = t_profile_c
            interp_data_corr['wv_profile'][ai,xi] = q_profile_c
        
            for i, v in enumerate(other_3d_vars):
                interp_data_corr[v][ai,xi] = other_profiles_c[i]
            
    for v in interp_data_uncorr.keys():
        if v not in interp_data_corr.keys():
            interp_data_corr[v] = interp_data_uncorr[v]
    
    return interp_data_corr
