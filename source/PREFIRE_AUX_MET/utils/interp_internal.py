#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import netCDF4 as n
import datetime as dt

from PREFIRE_tools.utils.aux import TIRS_L1B_time_array_to_dt


def compute_time_weights_GEOSIT(TIRS_geo_data,
                                GEOSIT_geo_data):
    """
    Compute temporal weights that will be used for interpolation of GEOS-IT analysis 
    data to TIRS scenes. These temporal weights are needed for both the "ESMF" 
    and "internal" interpolation methods (since ESMF only provides spatial 
    interpolation), and are independent of the GEOS-IT cubed sphere or equal 
    angle grid type.

    Parameters
    ----------
    TIRS_geo_data : dict
        Dictionary containing arrays of TIRS data from a given orbit file: 
        lat, lon, surface elevation, and time.
    GEOSIT_geo_data : dict
        Dictionary containing GEOS-IT constant lat / lon arrays and analysis times 
        for each GEOS-IT collection.

    Returns
    -------
    time_weights_dict : dict
        Interpolation weights for all GEOS-IT timesteps bounding the time span
        of the TIRS L1B file. Time weights are specific to each GEOS-IT collection.

    """

    # Time weights output arrays are shaped (n_frames, 8, n_analysis_times)
    #   - n_frames = number of frames in along-track direction
    #   - 8 = number of scenes in across-track direction
    #   - n_analysis_times = weights for analysis files for each collection, 
    #       ordered from earliest to latest

    time_weights_dict = {}
    
    TIRS_times_split = TIRS_geo_data['times_UTC']
    
    # Create array of TIRS datetime64 objects
    TIRS_dts = np.empty(shape=(np.shape(TIRS_times_split)[0],
                               np.shape(TIRS_times_split)[1]),
                        dtype='datetime64[us]')
    
    for atrack_ix in range(np.shape(TIRS_dts)[0]):
        for xtrack_ix in range(np.shape(TIRS_dts)[1]):
            TIRS_dt = TIRS_L1B_time_array_to_dt(TIRS_times_split[atrack_ix][xtrack_ix])
            TIRS_dts[atrack_ix, xtrack_ix] = np.datetime64(TIRS_dt)
    
    # Loop through GEOS-IT collections and create time weights array for each one
    GEOSIT_collections = [k.split('_time_UTC')[0] for k in GEOSIT_geo_data.keys() \
                          if k[0] == 'G']

    for collection in GEOSIT_collections:
        
        analysis_times = GEOSIT_geo_data[collection+'_time_UTC']
        
        time_weights = np.zeros(shape=(np.shape(TIRS_dts)[0],
                                       np.shape(TIRS_dts)[1],
                                       np.shape(analysis_times)[0]),
                                dtype='float32')
        
        for time_ix, analysis_time in enumerate(analysis_times[:-1]):
            
            window_begin_dt = np.datetime64(TIRS_L1B_time_array_to_dt(analysis_time))
            next_analysis_time = analysis_times[time_ix + 1]
            window_end_dt = np.datetime64(TIRS_L1B_time_array_to_dt(next_analysis_time))
            
            window_timedelta = window_end_dt - window_begin_dt
            TIRS_timedelta = TIRS_dts - window_begin_dt
            
            # Weights for window begin timestep
            time_weights_begin = (window_timedelta - TIRS_timedelta) / window_timedelta
            
            # If time weight is between 0 and 1 -- indicating that the given TIRS
            # time falls in the window between analysis time steps -- then place
            # the value in the time weights array. Otherwise, leave the pre-existing
            # values there. (When there are more than 2 GEOS-IT analysis time steps
            # to loop through, time steps after the first will encounter pre-existing
            # end-of-window weights placed there by earlier iterations of the loop.)
            time_weights[:,:,time_ix] = np.where(np.logical_and(time_weights_begin > 0,
                                                                time_weights_begin < 1),
                                                 time_weights_begin,
                                                 time_weights[:,:,time_ix])
                        
            # Weights for window end timestep
            time_weights_end = TIRS_timedelta / window_timedelta
            
            # Same explanation as above, but for the weights for the analysis time 
            # step at the end of the window.
            time_weights[:,:,time_ix+1] = np.where(np.logical_and(time_weights_end > 0,
                                                                  time_weights_end < 1),
                                                   time_weights_end,
                                                   time_weights[:,:,time_ix+1])
                
        time_weights_dict[collection] = time_weights
           
    return time_weights_dict


def compute_spatial_weights_GEOSIT(TIRS_geo_data,
                                   GEOSIT_geo_data):
    """
    Compute spatial weights that will be used for interpolation of GEOS-IT analysis 
    data to TIRS scenes using the "internal" interpolation method. This method will
    *ONLY* work with GEOS-IT data on *equal angle* grid.
    
    Parameters
    ----------
    TIRS_geo_data : dict
        Dictionary containing arrays of TIRS data from a given orbit file: 
        lat, lon, surface elevation, and time.
    GEOSIT_geo_data : dict
        Dictionary containing GEOS-IT constant lat / lon arrays and analysis times 
        for each GEOS-IT collection.
    
    Returns
    -------
    spatial_indices : numpy.ndarray
        Array indices of the four GEOS-IT grid points that surround each TIRS scene.
    spatial_weights : numpy.ndarray
        Interpolation weights for the four GEOS-IT grid points that surround each 
        TIRS scene.
        
    """
    
    # The differences between TIRS lats/lons and nearest GEOS-IT lats/lons are used
    #   to compute weights, and index values of W/E/S/N bounding points are stored
    #   for applying weights to met variables (in a different function).
    
    # Spatial weights and indices output arrays are shaped (n_frames, 8, 2, 2)
    #   - n_frames = number of frames in along-track direction
    #   - 8 = number of scenes in across-track direction
    #   - 2 = weights for W and E bounding longitudes, with W listed first
    #   - 2 = weights for S and N bounding latitudes, with S listed first

    # ** This section would need to be adapted for different met analysis sources
    #   with different grids -- the specific values below are used for the
    #   GEOS-IT equal angle grid, with spacing of 0.625 deg lon, 0.5 deg lat

    # ** OR this could be written to be more general for fixed grids, by including
    #   a step here to calculate the increments between successive lons and lats,
    #   and using this info to find the nearest lats and lons below
    #   - This would need a check to see how the met analysis grid handles
    #       lons at the date line
    #   - And a check to handle different longitude conventions? 
    #       (e.g. -180 to 180 vs 0 to 360)

    TIRS_lons = TIRS_geo_data['lons_deg']
    TIRS_lats = TIRS_geo_data['lats_deg']

    # Find nearest lon and lat to each TIRS point
    GEOSIT_nearest_lons = np.around(TIRS_lons*(8/5)) / (8/5)
    GEOSIT_nearest_lats = np.around(TIRS_lats*2) / 2

    # Find bounding lons on GEOS-IT grid that encompass TIRS points
    GEOSIT_w_bound_lons = np.where((TIRS_lons - GEOSIT_nearest_lons) > 0,
                                   GEOSIT_nearest_lons,
                                   GEOSIT_nearest_lons - (5/8))
    GEOSIT_e_bound_lons = np.where((TIRS_lons - GEOSIT_nearest_lons) > 0,
                                   GEOSIT_nearest_lons + (5/8),
                                   GEOSIT_nearest_lons)
    
    # GEOS-IT grid ranges from lons -180 to 179.375, so indices of eastern
    # boundary lons of 180 are out of bounds. Replace these eastern boundary
    # longitudes with -180 for indexing purposes.
    GEOSIT_e_bound_lons_for_ixs = np.copy(GEOSIT_e_bound_lons)
    np.place(GEOSIT_e_bound_lons_for_ixs, GEOSIT_e_bound_lons_for_ixs==180, -180)

    # Find bounding lats on GEOS-IT grid that encompass TIRS points        
    GEOSIT_s_bound_lats = np.where((TIRS_lats - GEOSIT_nearest_lats) > 0,
                                   GEOSIT_nearest_lats,
                                   GEOSIT_nearest_lats - (1/2))
    GEOSIT_n_bound_lats = np.where((TIRS_lats - GEOSIT_nearest_lats) > 0,
                                   GEOSIT_nearest_lats + (1/2),
                                   GEOSIT_nearest_lats)
    
    # Find array indices of GEOS-IT bounding lons
    GEOSIT_w_bound_ixs = ((GEOSIT_w_bound_lons + 180) / (5/8)).astype(int)
    GEOSIT_e_bound_ixs = ((GEOSIT_e_bound_lons_for_ixs + 180) / (5/8)).astype(int)
    
    # Find array indices of GEOS-IT bounding lats
    GEOSIT_s_bound_ixs = ((GEOSIT_s_bound_lats + 90) / (1/2)).astype(int)
    GEOSIT_n_bound_ixs = ((GEOSIT_n_bound_lats + 90) / (1/2)).astype(int)

    # Concatenate spatial indices into a single array (nframes,8,2,2)
    indices_lon = np.stack([GEOSIT_w_bound_ixs, GEOSIT_e_bound_ixs], axis=2)
    indices_lat = np.stack([GEOSIT_s_bound_ixs, GEOSIT_n_bound_ixs], axis=2)
    
    spatial_indices = np.concatenate([indices_lon[...,np.newaxis],
                                      indices_lat[...,np.newaxis]],
                                      axis=3)

    # Calulate weights for GEOS-IT bounding lons / lats
    w_bound_weights = (GEOSIT_e_bound_lons - TIRS_lons) / \
                      (GEOSIT_e_bound_lons - GEOSIT_w_bound_lons)
    e_bound_weights = 1 - w_bound_weights
    
    s_bound_weights = (GEOSIT_n_bound_lats - TIRS_lats) / \
                      (GEOSIT_n_bound_lats - GEOSIT_s_bound_lats)
    n_bound_weights = 1 - s_bound_weights
    
    # Concatenate spatial weights into a single array (nframes,8,2,2) 
    weights_lon = np.stack([w_bound_weights, e_bound_weights], axis=2)
    weights_lat = np.stack([s_bound_weights, n_bound_weights], axis=2)
    
    spatial_weights = np.concatenate([weights_lon[...,np.newaxis],
                                      weights_lat[...,np.newaxis]],
                                      axis=3)
    
    return spatial_indices, spatial_weights


def interp_loop_internal(spatial_indices,
                         spatial_weights,
                         time_weights,
                         var_all_timesteps,
                         var_input_fillvalue, var_output_fillvalue):
    """
    Helper for apply_interp_weights that loops through all the along-track and 
    cross-track indices of a TIRS array and performs the actual spatial and temporal 
    interpolation calculations.
    
    Parameters
    ----------
    spatial_indices : numpy.ndarray
        Array indices of the four GEOS-IT grid points that surround each TIRS scene.
    spatial_weights : numpy.ndarray
        Interpolation weights for the four GEOS-IT grid points that surround each 
        TIRS scene.
    time_weights : numpy.ndarray
        Interpolation weights for all GEOS-IT timesteps bounding the time span
        of the TIRS L1B file. An array of time weights for a given GEOS-IT 
        collection is passed to this function.
    var_all_timesteps : numpy.ndarray
        Data to be interpolated for a given variable. It is shaped 
        (ntimesteps x nlat x nlon), where ntimesteps is the number of analysis
        timesteps for a given GEOS-IT collection.
    var_input_fillvalue : dtype of 'var_all_timesteps'
        The input file value of "_FillValue" for 'var_all_timesteps'
    var_output_fillvalue : dtype of 'var_all_timesteps'
        The output file value of "_FillValue" for the associated output variable

    Returns
    -------
    interp_array_2D : numpy.ndarray
        Variable data interpolated to TIRS scenes.

    """
    # Output array in which interpolated data will be placed
    interp_array_2D = np.empty(shape=(np.shape(spatial_indices)[0],
                                      np.shape(spatial_indices)[1]),
                               dtype='float32')
    
    # Loop through TIRS along-track and cross-track indices and perform
    # interpolation
    for atrack_ix, xtrack_ix in np.ndindex(np.shape(interp_array_2D)):
            
        si = spatial_indices[atrack_ix, xtrack_ix]
        sw = spatial_weights[atrack_ix, xtrack_ix]
        tw = time_weights[atrack_ix, xtrack_ix]
        
        # Interpolated value starts as 0, then gets added to after
        # applying spatial and temporal weights
        v_interp = 0
        
        for time_ix in range(len(tw)):
            
            # Analysis value for box bounding TIRS observation point:
            #   - v11 = lower left (s,w corner)
            #   - v12 = upper left (n,w corner)
            #   - v21 = lower right (s,e corner)
            #   - v22 = upper right (n,e corner)
            v11 = var_all_timesteps[time_ix, si[0,1], si[0,0]]
            v12 = var_all_timesteps[time_ix, si[1,1], si[0,0]]
            v21 = var_all_timesteps[time_ix, si[0,1], si[1,0]]
            v22 = var_all_timesteps[time_ix, si[1,1], si[1,0]]
            
            # ** GEOS-IT ONLY: Set interpolated value to PREFIRE fill value 
            # for any TIRS scene center points with a GEOS-IT fill value 
            # at any of the 4 surrounding grid cells. The only 
            # known scenario where this will happen is for land_surface_temp and 
            # snow_cover values that are not completely surrounded by GEOS-IT 
            # land grid points.
            if np.allclose([max([v11, v12, v21, v22])], [var_input_fillvalue]):
                v_interp = var_output_fillvalue
                break
            
            else:
                # Weights bounding TIRS observation point:
                #   - wx1 = weight for x position 1 (w bounding lon)
                #   - wx2 = weight for x position 2 (e bounding lon)
                #   - wy1 = weight for y position 1 (s bounding lat)
                #   - wy2 = weight for y position 2 (n bounding lat)
                wx1 = sw[0,0]
                wx2 = sw[1,0]
                wy1 = sw[0,1]
                wy2 = sw[1,1]
                
                # Multiply spatial values by spatial weights to get 
                # spatially interpolated value at given timestep
                v = wx1*wy1*v11 + wx1*wy2*v12 + wx2*wy1*v21 + wx2*wy2*v22
                
                # Multiply spatially interpolated value by temporal weight
                # at a given timestep, and add to total spatially / 
                # temporally interpolated value
                v_interp += v*tw[time_ix]
        
        interp_array_2D[atrack_ix, xtrack_ix] = v_interp
    
    return interp_array_2D

def apply_interp_weights_GEOSIT(GEOSIT_collection_flists,
                                variables,
                                GEOSIT_ESDT_dict,
                                var_name_GEOSIT_dict,
                                spatial_indices,
                                spatial_weights,
                                time_weights_dict,
                                data_product_specs):

    """
    Apply interpolation weights to GEOS-IT analysis variables and return 
    dictionary of data interpolated to TIRS scenes for all variables.

    Parameters
    ----------
    GEOSIT_collection_flists : dict
        Dictionary containing the lists of files for each GEOS-IT collection that
        completely encompass the TIRS observation times. Only GEOS-IT collections
        containing data that we need (4 in total) are included.
    variables : list
        Names of variables for which interpolated data is needed.
    GEOSIT_ESDT_dict : dict
        Dictionary mapping Aux-Met variable names to GEOS-IT ESDTs.
    var_name_GEOSIT_dict : dict
        Dictionary mapping Aux-Met variable names to GEOS-IT variable names.        
    spatial_indices : numpy.ndarray
        Array indices of the four GEOS-IT grid points that surround each TIRS scene.
    spatial_weights : numpy.ndarray
        Interpolation weights for the four GEOS-IT grid points that surround each 
        TIRS scene.
    time_weights_dict : dict
        Interpolation weights for all GEOS-IT timesteps bounding the time span
        of the TIRS L1B file. Time weights are specific to each GEOS-IT collection.
    data_product_specs : dict
        Dictionary of data product specifications

    Returns
    -------
    interp_data : dict
        Interpolated data for all variables.

    """

    interp_data = {}
    GEOSIT_ds_objects = {}
    
    for var_name in variables:
        
        collection = GEOSIT_ESDT_dict[var_name]
        flist = GEOSIT_collection_flists[collection]
        time_weights = time_weights_dict[collection]

        for fpath in flist:
            if not fpath in GEOSIT_ds_objects:
                GEOSIT_ds_objects[fpath] = n.Dataset(fpath)

        # "peek" variable is used to determine shape of variable data (lat, lon, nlevels)
        peek_nc_path = flist[0]
        peek_var_shape = GEOSIT_ds_objects[peek_nc_path]. \
                                variables[var_name_GEOSIT_dict[var_name]].shape
        peek_var_input_fillvalue = GEOSIT_ds_objects[peek_nc_path]. \
              variables[var_name_GEOSIT_dict[var_name]].getncattr("_FillValue")
        var_output_fillvalue = \
                          data_product_specs["Aux_Met"][var_name]["fill_value"]
        
        # 2-D single-level variables
        if len(peek_var_shape) == 3:
            
            var_all_timesteps = np.empty(shape=(len(flist),
                                                peek_var_shape[1],
                                                peek_var_shape[2]),
                                         dtype='float32')
            
            # Build full variable array for all timesteps (# of timesteps depends 
            # on collection)
            #   - shape: ntimesteps x nlat x nlon
            for time_ix, fpath in enumerate(flist):
                var_timestep = GEOSIT_ds_objects[fpath]. \
                               variables[var_name_GEOSIT_dict[var_name]][0,:,:]

                var_all_timesteps[time_ix,:,:] = var_timestep

            interp_array = interp_loop_internal(spatial_indices,
                                                spatial_weights,
                                                time_weights,
                                                var_all_timesteps,
                                                peek_var_input_fillvalue,
                                                var_output_fillvalue)
            
        # 3-D profile variables
        elif len(peek_var_shape) == 4:
            
            # Output array in which interpolated data will be placed - third
            # dimension is model levels
            interp_array = np.empty(shape=(np.shape(spatial_indices)[0],
                                           8,
                                           peek_var_shape[1]),
                                    dtype='float32')
            
            # Treat each level as a separate variable for interpolation
            for model_lev in range(peek_var_shape[1]):
                
                var_all_timesteps = np.empty(shape=(len(flist),
                                                    peek_var_shape[2],
                                                    peek_var_shape[3]),
                                             dtype='float32')

                # Build full variable array for all timesteps (# of timesteps depends 
                # on collection)
                #   - shape: ntimesteps x nlat x nlon
                for time_ix, fpath in enumerate(flist):
                    var_timestep = GEOSIT_ds_objects[fpath].variables \
                              [var_name_GEOSIT_dict[var_name]][0,model_lev,:,:]
                    
                    var_all_timesteps[time_ix,:,:] = var_timestep

                interp_array[:,:,model_lev] = interp_loop_internal(spatial_indices,
                                                                   spatial_weights,
                                                                   time_weights,
                                                                   var_all_timesteps,
                                                                   peek_var_input_fillvalue,
                                                                   var_output_fillvalue)
        
        interp_data[var_name] = interp_array
        
        print('Finished interp var: '+var_name+' at '+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return interp_data


def land_frac_interp_GEOSIT_internal(asm_const_fpath, spatial_indices,
                                     spatial_weights):
    """
    Interpolate GEOS-IT land fraction (FRLAND + FRLANDICE) to each TIRS scene
    center point.

    Parameters
    ----------
    asm_const_fpath : str
        Filepath of GEOS-IT time-invariant file containing FRLAND and FRLANDICE
        variables.
    spatial_indices : numpy.ndarray
        Array indices of the four GEOS-IT grid points that surround each TIRS scene.
    spatial_weights : numpy.ndarray
        Interpolation weights for the four GEOS-IT grid points that surround each 
        TIRS scene.

    Returns
    -------
    land_fraction : numpy.ndarray
        GEOS-IT land fraction interpolated to each TIRS scene center point.

    """
    
    asm_const_nc = n.Dataset(asm_const_fpath)
    frland = asm_const_nc.variables['FRLAND'][0][...]
    frlandice = asm_const_nc.variables['FRLANDICE'][0][...]
    asm_const_nc.close()
    
    # Output array in which interpolated data will be placed
    land_fraction = np.empty(shape=(np.shape(spatial_indices)[0],
                                    np.shape(spatial_indices)[1]),
                             dtype='float32')
    
    # Loop through TIRS along-track and cross-track indices and perform
    # interpolation
    for atrack_ix, xtrack_ix in np.ndindex(np.shape(land_fraction)):
            
        si = spatial_indices[atrack_ix, xtrack_ix]
        sw = spatial_weights[atrack_ix, xtrack_ix]
                    
        # Analysis value for box bounding TIRS observation point:
        #   - v11 = lower left (s,w corner)
        #   - v12 = upper left (n,w corner)
        #   - v21 = lower right (s,e corner)
        #   - v22 = upper right (n,e corner)
        v11 = frland[si[0,1], si[0,0]] + frlandice[si[0,1], si[0,0]]
        v12 = frland[si[1,1], si[0,0]] + frlandice[si[1,1], si[0,0]]
        v21 = frland[si[0,1], si[1,0]] + frlandice[si[0,1], si[1,0]]
        v22 = frland[si[1,1], si[1,0]] + frlandice[si[1,1], si[1,0]]
            
        # Weights bounding TIRS observation point:
        #   - wx1 = weight for x position 1 (w bounding lon)
        #   - wx2 = weight for x position 2 (e bounding lon)
        #   - wy1 = weight for y position 1 (s bounding lat)
        #   - wy2 = weight for y position 2 (n bounding lat)
        wx1 = sw[0,0]
        wx2 = sw[1,0]
        wy1 = sw[0,1]
        wy2 = sw[1,1]
                
        # Multiply spatial values by spatial weights to get 
        # spatially interpolated value
        v = wx1*wy1*v11 + wx1*wy2*v12 + wx2*wy1*v21 + wx2*wy2*v22
        
        land_fraction[atrack_ix, xtrack_ix] = v
    
    return land_fraction
