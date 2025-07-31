#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import esmpy
import numpy as np
from netCDF4 import Dataset
import datetime as dt
import shutil
import os
import glob
import subprocess
import json
import sys

from PREFIRE_PRD_GEN.file_read import get_PREFIRE_Lx_field
# import PREFIRE_sim_tools.paths


def create_locstream_TIRS(TIRS_L1B_ds, atrack_range_to_process):
    """
    load lat/lon from PREFIRE/TIRS Geometry group and convert to EMSF LocStream object.
    
    lats, lons, msk are assumed to be the same shape, any dimension.
    msk is a boolean masking array with True for good indices.

    LocStream only takes 1D vectors of lat/lon, so lat/lon are
    basically flattened by the msk, before storing in the LocStream
    """

    TIRS_L1B_ds.set_auto_mask(False)  # Temporarily disable masked array output
    lats_unmsk = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry", "latitude",
                                    atrack_range_to_process)
    lons_unmsk = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry", "longitude",
                                    atrack_range_to_process)
    valid_msk = np.logical_not(np.isclose(lats_unmsk,
                               [TIRS_L1B_ds.groups["Geometry"]. \
                               variables["latitude"].getncattr("_FillValue")]))
    TIRS_L1B_ds.set_auto_mask(True)  # Re-enable masked array output

    lats = lats_unmsk[valid_msk]
    lons = lons_unmsk[valid_msk]

    # requires specific numeric type
    lats = lats.astype(np.float64)
    lons = lons.astype(np.float64)

    npoints = lats.shape[0]
    coord_sys = esmpy.CoordSys.SPH_DEG
    
    locstream = esmpy.LocStream(npoints, coord_sys=coord_sys)
    locstream["ESMF:Lon"] = lons
    locstream["ESMF:Lat"] = lats

    return locstream, valid_msk


def interp_GEOSIT_ESMF_ea(TIRS_L1B_ds, TIRS_L1B_fpath, GEOSIT_asm_const_fpath,
                          GEOSIT_collection_flists,
                          variables,
                          GEOSIT_ESDT_dict,
                          var_name_GEOSIT_dict,
                          time_weights_dict,
                          work_dir, atrack_range_to_process,
                          data_product_specs):
    """
    Interpolate GEOS-IT *equal angle* data to TIRS scenes in space and time,
    using ESMF for spatial interpolation. Unlike for cubed sphere data, ESMF 
    regridding is handled internally within this function, because system call
    to mpirun is not necessary for equal angle data.
    
    Parameters
    ----------
    TIRS_L1B_ds : netCDF4.Dataset
        A netCDF4.Dataset object referencing the TIRS Level-1B data file.
    TIRS_L1B_fpath : str
        Path to TIRS L1B netCDF file.
    GEOSIT_asm_const_fpath : str
        Filepath of the constant-in-time GEOS-IT input file. 
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
    time_weights_dict : dict
        Interpolation weights for all GEOS-IT timesteps bounding the time span
        of the TIRS L1B file. Time weights are specific to each GEOS-IT collection.
    work_dir : Work directory.
    atrack_range_to_process : 3-tuple (str, int, int)
        A 3-tuple containing the dimension name (in this case, "atrack") to
         subset, and start and stop indices (NumPy indexing convention) in the
         given granule to process
    data_product_specs : dict
        Dictionary of data product specifications

    Returns
    -------
    interp_data : dict
        Interpolated data for all variables.

    """
   
    locstream, valid_msk = create_locstream_TIRS(TIRS_L1B_ds,
                                                 atrack_range_to_process)
    
    # Use a vertical profile file to determine grid characteristics, including
    # number of vertical levels.
    fpath_3d = GEOSIT_collection_flists['GEOSIT_ASM_I3_L_V72'][0]
    grid = esmpy.Grid(filename=fpath_3d, filetype=esmpy.FileFormat.GRIDSPEC)
    with Dataset(fpath_3d, 'r') as grid_nc:
        lm = len(grid_nc.dimensions['lev'])
    
    srcfield2D = esmpy.Field(grid,name="srcfield2D",typekind=esmpy.TypeKind.R4)
    dstfield2D = esmpy.Field(locstream,name="dstfield2D",typekind=esmpy.TypeKind.R4)
    srcfield3D = esmpy.Field(grid,name="srcfield3D",ndbounds=[lm],typekind=esmpy.TypeKind.R4)
    dstfield3D = esmpy.Field(locstream,name="dstfield3D",ndbounds=[lm],typekind=esmpy.TypeKind.R4)
    # Bilinear interpolation
    regrid = esmpy.Regrid(srcfield2D,dstfield2D,regrid_method=esmpy.RegridMethod.BILINEAR)

    interp_data = {}

    for var_name in variables:
        # Only a single timestep for land_fraction
        if var_name == 'land_fraction':
            with Dataset(GEOSIT_asm_const_fpath) as const_nc:
                temp = const_nc.variables['FRLAND'][:] + const_nc.variables['FRLANDICE'][:]
            srcfield2D.data[:] = np.transpose(temp[0,:,:])
            regrid(srcfield2D,dstfield2D)
            output_shape = valid_msk.shape
            out_array = np.zeros(output_shape)
            # Reshapes the regridded data to 2D
            out_array[valid_msk] = dstfield2D.data
            interp_data[var_name] = out_array
        else:
            collection = GEOSIT_ESDT_dict[var_name]
            flist = GEOSIT_collection_flists[collection]
            time_weights = time_weights_dict[collection]

            peek_nc_fpath = flist[0]
            
            # "peek" variable is used to determine shape of variable data (lat, lon, nlevels)
            peek_nc = Dataset(peek_nc_fpath)
            peek_var_shape = peek_nc.variables[var_name_GEOSIT_dict[var_name]].shape
            peek_var_input_fillvalue = peek_nc. \
                                   variables[var_name_GEOSIT_dict[var_name]]. \
                                   getncattr("_FillValue")
            peek_nc.close()

            # Create arrays of zeros that will hold interpolated data for all timesteps
            # 2-D single-level variables
            if len(peek_var_shape) == 3:
                interp_array = np.zeros(shape=(valid_msk.shape),
                                        dtype='float32')
            # 3-D variables
            elif len(peek_var_shape) == 4:
                interp_array = np.zeros(shape=(valid_msk.shape[0],
                                               valid_msk.shape[1],
                                               peek_var_shape[1]),
                                        dtype='float32')
                
            # Build interp_array by cumulatively adding spatially interpolated data at each 
            # timestep multiplied by the temporal weight for that timestep
            for time_ix, GEOSIT_fpath in enumerate(flist):

                with Dataset(GEOSIT_fpath) as nc:
                    temp = nc.variables[var_name_GEOSIT_dict[var_name]][:]
                    
                if len(temp.shape) == 3:
                    srcfield2D.data[:] = temp[0,:,:].transpose()
                    regrid(srcfield2D,dstfield2D)
                    output_shape = valid_msk.shape
                    var_interp_timestep = np.zeros(output_shape)
                    # Reshapes the regridded data to 2D
                    var_interp_timestep[valid_msk] = dstfield2D.data
                elif len(temp.shape) == 4:
                    tempt = np.transpose(temp)
                    srcfield3D.data[:] = tempt[:,:,:,0]
                    regrid(srcfield3D,dstfield3D)
                    output_shape = valid_msk.shape + dstfield3D.data.shape[-1:]
                    var_interp_timestep = np.zeros(output_shape)
                    # Reshapes the regridded data to 2D
                    var_interp_timestep[valid_msk, :] = dstfield3D.data
                                
                for atrack_ix, xtrack_ix in np.ndindex(valid_msk.shape):
                    interp_array[atrack_ix, xtrack_ix] += var_interp_timestep[atrack_ix, xtrack_ix] * \
                                                          time_weights[atrack_ix, xtrack_ix, time_ix]

            if var_name in ["land_surface_temp", "snow_cover"]:
                np.copyto(interp_array,
                          data_product_specs["Aux-Met"][var_name]["fill_value"],
                          where=interp_array > peek_var_input_fillvalue*1.e-4)

            # Add spatially & temporally interpolated array to interp_data dictionary
            interp_data[var_name] = interp_array
            
        print('Finished interp var: '+var_name+' for '+os.path.split(TIRS_L1B_fpath)[-1]+' at '+
              dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return interp_data


def run_interp_GEOSIT_ESMF_cube(TIRS_L1B_fpath, GEOSIT_asm_const_fpath,
                                GEOSIT_collection_flists,
                                variables,
                                GEOSIT_fname_cs_dict,
                                GEOSIT_ESDT_dict,
                                var_name_GEOSIT_dict,
                                time_weights_dict,
                                ESMF_script_fpath, work_dir,
                                atrack_range_to_process, data_product_specs):
    """
    Interpolate GEOS-IT *cubed sphere* data to TIRS scenes in space and time,
    using ESMF for spatial interpolation.
    
    Spatial interpolation works by calling an external script to run ESMF using
    MPI. The external script writes arrays of data spatially interpolated to 
    TIRS scenes, which are then read back in and temporal weights are applied 
    to get final interpolated values.
    
    Parameters
    ----------
    TIRS_L1B_fpath : str
        Filepath of TIRS L1B netCDF file.
    GEOSIT_asm_const_fpath : str
        Filepath of the constant-in-time GEOS-IT input file.
    GEOSIT_collection_flists : dict
        Dictionary containing the lists of files for each GEOS-IT collection that
        completely encompass the TIRS observation times. Only GEOS-IT collections
        containing data that we need (4 in total) are included.
    variables : list
        Names of variables for which interpolated data is needed.
    GEOSIT_fname_cs_dict : dict
        Dictionary mapping Aux-Met variable names to the corresponding GEOS-IT
        cubed-sphere file names.
    GEOSIT_ESDT_dict : dict
        Dictionary mapping Aux-Met variable names to GEOS-IT ESDTs.
    var_name_GEOSIT_dict : dict
        Dictionary mapping Aux-Met variable names to GEOS-IT variable names.        
    time_weights_dict : dict
        Interpolation weights for all GEOS-IT timesteps bounding the time span
        of the TIRS L1B file. Time weights are specific to each GEOS-IT collection.
    ESMF_script_fpath : str
        Filepath of external script that runs ESMF using MPI.
    work_dir : Work directory.
    atrack_range_to_process : 3-tuple (str, int, int)
        A 3-tuple containing the dimension name (in this case, "atrack") to
         subset, and start and stop indices (NumPy indexing convention) in the
         given granule to process
    data_product_specs : dict
        Dictionary of data product specifications

    Returns
    -------
    interp_data : dict
        Interpolated data for all variables.

    """

    cfg_JSON_fpath = os.path.join(work_dir, "cfg-GEOSIT_cubed_sphere.json")

    # Gather configuration info, and write it to a JSON-format temporary file:
    config = {}
    config["variables"] = variables
    config["GEOSIT_fname_cs_dict"] = GEOSIT_fname_cs_dict
    config["GEOSIT_asm_const_fpath"] = GEOSIT_asm_const_fpath
    config["GEOSIT_collection_flists"] = GEOSIT_collection_flists
    config["GEOSIT_ESDT_dict"] = GEOSIT_ESDT_dict
    config["var_name_GEOSIT_dict"] = var_name_GEOSIT_dict
    config["TIRS_L1B_fpath"] = TIRS_L1B_fpath
    config["work_dir"] = work_dir
    config["atrack_range_to_process"] = atrack_range_to_process
    config["sys_path"] = sys.path
    with open(cfg_JSON_fpath, 'w') as out_f:
        json.dump(config, out_f)

    # Determine path to mpirun and python executables, and run_ESMF_cube_mpi.py
    # Assumes that create_AUX_MET_product.py is being run in an environment that has
    #   mpirun installed
    py_path = shutil.which('python')
    mpi_path = shutil.which('mpirun')
    
    # Call 'ESMF_script_fpath' using MPI with number of parallel elements = 6
    # This script will save numpy arrays of interpolated data for each variable
    #  in the working directory
    cmd_list = [mpi_path, "-n", "6", py_path, ESMF_script_fpath, cfg_JSON_fpath]
    subprocess.check_call(cmd_list)

    tirs_shape = np.shape(time_weights_dict['GEOSIT_ASM_I3_L_V72'])[0:2]

    interp_data = {}
    
    # Now read interpolated data for each variable back in and apply time weights
    for var_name in variables:
        # Only a single timestep for land_fraction
        if var_name == 'land_fraction':
            regrid_array_fname = os.path.join(work_dir, var_name+'_regrid.npy')
            interp_data[var_name] = np.load(regrid_array_fname, allow_pickle=True)
        else:
            collection = GEOSIT_ESDT_dict[var_name]
            flist = GEOSIT_collection_flists[collection]
            time_weights = time_weights_dict[collection]

            peek_nc_fpath = flist[0]
           
            # "peek" variable is used to determine shape of variable data (lat, lon, nlevels)
            peek_nc = Dataset(peek_nc_fpath)
            peek_var_shape = peek_nc.variables[var_name_GEOSIT_dict[var_name]].shape
            peek_var_input_fillvalue = peek_nc. \
                                   variables[var_name_GEOSIT_dict[var_name]]. \
                                   getncattr("_FillValue")
            peek_nc.close()

            # Create arrays of zeros that will hold interpolated data for all timesteps
            # 2-D single-level variables
            if len(peek_var_shape) == 4:
                interp_array = np.zeros(shape=(tirs_shape),
                                        dtype='float32')
                
            # 3-D variables
            elif len(peek_var_shape) == 5:
                interp_array = np.zeros(shape=(tirs_shape[0],
                                               tirs_shape[1],
                                               peek_var_shape[1]),
                                        dtype='float32')
                
            # Build interp_array by cumulatively adding spatially interpolated data at each 
            # timestep multiplied by the temporal weight for that timestep
            for time_ix, GEOSIT_fpath in enumerate(flist):
                GEOSIT_fname = os.path.basename(GEOSIT_fpath)
                regrid_array_fname = os.path.join(work_dir,
                                       f"{var_name}_{GEOSIT_fname}_regrid.npy")
                var_interp_timestep = np.load(regrid_array_fname, allow_pickle=True)
                                
                for atrack_ix, xtrack_ix in np.ndindex(tirs_shape):
                    interp_array[atrack_ix, xtrack_ix] += var_interp_timestep[atrack_ix, xtrack_ix] * \
                                                          time_weights[atrack_ix, xtrack_ix, time_ix]

            if var_name in ["land_surface_temp", "snow_cover"]:
                np.copyto(interp_array,
                          data_product_specs["Aux-Met"][var_name]["fill_value"],
                          where=interp_array > peek_var_input_fillvalue*1.e-4)
            
            # Add spatially & temporally interpolated array to interp_data dictionary
            interp_data[var_name] = interp_array
            
        print('Finished interp var: '+var_name+' at '+dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return interp_data
