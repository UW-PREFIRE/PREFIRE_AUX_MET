#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import esmpy
import sys
import warnings
import glob
import os
import argparse
import json


# Ignore warning that ESMP_GridCreateCubedSphere is a beta function
warnings.simplefilter("ignore", category=UserWarning)


def create_locstream_TIRS_mpi(file_name, atrack_range_to_process):
    """
    load lat/lon from PREFIRE/TIRS Geometry group and convert to EMSF LocStream object.
    
    lats, lons, msk are assumed to be the same shape, any dimension.
    msk is a boolean masking array with True for good indices.

    LocStream only takes 1D vectors of lat/lon, so lat/lon are
    basically flattened by the msk, before storing in the LocStream
    
    All points must go on processor 0 for ESMF cubed sphere interpolation
    run with MPI.
    """

    with Dataset(file_name, 'r') as TIRS_L1B_ds:
        TIRS_L1B_ds.set_auto_mask(False)
        lats_unmsk = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry", "latitude",
                                        atrack_range_to_process)
        lons_unmsk = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry", "longitude",
                                        atrack_range_to_process)
        valid_msk = lats_unmsk != -9999.0

    lats = lats_unmsk[valid_msk]
    lons = lons_unmsk[valid_msk]

    # requires specific numeric type
    lats = lats.astype(np.float64)
    lons = lons.astype(np.float64)

    npoints = lats.shape[0]
    coord_sys = esmpy.CoordSys.SPH_DEG

    # Ben Auer's comment:
    # for technical reason all your points go on processor 0 but I have to
    # create this object on all processor which is too bad so we
    # just won't use the results on processors 1 to 5
    if esmpy.local_pet() == 0:
        locstream = esmpy.LocStream(npoints, coord_sys=coord_sys)
        locstream["ESMF:Lon"] = lons
        locstream["ESMF:Lat"] = lats
    else:
        locstream = esmpy.LocStream(1, coord_sys=coord_sys)
        locstream["ESMF:Lon"] = [0.0]
        locstream["ESMF:Lat"] = [0.0]

    return locstream, valid_msk


def write_output_var(var_name, data, valid_msk, src_fname=None):
    """
    helper function to reshape the data back to 2D, accounting for
    bad values which are marked as zeros in valid_mask.
    the reshaped data is written to the h5 file object.
    """
    if data.ndim == 1:
        output_shape = valid_msk.shape
        out_array = np.zeros(output_shape)
        out_array[valid_msk] = data
    else:
        output_shape = valid_msk.shape + data.shape[-1:]
        out_array = np.zeros(output_shape)
        out_array[valid_msk, :] = data
    
    if var_name == 'land_fraction':
        # np.save(var_name+'_regrid.npy', out_array)
        out_array.dump(os.path.join(work_dir, var_name+'_regrid.npy'))
    else:
        # np.save(var_name+'_'+src_fname+'_regrid.npy', out_array)
        out_array.dump(os.path.join(work_dir,
                       var_name+'_'+os.path.basename(src_fname)+'_regrid.npy'))


if __name__ == "__main__":
    # Schematic command-line invocation of this script:
    #  mpirun -n 6 python  run_ESMF_cube_mpi-GEOSIT.py  cfg_JSON_fpath

    # Process arguments:
    arg_description = "Interpolate GEOS-IT *cubed sphere* data to TIRS " \
                      "scenes in space and time, using ESMF for spatial " \
                      "interpolation."
    arg_parser = argparse.ArgumentParser(description=arg_description)
    arg_parser.add_argument("cfg_JSON_fpath",
           help="Filepath of JSON-format file with configuration information.")
    args = arg_parser.parse_args()

    # Read configuration info:
    with open(args.cfg_JSON_fpath, 'r') as in_f:
        config = json.load(in_f)

    sys.path = config["sys_path"]
    from PREFIRE_PRD_GEN.file_read import get_PREFIRE_Lx_field

    TIRS_L1B_fpath = config["TIRS_L1B_fpath"]
    work_dir = config["work_dir"]
    variables = config["variables"]
    GEOSIT_asm_const_fpath = config["GEOSIT_asm_const_fpath"]
    GEOSIT_collection_flists = config["GEOSIT_collection_flists"]
    GEOSIT_ESDT_dict = config["GEOSIT_ESDT_dict"]
    var_name_GEOSIT_dict = config["var_name_GEOSIT_dict"]
    atrack_range_to_process = config["atrack_range_to_process"]

    locstream, valid_msk = create_locstream_TIRS_mpi(TIRS_L1B_fpath,
                                                     atrack_range_to_process)
    
    # Use a vertical profile file to determine grid characteristics, including
    # number of vertical levels.
    with Dataset(GEOSIT_collection_flists['GEOSIT_ASM_I3_L_V72'][0],
                 'r') as grid_nc:
        im_world = len(grid_nc.dimensions['Xdim'])
        lm = len(grid_nc.dimensions['lev'])
    grid = esmpy.Grid(tilesize=im_world)
    
    srcfield2D = esmpy.Field(grid,name="srcfield2D",typekind=esmpy.TypeKind.R4)
    dstfield2D = esmpy.Field(locstream,name="dstfield2D",typekind=esmpy.TypeKind.R4)
    srcfield3D = esmpy.Field(grid,name="srcfield3D",ndbounds=[lm],typekind=esmpy.TypeKind.R4)
    dstfield3D = esmpy.Field(locstream,name="dstfield3D",ndbounds=[lm],typekind=esmpy.TypeKind.R4)
    # Bilinear interpolation
    regrid = esmpy.Regrid(srcfield2D,dstfield2D,regrid_method=esmpy.RegridMethod.BILINEAR)

    localpet = esmpy.local_pet()
    
    # Land fraction is a special case since it comes from single time-invariant
    # file, and FRLAND + FRLANDICE must be added together before interpolation
    with Dataset(GEOSIT_asm_const_fpath) as const_nc:
        temp = const_nc.variables['FRLAND'][:] + const_nc.variables['FRLANDICE'][:]
        srcfield2D.data[:] = np.transpose(temp[0,localpet,:,:])
        regrid(srcfield2D,dstfield2D)
        if localpet == 0:
            write_output_var('land_fraction', dstfield2D.data, valid_msk)

    for var_name in variables:
        if var_name == "land_fraction":
            continue

        # List of all relevant GEOS-IT netCDF files in that contain the given
        #  variable:
        ESDT_mnk = GEOSIT_ESDT_dict[var_name]
        var_ncs = GEOSIT_collection_flists[ESDT_mnk]
        
        for src_fname in var_ncs:
            geosit_nc = Dataset(src_fname)
            temp = geosit_nc.variables[var_name_GEOSIT_dict[var_name]][:]
            dim_size = len(temp.shape)
            # Cubed sphere splits the problem by the 6 cube faces (note the localpet
            # index is used in the 'nface' dimension)
            if len(temp.shape) == 4:
                srcfield2D.data[:] = np.transpose(temp[0,localpet,:,:])
                regrid(srcfield2D,dstfield2D)
                if localpet == 0:
                    write_output_var(var_name, dstfield2D.data, valid_msk, src_fname=src_fname)
            elif len(temp.shape) == 5:
                tempt = np.transpose(temp)
                srcfield3D.data[:] = tempt[:,:,localpet,:,0]
                regrid(srcfield3D,dstfield3D)
                if localpet == 0:
                    write_output_var(var_name, dstfield3D.data, valid_msk, src_fname=src_fname)
            
            geosit_nc.close()
