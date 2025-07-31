"""
Functions to locate met analysis files near in time to a given TIRS orbit file

In operational practice, the SDPS glue code will do this for us. However,
 these functions are needed to be able to do non-SDPS testing.
"""

from datetime import timedelta
import glob
import os

import netCDF4

from PREFIRE_PRD_GEN.file_read import get_PREFIRE_Lx_field
from PREFIRE_tools.utils.aux import TIRS_L1B_time_array_to_dt


def _GEOSIT_collection_flist_fill(earliest_GEOSIT_time,
                                  latest_GEOSIT_time,
                                  fname_prefix,
                                  timestep, GEOSIT_parent_dir):
    """
    Helper to find_GEOSIT_files that takes the earliest and latest GEOS-IT
    analysis times needed for a given collection and fills in the full list of
    files for all times needed.

    Parameters
    ----------
    earliest_GEOSIT_time : datetime.datetime
        Earliest GEOS-IT time needed for data interpolation to TIRS scenes.
    latest_GEOSIT_time : datetime.datetime
        Latest GEOS-IT time needed for data interpolation to TIRS scenes.
    fname_prefix : str
        File name prefix for the given GEOS-IT collection.
    timestep : int
        Interval in hours between analysis times for this GEOS-IT collection.
    GEOSIT_parent_dir : str
        Parent directory housing GEOS-IT data.

    Returns
    -------
    GEOSIT_collection_flist : list
        List of all GEOS-IT filepaths needed for interpolation to TIRS scenes.
    
    """
    
    GEOSIT_collection_flist = []
    
    counter_dt = earliest_GEOSIT_time
    
    while counter_dt <= latest_GEOSIT_time:
        
        GEOSIT_collection_fname = fname_prefix+\
                                  counter_dt.strftime('%Y')+\
                                  '-'+\
                                  counter_dt.strftime('%m')+\
                                  '-'+\
                                  counter_dt.strftime('%d')+\
                                  'T'+\
                                  counter_dt.strftime('%H')+\
                                  counter_dt.strftime('%M')+\
                                  '.V01.nc4'

        # Try to find the filename assuming a flat directory structure:
        glob_str = os.path.join(GEOSIT_parent_dir, GEOSIT_collection_fname)
        fpath_glob = glob.glob(glob_str)
        if len(fpath_glob) == 0:
            # Find the filename assuming a day-of-year directory structure:
            glob_str = os.path.join(GEOSIT_parent_dir, '*',
                                    GEOSIT_collection_fname)
            fpath_glob = glob.glob(glob_str)
            if len(fpath_glob) == 0:
                raise FileNotFoundError("No GEOS-IT files found using the "
                                        "search string '{}'".format(glob_str))
        GEOSIT_collection_fpath = fpath_glob[0]
        
        GEOSIT_collection_flist.append(GEOSIT_collection_fpath)
        
        counter_dt += timedelta(hours=timestep)
    
    return GEOSIT_collection_flist
    

# ** Future note: this may need to be adapted for running on SDPS. If we need to
#   use a tool to find / download analysis files, then this function can be
#   adapted to return the earliest and latest date / time needed for each GEOS-IT
#   collection, and then fed to the tool to generate a list of files for each 
#   collection. Then the load_GEOSIT_geo function can take the list of 
#   files for each collection as input (instead of a dictionary).

#   In this case, the order of operations in the driver script would probably
#   resemble the following:
#   (1) Call the tool to find the TIRS L1B file of interest
#   (2) Run python script to find earliest and latest GEOS-IT files needed
#       from each collection
#   (3) Call the tool for each collection to generate a text file with
#       list of file names for each collection
#   (4) Use these file lists as input for the remainder of the product
#       generation workflow

def find_GEOSIT_files(TIRS_L1B_ds, analysis_source, met_analysis_dir, artp):
    """
    Given the path to a TIRS L1B netCDF file, find the GEOS-IT files for each 
    collection that are needed to completely encompass the TIRS observation
    times for data interpolation.

    Parameters
    ----------
    TIRS_L1B_ds : netCDF4.Dataset
        A netCDF4.Dataset object referencing the TIRS Level-1B data file.
    analysis_source : str
        Met analysis data source and grid type (e.g., GEOSIT_equal_angle).
    met_analysis_dir : str
        Parent directory housing meteorology analysis (GEOS-IT) data.
    artp : 3-tuple (str, int, int)
        (optional) A 3-tuple containing the dimension name to subset, and start
         and stop indices (NumPy indexing convention) in the given granule

    Returns
    -------
    A 3-tuple of:
    GEOSIT_collection_flists : dict
        Dictionary containing the lists of files for each GEOS-IT collection
        that completely encompass the TIRS observation times. Only GEOS-IT
        collections containing data that we need are included.
    GEOSIT_asm_const_fpath : str
        Filepath of the constant-in-time GEOS-IT input file.
    GEOSIT_idst : str
        Identification string from the GEOS-IT input data.
    
    """
    
    # The temporal matching of GEOS-IT met analysis data to TIRS measurements must 
    #   always be an *interpolation* with no *extrapolation* involved -- GEOS-IT
    #   files must completely encompass the time period of TIRS measurements in a
    #   given L1B file.
    
    # GEOS-IT files have different analysis times depending on which collection
    #   they are in:
    #   - GEOSIT_ASM_I1_L_SLV (2-D atmospheric data collection):
    #       1-hourly, instantaneous, top of the hour (0000, 0100, ..., 2300 UTC)
    #   - GEOSIT_OCN_T1_L_SLV (2-D ocean data collection):
    #       1-hourly, time-averaged, middle of the hour (0030, 0130, ..., 2330 UTC)
    #   - GEOSIT_LND_T1_L_SLV (2-D land data collection):
    #       1-hourly, time-averaged, middle of the hour (0030, 0130, ..., 2330 UTC)
    #   - GEOSIT_ASM_I3_L_V72 (3-D atmospheric data collection):
    #       3-hourly, instantaneous, top of the hour (0000, 0300, ..., 2100 UTC)
    
    # TIRS L1B files can be expected to span approximately 95 minutes (?)
    
    # So for example, a TIRS L1B orbit file with measurement times spanning from
    #   0057 to 0232 UTC would need the following GEOS-IT files for interpolation:
    #   - GEOSIT_ASM_I1_L_SLV: 0000, 0100, 0200, 0300 UTC
    #   - GEOSIT_OCN_T1_L_SLV: 0030, 0130, 0230, 0330 UTC
    #   - GEOSIT_LND_T1_L_SLV: 0030, 0130, 0230, 0330 UTC
    #   - GEOSIT_ASM_I3_L_V72: 0000, 0300 UTC
    
    if analysis_source == 'GEOSIT_cubed_sphere':
        grid_name = 'C180x180x6'
    elif analysis_source == 'GEOSIT_equal_angle':
        grid_name = 'L576x361'

    GEOSIT_idstr_set = set()

    # Find constant-in-time file:
    search_str = os.path.join(met_analysis_dir, "*asm_const_*nc4")
    glob_res = glob.glob(search_str)
    if len(glob_res) == 0:
        raise FileNotFoundError("No constant GEOS-IT file found using the "
                                "search string '{}'".format(search_str))
    GEOSIT_asm_const_fpath = glob_res[0]

    # Record GEOS-IT version string for this file:
    with netCDF4.Dataset(GEOSIT_asm_const_fpath, 'r') as nc_ds:
        GEOSIT_idstr_set.add(nc_ds.getncattr("Title"))

    earliest_TIRS_L1B_dt, latest_TIRS_L1B_dt = _TIRS_L1B_bookend_times_find(
                                                             TIRS_L1B_ds, artp)
    
    GEOSIT_collection_flists = {}
    
    # ---------- GEOSIT_ASM_I1_L_SLV ----------
    
    # Earliest GEOSIT_ASM_I1_L_SLV file needed is at the top of the hour preceding
    #   the earliest time in L1B file
    earliest_GEOSIT_ASM_I1_L_SLV_dt = earliest_TIRS_L1B_dt.replace(microsecond=0, second=0, minute=0)
    
    # Latest GEOSIT_ASM_I1_L_SLV file needed is at the top of the hour after
    #   the latest time in L1B file
    latest_dt_mins_to_60 = 60 - latest_TIRS_L1B_dt.minute
    latest_GEOSIT_ASM_I1_L_SLV_dt = latest_TIRS_L1B_dt.replace(microsecond=0, second=0) +\
                                    timedelta(minutes=latest_dt_mins_to_60)
    
    # Generate list of GEOSIT_ASM_I1_L_SLV files needed for the given L1B file
    GEOSIT_ASM_I1_L_SLV_prefix = (
                        f'GEOS.it.asm.asm_inst_1hr_glo_{grid_name}_slv.GEOS*.')
    GEOSIT_ASM_I1_L_SLV_timestep = 1

    GEOSIT_ASM_I1_L_SLV_flist = _GEOSIT_collection_flist_fill(
                earliest_GEOSIT_ASM_I1_L_SLV_dt, latest_GEOSIT_ASM_I1_L_SLV_dt,
                GEOSIT_ASM_I1_L_SLV_prefix, GEOSIT_ASM_I1_L_SLV_timestep,
                met_analysis_dir)

    # Record GEOS-IT version string(s) for this processing time interval:
    for in_fp in GEOSIT_ASM_I1_L_SLV_flist:
        with netCDF4.Dataset(in_fp, 'r') as nc_ds:
            GEOSIT_idstr_set.add(nc_ds.getncattr("Title"))

    # Add list of GEOSIT_ASM_I1_L_SLV files to output dictionary
    GEOSIT_collection_flists['GEOSIT_ASM_I1_L_SLV'] = GEOSIT_ASM_I1_L_SLV_flist
    
    
    # ---------- GEOSIT_OCN_T1_L_SLV and GEOSIT_LND_T1_L_SLV ----------
    
    # Earliest GEOSIT_OCN_T1_L_SLV / GEOSIT_LND_T1_L_SLV files needed are at the 
    #   middle of the hour preceding the earliest time in L1B file
    if earliest_TIRS_L1B_dt.minute >= 30:
        earliest_GEOSIT_OCN_T1_L_SLV_dt = earliest_TIRS_L1B_dt.replace(microsecond=0, second=0, minute=30)
    else:
        earliest_dt_mins_to_30 = 30 - earliest_TIRS_L1B_dt.minute
        earliest_GEOSIT_OCN_T1_L_SLV_dt = earliest_TIRS_L1B_dt.replace(microsecond=0, second=0) +\
                                          timedelta(minutes=earliest_dt_mins_to_30) -\
                                          timedelta(minutes=60)
    
    # Latest GEOSIT_OCN_T1_L_SLV / GEOSIT_LND_T1_L_SLV files needed are at the 
    #   middle of the hour after the latest time in L1B file
    if latest_TIRS_L1B_dt.minute < 30:
        latest_GEOSIT_OCN_T1_L_SLV_dt = latest_TIRS_L1B_dt.replace(microsecond=0, second=0, minute=30)
    else:
        latest_dt_mins_to_60 = 60 - latest_TIRS_L1B_dt.minute
        latest_GEOSIT_OCN_T1_L_SLV_dt = latest_TIRS_L1B_dt +\
                                        timedelta(minutes=latest_dt_mins_to_60) +\
                                        timedelta(minutes=30)

    # Generate lists of GEOSIT_OCN_T1_L_SLV and GEOSIT_LND_T1_L_SLV files needed 
    #   for the given L1B file
    GEOSIT_OCN_T1_L_SLV_prefix = (
                        f'GEOS.it.asm.ocn_tavg_1hr_glo_{grid_name}_slv.GEOS*.')
    GEOSIT_OCN_T1_L_SLV_timestep = 1

    GEOSIT_OCN_T1_L_SLV_flist = _GEOSIT_collection_flist_fill(
                earliest_GEOSIT_OCN_T1_L_SLV_dt, latest_GEOSIT_OCN_T1_L_SLV_dt,
                GEOSIT_OCN_T1_L_SLV_prefix, GEOSIT_OCN_T1_L_SLV_timestep,
                met_analysis_dir)

    # Record GEOS-IT version string(s) for this processing time interval:
    for in_fp in GEOSIT_OCN_T1_L_SLV_flist:
        with netCDF4.Dataset(in_fp, 'r') as nc_ds:
            GEOSIT_idstr_set.add(nc_ds.getncattr("Title"))

    GEOSIT_LND_T1_L_SLV_prefix = (
                        f'GEOS.it.asm.lnd_tavg_1hr_glo_{grid_name}_slv.GEOS*.')
    
    # "LND" collection has the same time parameters as "OCN" collection
    GEOSIT_LND_T1_L_SLV_flist = _GEOSIT_collection_flist_fill(
                earliest_GEOSIT_OCN_T1_L_SLV_dt, latest_GEOSIT_OCN_T1_L_SLV_dt,
                GEOSIT_LND_T1_L_SLV_prefix, GEOSIT_OCN_T1_L_SLV_timestep,
                met_analysis_dir)

    # Record GEOS-IT version string(s) for this processing time interval:
    for in_fp in GEOSIT_LND_T1_L_SLV_flist:
        with netCDF4.Dataset(in_fp, 'r') as nc_ds:
            GEOSIT_idstr_set.add(nc_ds.getncattr("Title"))

    # Add lists of GEOSIT_OCN_T1_L_SLV and GEOSIT_LND_T1_L_SLV files to output dictionary
    GEOSIT_collection_flists['GEOSIT_OCN_T1_L_SLV'] = GEOSIT_OCN_T1_L_SLV_flist
    GEOSIT_collection_flists['GEOSIT_LND_T1_L_SLV'] = GEOSIT_LND_T1_L_SLV_flist
    
    
    # ---------- GEOSIT_ASM_I3_L_V72 ----------

    # Earliest GEOSIT_ASM_I3_L_V72 file needed is at the top of the hour at the 
    #   3-hourly timestep preceding the earliest time in L1B file
        
    if earliest_TIRS_L1B_dt.hour == 0:
            earliest_GEOSIT_ASM_I3_L_V72_dt = earliest_TIRS_L1B_dt.\
                                              replace(microsecond=0, second=0, minute=0)   
    else:
        if earliest_TIRS_L1B_dt.hour % 3 == 0:
            earliest_GEOSIT_ASM_I3_L_V72_dt = earliest_TIRS_L1B_dt.\
                                              replace(microsecond=0, second=0, minute=0, \
                                              hour=earliest_TIRS_L1B_dt.hour)
        elif earliest_TIRS_L1B_dt.hour % 3 == 1:
            earliest_GEOSIT_ASM_I3_L_V72_dt = earliest_TIRS_L1B_dt.\
                                              replace(microsecond=0, second=0, minute=0, \
                                              hour=earliest_TIRS_L1B_dt.hour - 1)
        elif earliest_TIRS_L1B_dt.hour % 3 == 2:
            earliest_GEOSIT_ASM_I3_L_V72_dt = earliest_TIRS_L1B_dt.\
                                              replace(microsecond=0, second=0, minute=0, \
                                              hour=earliest_TIRS_L1B_dt.hour - 2)

    # Latest GEOSIT_ASM_I3_L_V72 file needed is at the top of the hour at the 
    #   3-hourly timestep after the latest time in L1B file
    
    if latest_TIRS_L1B_dt.hour < 21:
        if latest_TIRS_L1B_dt.hour % 3 == 0:
            latest_GEOSIT_ASM_I3_L_V72_dt = latest_TIRS_L1B_dt.\
                                            replace(microsecond=0, second=0, minute=0, \
                                            hour=latest_TIRS_L1B_dt.hour + 3)
        elif latest_TIRS_L1B_dt.hour % 3 == 1:
            latest_GEOSIT_ASM_I3_L_V72_dt = latest_TIRS_L1B_dt.\
                                            replace(microsecond=0, second=0, minute=0, \
                                            hour=latest_TIRS_L1B_dt.hour + 2)
        elif latest_TIRS_L1B_dt.hour % 3 == 2:
            latest_GEOSIT_ASM_I3_L_V72_dt = latest_TIRS_L1B_dt.\
                                            replace(microsecond=0, second=0, minute=0, \
                                            hour=latest_TIRS_L1B_dt.hour + 1)
    else:
        latest_GEOSIT_ASM_I3_L_V72_dt = latest_TIRS_L1B_dt.replace(microsecond=0, second=0, minute=0, hour=0) +\
                                        timedelta(days=1)

    # Generate list of GEOSIT_ASM_I3_L_V72 files needed for the given L1B file
    GEOSIT_ASM_I3_L_V72_prefix = (
                        f'GEOS.it.asm.asm_inst_3hr_glo_{grid_name}_v72.GEOS*.')
    GEOSIT_ASM_I3_L_V72_timestep = 3

    GEOSIT_ASM_I3_L_V72_flist = _GEOSIT_collection_flist_fill(
                earliest_GEOSIT_ASM_I3_L_V72_dt, latest_GEOSIT_ASM_I3_L_V72_dt,
                GEOSIT_ASM_I3_L_V72_prefix, GEOSIT_ASM_I3_L_V72_timestep,
                met_analysis_dir)

    # Record GEOS-IT version string(s) for this processing time interval:
    for in_fp in GEOSIT_ASM_I3_L_V72_flist:
        with netCDF4.Dataset(in_fp, 'r') as nc_ds:
            GEOSIT_idstr_set.add(nc_ds.getncattr("Title"))

    # Add list of GEOSIT_ASM_I3_L_V72 files to output dictionary        
    GEOSIT_collection_flists['GEOSIT_ASM_I3_L_V72'] = GEOSIT_ASM_I3_L_V72_flist

    GEOSIT_idstr = sorted(GEOSIT_idstr_set)

    return (GEOSIT_collection_flists, GEOSIT_asm_const_fpath, GEOSIT_idstr)


def _TIRS_L1B_bookend_times_find(TIRS_L1B_ds, atrack_range_to_process):
    """
    Determine the earliest and latest scene times in a TIRS Level-1B orbit 
    file and return them as datetime objects
    
    Parameters
    ----------
    TIRS_L1B_ds : netCDF4.Dataset
        A netCDF4.Dataset object referencing the TIRS Level-1B data file.
    atrack_range_to_process : 3-tuple (str, int, int)
        A 3-tuple containing the dimension name (in this case, "atrack") to
         subset, and start and stop indices (NumPy indexing convention) in the
         given granule to process

    Returns
    -------
    earliest_TIRS_L1B_dt : datetime.datetime
        Earliest scene time in the relevant portion of the TIRS Level-1B file.
    latest_TIRS_L1B_dt : datetime.datetime
        Latest scene time in the relevant portion of the TIRS Level-1B file.

    """

    # Get earliest and latest time from L1B file
    earliest_time_array = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry",
                                        "time_UTC_values", atrack_range_to_process)[0]
    latest_time_array = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry",
                                       "time_UTC_values", atrack_range_to_process)[-1]
    
    earliest_TIRS_L1B_dt = TIRS_L1B_time_array_to_dt(earliest_time_array)
    latest_TIRS_L1B_dt = TIRS_L1B_time_array_to_dt(latest_time_array)
    
    return earliest_TIRS_L1B_dt, latest_TIRS_L1B_dt
