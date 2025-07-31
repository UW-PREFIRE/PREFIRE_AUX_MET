import numpy as np
import netCDF4
import os
import datetime

from PREFIRE_PRD_GEN.file_creation import write_data_fromspec
from PREFIRE_PRD_GEN.file_read import load_all_vars_of_nc4group, \
                                get_PREFIRE_Lx_field, load_all_atts_of_nc4group
import PREFIRE_AUX_MET.filepaths as AUX_MET_fpaths
from PREFIRE_tools.utils.filesys import mkdir_p


def load_TIRS_geo(TIRS_L1B_ds, atrack_range_to_process):
    """
    Load required geolocation data arrays from TIRS L1B file to prepare for 
    interpolation calculations. Arrays are:
       - lat of FOV center
       - lon of FOV center
       - FOV vertices lats/lons
       - mean surface elevation within FOV
       - land fraction within FOV
       - per-scene times

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
    TIRS_geo_data : dict
        Dictionary containing arrays of TIRS data from a given orbit file: 
        lat, lon, surface elevation, and time.   
    """
    # TIRS L1B arrays are shaped (nframes, 8)
    #   - nframes = number of frames in the along-track direction in an orbit file
    #   - 8 = number of scenes in the cross-track direction at a given along-track 
    #     position

    # Time may be shaped (nframes) if all 8 scenes in a frame are collected at 
    #   exactly the same time. The simulated TIRS L1B files are structured this way.
    #   However, this may not prove to be true in practice -- there may be a time
    #   offset between the 8 cross-track scenes at the same along-track position.
    
    #   - ** To prepare for this possibility, this function reads times provided 
    #     with (nframes,7) dimensions and returns times with (nframes,8,7) dimensions.
    #     Time values are replicated across the 8 cross track positions from the 
    #     single along-track value provided. The dimension with size 7 contains the
    #     components of times: [year, month, day, hour, minute, second, microsecond]
    lats_deg = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry", "latitude",
                                  atrack_range_to_process, as_type="float32")
    lons_deg = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry", "longitude",
                                  atrack_range_to_process, as_type="float32")
    
    sfc_elev_m = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry", "elevation",
                                    atrack_range_to_process)
    sfc_elev_m_fillvalue = TIRS_L1B_ds.groups["Geometry"]. \
                                  variables["elevation"].getncattr("_FillValue")
    
    land_fraction = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry",
                                      "land_fraction", atrack_range_to_process)

    vertex_lats_deg = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry",
                                    "vertex_latitude", atrack_range_to_process)
    vertex_lons_deg = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry",
                                   "vertex_longitude", atrack_range_to_process)

    # Times are provided as an array with dimensions (nframes, 7). The 7 values in 
    #   this array for each frame are: 
    #   [year, month, day, hour, minute, second, microsecond]

    times_atrack_UTC = get_PREFIRE_Lx_field(TIRS_L1B_ds, "Geometry", "time_UTC_values",
                                          atrack_range_to_process)

    times_allscenes_UTC = np.empty([np.shape(times_atrack_UTC)[0],8,7],
                                   dtype=times_atrack_UTC.dtype)
    
    for atrack_ix in np.arange(0, np.shape(times_allscenes_UTC)[0], 1):
        for xtrack_ix in np.arange(0, np.shape(times_allscenes_UTC)[1], 1):
            times_allscenes_UTC[atrack_ix,xtrack_ix] = times_atrack_UTC[atrack_ix]
            
    TIRS_geo_data = {
                     'lats_deg':lats_deg,
                     'lons_deg':lons_deg,
                     'sfc_elev_m':sfc_elev_m,
                     'sfc_elev_m_fillvalue':sfc_elev_m_fillvalue,
                     "land_fraction": land_fraction,
                     "vertex_lats_deg": vertex_lats_deg,
                     "vertex_lons_deg": vertex_lons_deg,
                     'times_UTC':times_allscenes_UTC
                     }
    
    return TIRS_geo_data


def load_GEOSIT_geo(GEOSIT_collection_flists):
    """
    Load required geolocation data arrays from GEOS-IT collection files to prepare 
    for interpolation calculations. Arrays are:
        - lats
        - lons
        - analysis times

    Parameters
    ----------
    GEOSIT_collection_flists : dict
        Dictionary containing the lists of files for each GEOS-IT collection that
        completely encompass the TIRS observation times. Only GEOS-IT collections
        containing data that we need (4 in total) are included.

    Returns
    -------
    GEOSIT_geo_data : dict
        Dictionary containing GEOS-IT constant lat / lon arrays and analysis times 
        for each GEOS-IT collection.
    
    """
    
    # GEOS-IT analysis time arrays have dimensions (ntimestamps, 7)
    #   - ntimestamps varies because of the varying analysis times of each GEOS-IT
    #     collection, and because time spans of ~95-minute TIRS L1B files cross
    #     varying combinations of GEOS-IT timestamps.
    #   - Array at each timestamp contains the components of times:
    #     [year, month, day, hour, minute, second, microsecond]
    
    GEOSIT_geo_data = {}
    
    for collection in GEOSIT_collection_flists.keys():
                
        flist = GEOSIT_collection_flists[collection]
        
        grid_type = flist[0].split('glo_')[1].split('_')[0]            
        
        # Only need to record lats/lons in output dictionary once, because they
        #   are the same across all GEOS-IT files
                        
        if 'lats' in GEOSIT_geo_data.keys():
            pass
        else:            
            geoloc_nc = netCDF4.Dataset(flist[0])

            if grid_type == 'C180x180x6':
                lats = geoloc_nc.variables['lats'][...]
                lons = geoloc_nc.variables['lons'][...]
            elif grid_type == 'L576x361':
                lats = geoloc_nc.variables['lat'][...]
                lons = geoloc_nc.variables['lon'][...]
            
            geoloc_nc.close()
    
            GEOSIT_geo_data['lats'] = lats
            GEOSIT_geo_data['lons'] = lons
            
        collection_times = np.empty([len(flist), 7], dtype='int16')
        
        for time_ix, fname in enumerate(flist):
            yr_str = fname[-23:-19]
            mth_str = fname[-18:-16]
            day_str = fname[-15:-13]
            hr_str = fname[-12:-10]
            min_str = fname[-10:-8]
            
            time_array = np.array(
                                  [int(yr_str),
                                   int(mth_str),
                                   int(day_str),
                                   int(hr_str),
                                   int(min_str),
                                   0,
                                   0],
                                   dtype='int16'
                                  )
                
            collection_times[time_ix] = time_array
        
        GEOSIT_geo_data[collection+'_time_UTC'] = collection_times
    
    return GEOSIT_geo_data
    

def write_AUXMET_product(
        TIRS_L1B_fpath, TIRS_L1B_ds, product_specs_fpath,
        AUXMET_output_dir, interp_data, artp,
        met_analysis_source, met_an_idstr, viirs_st_yr, bas_antarctic_version,
        product_full_version,
        ancsim_vars, ancsim_force_clearsky):
    """
    Write AUX-MET output NetCDF file using PREFIRE generic write_data_fromspec
    function.
    
    Parameters
    ----------
    TIRS_L1B_fpath : str
        Filepath of the TIRS Level-1B file corresponding to the AUX-MET file to
         be written.
    TIRS_L1B_ds : netCDF4.Dataset
        A netCDF4.Dataset object referencing the TIRS Level-1B data file.
    product_specs_fpath : str
        Filepath of the product data specification JSON-format file to be used
    AUXMET_output_dir : str
        Directory to which the AUX-MET NetCDF file will be written.
    interp_data : dict
        Dictionary containing arrays of met analysis data. This function can write
        data from any stage of AUX-MET processing, but in final version this should
        contain data interpolated spatially and temporally to TIRS scenes, with 
        vertical profile variables surface-corrected and interpolated to PCRTM 
        fixed-101 pressure levels.
    artp : 3-tuple (str, int, int)
        (optional) A 3-tuple containing the dimension name to subset, and start
         and stop indices (NumPy indexing convention) in the given granule
    met_analysis_source : str
        Meteorological analysis data source (e.g. GEOSIT_cubed_sphere)
    met_an_idstr : list of str
        Meteorological analysis identifier string(s) (e.g., GEOS-5.27.1 GEOS-IT)
    viirs_st_yr : str
        Year for which the VIIRS surface type input data is valid.
    bas_antarctic_version : str
        Version of the British Antarctic Survey medium resolution Antarctic
        coastlines dataset used to calculate Antarctic land and ice shelf fraction.
    product_full_version : str
        Full product version ID (e.g., "Pxx_Rzz").
    ancsim_vars : list
        Extra cloud-related variables output if code is run in ANC-SimTruth
        mode.
    ancsim_force_clearsky : bool
        Whether to zero out any cloud information, "forcing" an interpretation
        as clear-sky.
        
    Returns
    -------
    None.
    
    """

    # Collect/determine fields to be output:
    dat = {}
    dat["Aux-Met"] = interp_data
    
    # If code is run in ANC-SimTruth mode, extract ANC-SimTruth variables from
    # the Aux-Met output dictionary and place in a separate output dictionary.
    if ancsim_vars:
        dat["SimTruth"] = {}
        for v in ancsim_vars:
            dat["SimTruth"][v] = dat["Aux-Met"][v]
            del dat["Aux-Met"][v]
            
        other_cloud_vars = [
            'cloud_fraction_profile_correlated',
            'cloud_mask_profile_correlated',
            'cloud_flag',
            'cloud_od',
            'cloud_de',
            'cloud_dp'
            ]
        
        for v in other_cloud_vars:
            dat["SimTruth"][v] = dat["Aux-Met"][v]
            del dat["Aux-Met"][v]
            
    global_atts = {}

    # Load "Geometry" group and its group attributes from the Level-1B file:
    dat["Geometry"] = load_all_vars_of_nc4group("Geometry",TIRS_L1B_ds, artp)
    dat["Geometry_Group_Attributes"] = load_all_atts_of_nc4group("Geometry",
                                                                 TIRS_L1B_ds)

    atdim_full = TIRS_L1B_ds.dimensions[artp[0]].size

    global_atts["granule_ID"] = TIRS_L1B_ds.granule_ID

    global_atts["spacecraft_ID"] = TIRS_L1B_ds.spacecraft_ID
    global_atts["sensor_ID"] = TIRS_L1B_ds.sensor_ID
    global_atts["ctime_coverage_start_s"] = TIRS_L1B_ds.ctime_coverage_start_s
    global_atts["ctime_coverage_end_s"] = TIRS_L1B_ds.ctime_coverage_end_s
    global_atts["UTC_coverage_start"] = TIRS_L1B_ds.UTC_coverage_start
    global_atts["UTC_coverage_end"] = TIRS_L1B_ds.UTC_coverage_end
    global_atts["orbit_sim_version"] = TIRS_L1B_ds.orbit_sim_version

    with open(AUX_MET_fpaths.scipkg_prdgitv_fpath, 'r') as in_f:
        line_parts = in_f.readline().split('(', maxsplit=1)
        global_atts["provenance"] = "{}{} ( {}".format(line_parts[0],
                                                       product_full_version,
                                                       line_parts[1].strip())

    with open(AUX_MET_fpaths.scipkg_version_fpath) as f:
        global_atts["processing_algorithmID"] = f.readline().strip()

    L1B_fn = os.path.basename(TIRS_L1B_fpath)
    global_atts["input_product_files"] = L1B_fn

    global_atts["full_versionID"] = product_full_version
    global_atts["archival_versionID"] = (
                           product_full_version.split('_')[0].replace('R', ''))
    global_atts["netCDF_lib_version"] = netCDF4.getlibversion().split()[0]

    # Generate AUX-MET / ANC-SimTruth output file name:
    tokens = L1B_fn.split('_')

    if ancsim_vars:    
        fname_tmp = "PREFIRE_SAT{}_ANC-SimTruth_{}_{}_{}.nc".format(
            global_atts["spacecraft_ID"][-1], global_atts["full_versionID"],
            tokens[5], global_atts["granule_ID"])
        global_atts["summary"] = ("The PREFIRE AUX-MET product provides "
             "meteorological fields from an analysis data source (e.g., "
             "GEOS-IT) interpolated in space and time to PREFIRE TIRS scenes.")
    else:
        fname_tmp = "PREFIRE_SAT{}_AUX-MET_{}_{}_{}.nc".format(
            global_atts["spacecraft_ID"][-1], global_atts["full_versionID"],
            tokens[5], global_atts["granule_ID"])
        global_atts["summary"] = ("The PREFIRE ANC-SimTruth product provides "
             "meteorological fields from an analysis data source (e.g., "
             "GEOS-IT) interpolated in space and time to PREFIRE TIRS scenes.")

    if (artp[1] == 0) and (artp[2] is None or artp[2] == atdim_full-1):
        AUXMET_fname = "raw-"+fname_tmp
    else:
        if artp[2] is None:
            tmp_idx = atdim_full-1
        else:
            tmp_idx = artp[2]-1
        AUXMET_fname = "raw-"+fname_tmp[:-3]+ \
              f"-{artp[0]}_{artp[1]:05d}_{tmp_idx:05d}_of_{atdim_full:05d}f.nc"

    global_atts["file_name"] = AUXMET_fname

    auxmet_group_attrs = {
        'met_analysis_name': met_analysis_source,
        "met_analysis_ID": ', '.join(met_an_idstr),
        'VIIRS_annual_blended_surface_type_valid_year': viirs_st_yr,
        "British_Antarctic_Survey_med_res_Antarctic_coastlines_version": bas_antarctic_version
        }

    now_UTC_DT = datetime.datetime.now(datetime.timezone.utc)
    global_atts["UTC_of_file_creation"] = now_UTC_DT.strftime(
                                                        "%Y-%m-%dT%H:%M:%S.%f")

    if ancsim_force_clearsky:
        global_atts["additional_file_description"] = (
            "This granule was created in a fashion that ignores all cloud"
            "information, forcing a clear-sky interpretation for every FOV.")

    # Add global and group attributes to output dictionary:
    dat["Global_Attributes"] = global_atts
    dat['Aux-Met_Group_Attributes'] = auxmet_group_attrs

    AUXMET_fpath = os.path.join(AUXMET_output_dir, AUXMET_fname)
    mkdir_p(AUXMET_output_dir)

    # Use generic PREFIRE product writer to produce the AUX-MET / ANC-SimTruth
    #  output file:
    write_data_fromspec(dat, AUXMET_fpath, product_specs_fpath, verbose=True)
