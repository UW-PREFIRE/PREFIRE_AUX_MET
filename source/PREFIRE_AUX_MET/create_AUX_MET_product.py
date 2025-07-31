"""
Create AUX-MET / ANC-SimTruth product corresponding to a TIRS Level-1B file.

This program requires python version 3.6 or later, and is importable as a
python module.
"""

from pathlib import Path
import numpy as np
import sys
import os
import glob
import json
import netCDF4 as nc4
import datetime as dt
import warnings

from PREFIRE_AUX_MET.utils.classify_sfc_type_prelim import \
    calc_VIIRS_IGBP_types, calc_antarctic_overlap, calc_sfc_type_prelim

from PREFIRE_AUX_MET.utils.met_files_find import find_GEOSIT_files
from PREFIRE_AUX_MET.utils.data_IO import load_TIRS_geo, load_GEOSIT_geo, \
                                          write_AUXMET_product
from PREFIRE_AUX_MET.utils.interp_internal import compute_time_weights_GEOSIT,\
                                                  compute_spatial_weights_GEOSIT,\
                                                  apply_interp_weights_GEOSIT,\
                                                  land_frac_interp_GEOSIT_internal
from PREFIRE_AUX_MET.utils.interp_ESMF import run_interp_GEOSIT_ESMF_cube,\
                                              interp_GEOSIT_ESMF_ea
from PREFIRE_AUX_MET.utils.surf_correction import apply_surf_correction
from PREFIRE_AUX_MET.utils.CO2_CH4_model_eval import apply_CO2_CH4_model
from PREFIRE_sim_tools.utils.level_interp import pressure_interp
import PREFIRE_sim_tools.paths
import PREFIRE_AUX_MET.filepaths

# Only used in ANC-SimTruth mode
from PREFIRE_AUX_MET.utils.cloud_mask import apply_cloud_mask, add_estimated_cloudprops


kg_to_g = 1000.

variables = [
             'land_surface_temp',
             'skin_temp',
             'temp_2m',
             'temp_10m',
             'surface_phi',
             'seaice_concentration',
             'snow_cover',
             'surface_pressure',
             'temp_profile',
             'wv_profile',
             'total_column_wv',
             'o3_profile',
             'altitude_profile',
             'u_10m',
             'u_profile',
             'v_10m',
             'v_profile',
             'omega_profile'
            ]

ancsim_variables = ['qi_profile','ql_profile','cloud_fraction_profile']


def process_GEOSIT(ancillary_Path, met_analysis_dir, work_dir, TIRS_L1B_ds,
                   TIRS_L1B_fpath, met_analysis_source, interp_method,
                   TIRS_geo_data, artp, data_product_specs,
                   ancsim_mode=False):
    """Process GEOS-IT """

    GEOSIT_ESDT_dict = {
        'land_surface_temp':'GEOSIT_LND_T1_L_SLV',
        'skin_temp':'GEOSIT_ASM_I1_L_SLV',
        'temp_2m':'GEOSIT_ASM_I1_L_SLV',
        'temp_10m':'GEOSIT_ASM_I1_L_SLV',
        'surface_phi':'GEOSIT_ASM_I3_L_V72',
        'seaice_concentration':'GEOSIT_OCN_T1_L_SLV',
        'snow_cover':'GEOSIT_LND_T1_L_SLV',
        'surface_pressure':'GEOSIT_ASM_I1_L_SLV',
        'temp_profile':'GEOSIT_ASM_I3_L_V72',
        'wv_profile':'GEOSIT_ASM_I3_L_V72',
        'total_column_wv':'GEOSIT_ASM_I1_L_SLV',
        'o3_profile':'GEOSIT_ASM_I3_L_V72',
        'altitude_profile':'GEOSIT_ASM_I3_L_V72',
        'u_10m':'GEOSIT_ASM_I1_L_SLV',
        'u_profile':'GEOSIT_ASM_I3_L_V72',
        'v_10m':'GEOSIT_ASM_I1_L_SLV',
        'v_profile':'GEOSIT_ASM_I3_L_V72',
        'omega_profile':'GEOSIT_ASM_I3_L_V72',
        # These variables will only be populated in ANC-SimTruth
        # mode. Only support ANC-SimTruth variables for GEOS-IT
        # cubed sphere data for now.
        'qi_profile':'GEOSIT_ASM_I3_L_V72',
        'ql_profile':'GEOSIT_ASM_I3_L_V72',
        'cloud_fraction_profile':'GEOSIT_ASM_I3_L_V72'
        }

    GEOSIT_fname_cs_dict = {
        'land_surface_temp':'lnd_tavg_1hr_glo_C180x180x6_slv',
        'skin_temp':'asm_inst_1hr_glo_C180x180x6_slv',
        'temp_2m':'asm_inst_1hr_glo_C180x180x6_slv',
        'temp_10m':'asm_inst_1hr_glo_C180x180x6_slv',
        'surface_phi':'asm_inst_3hr_glo_C180x180x6_v72',
        'seaice_concentration':'ocn_tavg_1hr_glo_C180x180x6_slv',
        'snow_cover':'lnd_tavg_1hr_glo_C180x180x6_slv',
        'surface_pressure':'asm_inst_1hr_glo_C180x180x6_slv',
        'temp_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        'wv_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        'total_column_wv':'asm_inst_1hr_glo_C180x180x6_slv',
        'o3_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        'altitude_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        'u_10m':'asm_inst_1hr_glo_C180x180x6_slv',
        'u_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        'v_10m':'asm_inst_1hr_glo_C180x180x6_slv',
        'v_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        'omega_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        # These variables will only be populated in ANC-SimTruth
        # mode. Only support ANC-SimTruth variables for GEOS-IT
        # cubed sphere data for now.
        'qi_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        'ql_profile':'asm_inst_3hr_glo_C180x180x6_v72',
        'cloud_fraction_profile':'asm_inst_3hr_glo_C180x180x6_v72'
        }

    GEOSIT_var_name_dict = {
        'land_surface_temp':'TSURF',
        'skin_temp':'TS',
        'temp_2m':'T2M',
        'temp_10m':'T10M',
        'surface_phi':'PHIS',
        'seaice_concentration':'FRSEAICE',
        'snow_cover':'FRSNO',
        'surface_pressure':'PS',
        'temp_profile':'T',
        'wv_profile':'QV',
        'total_column_wv':'TQV',
        'o3_profile':'O3',
        'altitude_profile':'H',
        'u_10m':'U10M',
        'u_profile':'U',
        'v_10m':'V10M',
        'v_profile':'V',
        'omega_profile':'OMEGA',
        # These variables will only be populated in ANC-SimTruth
        # mode.
        'qi_profile':'QI',
        'ql_profile':'QL',
        'cloud_fraction_profile':'CLOUD'
        }

    GEOS5_coef_fpath = str(ancillary_Path / "geos5_hybrid_sigma_72_double.txt")

    GEOSIT_collection_flists, GEOSIT_asm_const_fpath, GEOSIT_idstr = \
                            find_GEOSIT_files(TIRS_L1B_ds, met_analysis_source,
                                              met_analysis_dir, artp)

    GEOSIT_geo_data = load_GEOSIT_geo(GEOSIT_collection_flists)

    time_weights_dict = compute_time_weights_GEOSIT(TIRS_geo_data,
                                                    GEOSIT_geo_data)

    # Only support GEOS-IT cubed-sphere data and ESMF interpolation for
    # ANC-SimTruth variables (can change later if needed)
    ancsim_vars = None  # Default

    if met_analysis_source == "GEOSIT_equal_angle" and \
          interp_method == "internal":
        spatial_indices, spatial_weights = \
                                compute_spatial_weights_GEOSIT(TIRS_geo_data,
                                                               GEOSIT_geo_data)
        interp_data_uncorr = \
            apply_interp_weights_GEOSIT(
                GEOSIT_collection_flists,
                variables,
                GEOSIT_ESDT_dict,
                GEOSIT_var_name_dict,
                spatial_indices,
                spatial_weights,
                time_weights_dict,
                data_product_specs
                )
        # Add GEOS-IT interpolated land fraction to output data
        interp_data_uncorr["land_fraction"] = \
                         land_frac_interp_GEOSIT_internal(
                                          GEOSIT_asm_const_fpath,
                                          spatial_indices, spatial_weights)

    elif met_analysis_source == "GEOSIT_equal_angle" and \
          interp_method == "ESMF":
        # ESMF version of code interpolates land fraction alongside other 
        # variables (not in separate function like "internal" version of code)
        variables.append("land_fraction")
        interp_data_uncorr = interp_GEOSIT_ESMF_ea(
            TIRS_L1B_ds, TIRS_L1B_fpath, GEOSIT_asm_const_fpath,
            GEOSIT_collection_flists,
            variables,
            GEOSIT_ESDT_dict,
            GEOSIT_var_name_dict,
            time_weights_dict,
            work_dir,
            artp, data_product_specs
            )
    elif met_analysis_source == "GEOSIT_cubed_sphere" and \
         interp_method == "ESMF":
        # ESMF version of code interpolates land fraction alongside other 
        # variables (not in separate function like "internal" version of code)
        variables.append("land_fraction")
        ESMF_script_path = os.path.join(PREFIRE_AUX_MET.filepaths.utils_dir,
                                        "run_ESMF_cube_mpi-GEOSIT.py")
        
        # Only support GEOS-IT cubed-sphere data and ESMF interpolation for
        # ANC-SimTruth variables (can change later if needed)
        if ancsim_mode:
            ancsim_vars = ancsim_variables
            variables.extend(ancsim_vars)

        interp_data_uncorr = \
            run_interp_GEOSIT_ESMF_cube(
                TIRS_L1B_fpath,
                GEOSIT_asm_const_fpath,
                GEOSIT_collection_flists,
                variables,
                GEOSIT_fname_cs_dict,
                GEOSIT_ESDT_dict,
                GEOSIT_var_name_dict,
                time_weights_dict,
                ESMF_script_path, work_dir,
                artp, data_product_specs
                )
        
    # Surface correction function requires water vapor in kg/kg, so convert
    #  to g/kg (to match file spec and PCRTM input) *after* surface correction
    #  (below):
    interp_data_sfc_corr = apply_surf_correction(interp_data_uncorr,
                                                 TIRS_geo_data["sfc_elev_m"],
                                         TIRS_geo_data["sfc_elev_m_fillvalue"],
                                                 GEOS5_coef_fpath,
                                                 data_product_specs,
                                                 ancsim_vars)

    t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Applied surface correction at {t_now}')

    # Convert wv_profile from kg/kg to g/kg
    FillValue = data_product_specs["Aux-Met"]["wv_profile"]["fill_value"]
    interp_data_sfc_corr["wv_profile"] = \
                    np.where(np.logical_not( \
                                 np.isclose(interp_data_sfc_corr["wv_profile"],
                                            [FillValue])),
                               interp_data_sfc_corr["wv_profile"]*kg_to_g,
                               FillValue)

    # Convert o3_profile from kg/kg to ppm
    M_d = 28.87  # [amu] dry air
    M_O3 = 48.0  # [amu] ozone
    FillValue = data_product_specs["Aux-Met"]["o3_profile"]["fill_value"]
    interp_data_sfc_corr["o3_profile"] = \
                    np.where(np.logical_not( \
                                 np.isclose(interp_data_sfc_corr["o3_profile"],
                                            [FillValue])),
                            interp_data_sfc_corr["o3_profile"]*(M_d/M_O3)*1.e6,
                            FillValue)
                        
    # If ANC-SimTruth mode, convert qi and ql from kg/kg to g/kg
    if ancsim_mode:
        interp_data_sfc_corr["qi_profile"] = interp_data_sfc_corr["qi_profile"]*kg_to_g
        interp_data_sfc_corr["ql_profile"] = interp_data_sfc_corr["ql_profile"]*kg_to_g

    return (GEOSIT_collection_flists, GEOSIT_asm_const_fpath,
            interp_data_sfc_corr, GEOSIT_idstr, ancsim_vars)


def create_AUX_MET_product(TIRS_L1B_fpath, AUX_MET_output_dir, tmpfiles_dir,
                           met_analysis_source, interp_method, ancillary_dir,
                           met_analysis_dir, product_full_version,
                           ancsim_mode=False, ancsim_force_clearsky=False,
                           ancsim_force_noseaice=False,
                           ancsim_force_allseaice=False,
                           atrack_range_to_process=("atrack",0,None)):
    """
    Create AUX-MET data product from all or some of a TIRS Level-1B file.

    Parameters
    ----------
    TIRS_L1B_fpath : str
        Path to TIRS Level-1B file.
    AUX_MET_output_dir : str
        Directory to which AUX-MET NetCDF-format file(s) will be written.
    tmpfiles_dir : str
        Directory to which temporary files will be written during processing.
    met_analysis_source : str
        Meteorological analysis dataset moniker (currently "GEOSIT_equal_angle"
         and "GEOSIT_cubed_sphere" are supported).
    interp_method : str
        Method to use for interpolation.  Either "internal" (not for
         cubed-sphere analyses) or "ESMF".
    ancillary_dir : str
        Directory in which ancillary files are located.
    met_analysis_dir : str
        Directory in which the meteorological analysis dataset is located.
    product_full_version : str
        Full product version ID (e.g., "Pxx_Rzz").
    ancsim_mode : bool
        Whether to run Aux-Met code in ANC-SimTruth mode, which outputs extra
        cloud fields for use in simulated data.
    ancsim_force_clearsky : bool
        Whether to zero out any cloud information, "forcing" an interpretation
        as clear-sky.
    ancsim_force_noseaice : bool
        Whether to set output 'merged_surface_type_prelim' values of 2 or 3
        (sea-ice or partial-sea-ice, respectively) to 1 (open water) for all
        FOVs, *and* then set the sea-ice concentration array to zero for all
        FOVs (since non-zero values of that array can occur even for
        'merged_surface_type_prelim' values of 1 or 4-8, due to heterogeneous
        scenes).
    ancsim_force_allseaice : bool
        Whether to set output 'merged_surface_type_prelim' values of 1 or 3
        (open water or partial-sea-ice, respectively) to 2 (sea-ice) for all
        FOVs, *and* then set the sea-ice concentration array values to one for
        all FOVs with a 'merged_surface_type_prelim' value of 2 (sea-ice) and
        to zero for all other FOVs (since non-zero values of that array can
        occur even for 'merged_surface_type_prelim' values of 1 or 4-8, due to
        heterogeneous scenes).
    atrack_range_to_process : 3-tuple (str, int, int)
        (optional) A 3-tuple containing the dimension name to subset, and start
         and stop indices (NumPy indexing convention) in the given granule

    Returns
    -------
    None.

    """
    t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tirs_fname = os.path.basename(TIRS_L1B_fpath)
    print(f'Started {tirs_fname} at {t_now}')

    # Set ancillary filepaths:
    ancillary_Path = Path(ancillary_dir)
    CO2_modelfit_fpath = str(ancillary_Path / "CO2_model_spline_tck.h5")
    CH4_modelfit_fpath = str(ancillary_Path / "CH4_model_spline_tck.h5")
    product_specs_fpath = str(ancillary_Path / "Aux-Met_product_filespecs.json")

    tmpfiles_Path = Path(tmpfiles_dir) / Path(TIRS_L1B_fpath).stem
    if not os.path.exists(tmpfiles_Path):
        os.mkdir(tmpfiles_Path)

    # Open TIRS Level-1B file -- do only once, use the Dataset object elsewhere
    TIRS_L1B_ds = nc4.Dataset(TIRS_L1B_fpath)
    
    # Read geometry/timing info:
    TIRS_geo_data = load_TIRS_geo(TIRS_L1B_ds, atrack_range_to_process)

    t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Read TIRS geoloc data at {t_now}')
    
    # Determine which year's VIIRS annual surface type file matches the times 
    # in PREFIRE granule
    # - To account for granules that span the boundary across 2 years, use the
    #   time of TIRS scene in the middle of the granule to match to VIIRS year
    #   (and BAS date range)
    TIRS_midpt_UTC_parts = \
        TIRS_geo_data['times_UTC'][int(TIRS_geo_data['times_UTC'].shape[0]/2),3,:]
    TIRS_midpt_t = dt.datetime(
        TIRS_midpt_UTC_parts[0],
        TIRS_midpt_UTC_parts[1],
        TIRS_midpt_UTC_parts[2],
        TIRS_midpt_UTC_parts[3],
        TIRS_midpt_UTC_parts[4],
        TIRS_midpt_UTC_parts[5],
        TIRS_midpt_UTC_parts[6],
        )
    TIRS_midpt_yr = TIRS_midpt_t.year
    
    # Check for VIIRS annual surface type file from the year of PREFIRE
    # granule. If not present, move on to the previous year, then two years
    # prior, and finally raise exception if none of these files are found.
    viirs_ast_yrs = [TIRS_midpt_yr, TIRS_midpt_yr-1, TIRS_midpt_yr-2]
    for yr in viirs_ast_yrs:
        try:
            viirs_path_yr = glob.glob(str(ancillary_Path / \
                f"VIIRS-AST-IGBP17-GEO_v1r0_multi_s{yr}0101_e{yr}1231_*.nc"))[0]
            VIIRS_sfc_type_fpath = viirs_path_yr
            break
        except IndexError:
            pass

    try:
        # Get year of VIIRS surface type dataset for output file global attributes
        VIIRS_st_fname = os.path.basename(VIIRS_sfc_type_fpath)
        VIIRS_st_yr = VIIRS_st_fname.split('_s')[1][0:4]
    except:
        raise Exception("No VIIRS surface type file found")
    
    # BAS shapefile - match time of TIRS scenes to valid period for BAS file
    # - Assume that:
    #   - BAS file is valid forward in time between updates, e.g. the v7.6
    #     file updated on 2022-11-09 is valid for the period 2022-11-09 through
    #     2023-05-11 (when v7.7 file was updated)
    #   - The most recent file available is valid forward in time indefinitely,
    #     e.g. the v7.7 file released on 2023-05-11 is valid for all PREFIRE
    #     scenes after this date (if v7.8 file has not yet been released)
    bas_date_ranges = {
        '7.3':(dt.datetime(2020,10,30,0), dt.datetime(2021,4,30,0)),
        '7_4':(dt.datetime(2021,4,30,0), dt.datetime(2022,5,6,0)),
        '7_5':(dt.datetime(2022,5,6,0), dt.datetime(2022,11,9,0)),
        '7_6':(dt.datetime(2022,11,9,0), dt.datetime(2023,5,11,0)),
        '7_7':(dt.datetime(2023,5,11,0), dt.datetime.now())
        }
    
    # Use time of TIRS scene in the middle of the granule to determine which
    # BAS Antarctic coastlines file matches TIRS granule time
    for k in bas_date_ranges.keys():
        if bas_date_ranges[k][0] < TIRS_midpt_t <= bas_date_ranges[k][1]:
            bas_antarctic_version = k
    
    antarctic_shapefile_path = str(ancillary_Path / \
        f"add_coastline_medium_res_polygon_v{bas_antarctic_version}.shp")

    # Read output data product specifications:
    with open(product_specs_fpath, 'r') as in_f:
        data_product_specs = json.load(in_f)
    
    if "GEOSIT" in met_analysis_source:
        met_an_collection_flists, met_an_const_fpath, \
            interp_data_sfc_corr, met_an_idstr, \
            ancsim_vars = process_GEOSIT(ancillary_Path, met_analysis_dir,
                                         str(tmpfiles_Path), TIRS_L1B_ds,
                                         TIRS_L1B_fpath,
                                         met_analysis_source, interp_method,
                                         TIRS_geo_data,
                                         atrack_range_to_process,
                                         data_product_specs,
                                         ancsim_mode)
    else:
        print ("Unsupported met analysis source.")
        print ("Currently supported:\n"+
               "   GEOSIT_equal_angle\n"+
               "   GEOSIT_cubed_sphere\n")
        sys.exit(1)

    # For temperature, use surf_extrap_method=3 to interpolate to PCRTM levels
    # For all other profile variables, use surf_extrap_method=0
    PCRTM_levs_fpath = os.path.join(PREFIRE_sim_tools.paths._data_dir,
                                    "plevs101.txt")
    keys_filt = [k for k in list(interp_data_sfc_corr.keys()) if k not in \
                                                              ['temp_profile']]
    interp_data_no_temp_profile = {k:interp_data_sfc_corr[k] for k in keys_filt}
    
    interp_data = pressure_interp(interp_data_no_temp_profile,
                                  np.loadtxt(PCRTM_levs_fpath),
                                  surf_extrap_method=0)
    
    interp_data_temp_profile = pressure_interp({k:interp_data_sfc_corr[k] for \
                                     k in ['temp_profile','pressure_profile']},
                                               np.loadtxt(PCRTM_levs_fpath),
                                               surf_extrap_method=3)
    interp_data.update(interp_data_temp_profile)
    
    # Populate the below-surface flag variable
    interp_data['below_surface_flag'] = np.full(
        np.shape(interp_data['temp_profile']), 0
        )
    for ai, xi in np.ndindex(np.shape(interp_data['surface_pressure'])):
        interp_data['below_surface_flag'][ai,xi] = \
            np.where(
                interp_data['pressure_profile'] > interp_data['surface_pressure'][ai,xi],
                1, 0
                )

    t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Finished pressure interpolation at {t_now}')
    
    if ancsim_mode:
        # Derive cloud mask from vertical profiles of cloud fraction.
        # In cloud mask, cloud fraction, qi, and ql profiles, the
        #  apply_cloud_mask function sets below-surface values to 0.
        tmp_a1 = interp_data['cloud_fraction_profile'].copy()
        tmp_a2 = interp_data['qi_profile'].copy()
        tmp_a3 = interp_data['ql_profile'].copy()
        interp_data['cloud_mask_profile_correlated'],\
            interp_data['cloud_fraction_profile'],\
            interp_data['cloud_fraction_profile_correlated'],\
            interp_data['qi_profile'],\
            interp_data['ql_profile'] = apply_cloud_mask(
                data_product_specs,
                tmp_a1, tmp_a2, tmp_a3,
                interp_data['pressure_profile'],
                interp_data['surface_pressure']
            )

    # Replace nan values and met analysis fill values with PREFIRE FillValue
    # - nan values should occur when profiles completely filled with FillValue
    #   are passed to pressure_interp function. These profiles in turn should
    #   only be produced when the TIRS surface elevation value is masked and/or
    #   there is some other issue with the TIRS scene as identified by the
    #   quality flag.
    #   ** So revisit this step when the new high-res TIRS scene elevation data 
    #   and the quality flag are both implemented.
    for var in variables:
        if var in ancsim_variables:
            fval = data_product_specs["SimTruth"][var]["fill_value"]
        else:
            fval = data_product_specs["Aux-Met"][var]["fill_value"]
        np.place(interp_data[var], np.isnan(interp_data[var]), fval)
        
        # Met analysis fill value (10000000000000) is found in land_surface_temp and
        #   snow_cover variables over water. The effect of replacing arbitrarily
        #   large values (> 100000000) in the following line is that land_surface_temp
        #   and snow_cover will be -9999 for any TIRS scene not completely
        #   surrounded by four GEOS-IT land grid points.
        # If interpolation method is ESMF, a nan value at any of the four
        #   surrounding grid points should result in an interpolated value of
        #   nan, so this check should not be necessary.
#        if interp_method == 'internal':
#            np.place(interp_data[var], interp_data[var] > 100000000, FillValue)

    if ancsim_force_clearsky:
        # Set output cloud variables to zero (or equivalent), "forcing"
        #  a clear-sky interpretation of all FOVs -- but only for
        #  non-missing values:
        vn = "cloud_mask_profile_correlated"  # integer
        fval = data_product_specs["SimTruth"][vn]["fill_value"]
        interp_data[vn][interp_data[vn] != fval] = 0  # integer
        for vn in ["cloud_fraction_profile",
                   "cloud_fraction_profile_correlated",
                   "qi_profile", "ql_profile"]:
            fval = data_product_specs["SimTruth"][vn]["fill_value"]
              # Assumes missing/fill_value is < 0:
            interp_data[vn][interp_data[vn] > fval+0.01] = 0.

    # Add modelled CO2 and CH4 to output (based on CAMS EGG4 data for 2003-2020)
    interp_data['xco2'] = apply_CO2_CH4_model(TIRS_geo_data, CO2_modelfit_fpath,
                           data_product_specs["Aux-Met"]["xco2"]["fill_value"])
    interp_data['xch4'] = apply_CO2_CH4_model(TIRS_geo_data, CH4_modelfit_fpath,
                           data_product_specs["Aux-Met"]["xch4"]["fill_value"])
    
    t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Calculated CO2/CH4 at {t_now}')
    
    # Add VIIRS IGBP types (within-scene pixel counts) to output
    t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Started VIIRS surface type processing at {t_now}')
    interp_data['VIIRS_surface_type'] = \
        calc_VIIRS_IGBP_types(
            TIRS_geo_data, VIIRS_sfc_type_fpath, data_product_specs
            )
    
    # Calculate Antarctic land and ice shelf fraction of each TIRS scene
    interp_data['antarctic_land_fraction'],\
    interp_data['antarctic_ice_shelf_fraction'] = \
        calc_antarctic_overlap(TIRS_geo_data, antarctic_shapefile_path,
                               data_product_specs)
    t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Calculated Antarctic land and ice shelf fraction at {t_now}')
    
    # Classify preliminary surface type using L1B land fraction, Antarctic 
    # land and ice shelf fraction, VIIRS surface type, and GEOS-IT sea ice and
    # snow cover.
    interp_data['merged_surface_type_prelim'], \
    interp_data['merged_land_fraction_prelim_data_source'], \
    interp_data['merged_seaice_prelim_data_source'], \
    interp_data['merged_snow_prelim_data_source'] = \
        calc_sfc_type_prelim(TIRS_geo_data, interp_data, data_product_specs)

    if ancsim_force_noseaice:
        # Set output 'merged_surface_type_prelim' values of 2 or 3 (sea-ice or
        # partial-sea-ice, respectively) to 1 (open water) for all FOVs,
        # *and* then
        # set the sea-ice concentration array to zero for all FOVs (since
        # non-zero values of that array can occur even for
        # 'merged_surface_type_prelim' values of 1 or 4-8, due to heterogeneous
        # scenes):
        indv2 = np.nonzero(interp_data['merged_surface_type_prelim'][...] == 2)
        indv3 = np.nonzero(interp_data['merged_surface_type_prelim'][...] == 3)
        interp_data['merged_surface_type_prelim'][indv2] = 1
        interp_data['merged_seaice_prelim_data_source'][indv2] = -1
        interp_data['merged_surface_type_prelim'][indv3] = 1
        interp_data['merged_seaice_prelim_data_source'][indv3] = -1
        interp_data['seaice_concentration'][...] = 0.
    elif ancsim_force_allseaice:
        # Set output 'merged_surface_type_prelim' values of 1 or 3 (open water
        # or partial-sea-ice, respectively) to 2 (sea-ice) for all FOVs,
        # *and* then
        # set the sea-ice concentration array values to one for all FOVs with a
        # 'merged_surface_type_prelim' value of 2 (sea-ice) and to zero for all
        # other FOVs (since non-zero values of that array can occur even for
        # 'merged_surface_type_prelim' values of 1 or 4-8, due to heterogeneous
        # scenes):
        indv1 = np.nonzero(interp_data['merged_surface_type_prelim'][...] == 1)
        indv3 = np.nonzero(interp_data['merged_surface_type_prelim'][...] == 3)
        interp_data['merged_surface_type_prelim'][indv1] = 2
        interp_data['merged_seaice_prelim_data_source'][indv1] = -1
        interp_data['merged_surface_type_prelim'][indv3] = 2
        interp_data['merged_seaice_prelim_data_source'][indv3] = -1
        interp_data['seaice_concentration'][...] = 0.
        indv2 = np.nonzero(interp_data['merged_surface_type_prelim'][...] == 2)
        interp_data['seaice_concentration'][indv2] = 1.

    t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Calculated prelim surface type at {t_now}')
    
    # If run in ANC-SimTruth mode, add extra cloud variables (cloud flag, cloud
    # optical depth, cloud effective diameter, and cloud pressure profile).
    if ancsim_mode:
        interp_data["cloud_flag"], interp_data["cloud_od"], \
             interp_data["cloud_de"], interp_data["cloud_dp"] = (
                                         add_estimated_cloudprops(interp_data))

        if ancsim_force_clearsky:
            # Set all output cloud variables to zero (or equivalent), "forcing"
            #  a clear-sky interpretation of all FOVs -- but only for
            #  non-missing values:
            vn = "cloud_flag"  # integer
            fval = data_product_specs["SimTruth"][vn]["fill_value"]
            interp_data[vn][interp_data[vn] != fval] = 0  # integer
              # "cloud_dp" is not currently zeroed
            for vn in ["cloud_od", "cloud_de"]:
                fval = data_product_specs["SimTruth"][vn]["fill_value"]
                  # Assumes missing/fill_value is < 0:
                interp_data[vn][interp_data[vn] > fval+0.01] = 0.
        
        t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Added estimated extra cloud variables at {t_now}')
        
    write_AUXMET_product(
        TIRS_L1B_fpath, TIRS_L1B_ds, product_specs_fpath,
        AUX_MET_output_dir, interp_data,
        atrack_range_to_process,
        met_analysis_source, met_an_idstr, VIIRS_st_yr, bas_antarctic_version,
        product_full_version,
        ancsim_vars, ancsim_force_clearsky
        )

    TIRS_L1B_ds.close()  # Done with this input file object
  
    # Remove temporary regridded .npy files from working directory:
    npy_fpaths = glob.glob(str(tmpfiles_Path / "*_regrid.npy"))
    for npy_fpath in npy_fpaths:
        os.remove(npy_fpath)
