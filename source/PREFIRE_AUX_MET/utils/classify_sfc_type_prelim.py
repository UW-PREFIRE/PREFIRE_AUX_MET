import numpy as np
import numpy.ma as ma
import netCDF4 as n
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely
import datetime as dt

from PREFIRE_tools.utils.aux import generate_gc_poly_bounds, poly_antimeridian_split


def calc_VIIRS_IGBP_types(TIRS_geo_data, VIIRS_sfc_type_path,
                          data_product_specs):
    """
    Calculate within-scene VIIRS pixel counts of each IGBP type for each TIRS
    scene.

    Parameters
    ----------
    TIRS_geo_data : dict
        Contains selected TIRS Level-1B geometry data.
    VIIRS_sfc_type_path : str
        Path to VIIRS surface type netCDF file.
    data_product_specs : dict
        Dictionary of data product specifications

    Returns
    -------
    viirs_st_counts : numpy.ndarray
        Count of VIIRS pixels of each IGBP type within each TIRS scene.

    """
    
    # Read VIIRS global surface type array and coordinates
    with n.Dataset(VIIRS_sfc_type_path) as nc:
        viirs_lons = nc.variables['lon'][:]
        viirs_lats = nc.variables['lat'][:]
        viirs_st_global = nc.variables['surface_type'][:]

    # Find nearest VIIRS lat/lon point to each TIRS scene center point
    nearest_viirs_lons = np.around(TIRS_geo_data['lons_deg']*120) / 120
    nearest_viirs_lats = np.around(TIRS_geo_data['lats_deg']*120) / 120

    # Find array ixs of nearest VIIRS coordinates
    viirs_lon_ixs = ((nearest_viirs_lons + 180) / (1/120)).astype(int)
    viirs_lat_ixs = (21600 - ((nearest_viirs_lats + 90) / (1/120))).astype(int)
    
    # Fixed array of IGBP surface type classes
    igbp_classes = np.arange(1,18,1)

    # The number of VIIRS array indices beyond which the "buffer" grid will
    # extend past the max/min lat/lon TIRS scene center coordinates of each
    # along-track frame
    lon_buffer_npts = 250
    lat_buffer_npts = 35
    
    # Output array that will be filled with IGBP type counts within each TIRS
    # scene
    viirs_st_counts = np.full(
        (viirs_lon_ixs.shape[0], viirs_lon_ixs.shape[1], igbp_classes.shape[0]),
        data_product_specs['Aux-Met']['VIIRS_surface_type']['fill_value'],
        dtype=data_product_specs['Aux-Met']['VIIRS_surface_type']['np_dtype']
    )

    # Loop through along-track indices and determine VIIRS IGBP type counts
    # within all cross-track scenes for each along-track frame
    for ai in np.arange(0, viirs_lon_ixs.shape[0], 1):
        # Construct array of VIIRS surface type and STRtree of VIIRS lat/lon
        # coordinates within a buffer grid surrounding all 8 TIRS scenes of
        # the given along-track frame
        viirs_st_buffer, buffer_tree = _construct_buffer_data(
            viirs_lons, viirs_lats, viirs_st_global,
            viirs_lon_ixs, viirs_lat_ixs,
            lon_buffer_npts, lat_buffer_npts,
            ai
            )
          
        # Find VIIRS indices that are within each cross-track TIRS scene at the
        # given along-track frame
        for xi in np.arange(0, viirs_lon_ixs.shape[1], 1):
            tirs_intersect_ixs = _calc_tirs_intersect_ixs(
                TIRS_geo_data, buffer_tree, ai, xi
                )
                            
            # Extract VIIRS surface type values at these indices
            viirs_st_scene = viirs_st_buffer[tirs_intersect_ixs]
    
            # Count the VIIRS surface type values within each IGBP category,
            # then add counts to output array
            for igbp_class in igbp_classes:
                npx_type = np.sum(np.where(viirs_st_scene == igbp_class, 1, 0))
                viirs_st_counts[ai, xi, igbp_class-1] = npx_type
                
        # Print time at every 1000th along-track frame
        if (ai % 1000) == 0:
            now = dt.datetime.now()
            print(f'Finished atrack ix: {ai} for VIIRS sfc type at {now}')
    
    return viirs_st_counts

def _construct_buffer_data(viirs_lons, viirs_lats, viirs_st_global,
                           viirs_lon_ixs, viirs_lat_ixs,
                           lon_buffer_npts, lat_buffer_npts,
                           ai):
    """
    Helper to get_VIIRS_IGBP_types that builds a "buffer" grid of VIIRS surface
    type array coordinates surrounding all cross-track TIRS scenes in the given
    along-track frame. The coordinates are used to build (1) an array of VIIRS
    IGBP type values within the buffer, and (2) an STRtree containing the 
    lat/lon coordinates of all VIIRS points in the buffer.

    Parameters
    ----------
    viirs_lons : numpy.ndarray
        Longitudes of the global VIIRS surface type array.
    viirs_lats : numpy.ndarray
        Latitudes of the global VIIRS surface type array.
    viirs_st_global : numpy.ndarray
        Global VIIRS surface type array.
    viirs_lon_ixs : numpy.ndarray
        Longitude indices into the global VIIRS surface type array for the 
        nearest VIIRS global point to each TIRS scene center.
    viirs_lat_ixs : numpy.ndarray
        Latitude indices into the global VIIRS surface type array for the 
        nearest VIIRS global point to each TIRS scene center.
    lon_buffer_npts : int
        Number of VIIRS longitude indices to extend the buffer outside of the
        max/min longitudes of the full along-track frame.
    lat_buffer_npts : int
        Number of VIIRS latitude indices to extend the buffer outside of the
        max/min latitudes of the full along-track frame.
    ai : int
        TIRS along-track index.

    Returns
    -------
    viirs_st_buffer : numpy.ndarray
        VIIRS IGBP surface type array (flattened to 1D) within the buffer.
    buffer_tree : shapely.STRtree
        STRtree containing all combinations of lat/lon coordinates in the buffer.

    """
    
    # Find the min and max lat/lon indices of the global VIIRS grid across
    # all cross-track scenes at the given along-track TIRS frame
    viirs_min_lon_ix_ai = np.min(viirs_lon_ixs[ai,:])
    viirs_max_lon_ix_ai = np.max(viirs_lon_ixs[ai,:])
    viirs_min_lat_ix_ai = np.min(viirs_lat_ixs[ai,:])
    viirs_max_lat_ix_ai = np.max(viirs_lat_ixs[ai,:])
    
    # Latitude start/end indices of buffer are straightforward because TIRS
    # scenes never approach the VIIRS grid top or bottom edges
    lat_start_ix_n = viirs_min_lat_ix_ai - lat_buffer_npts
    lat_end_ix_s = viirs_max_lat_ix_ai + lat_buffer_npts
    lat_ixs_buffer = np.arange(lat_start_ix_n, lat_end_ix_s+1, 1)

    # Longitude start/end indices of buffer are more complicated because of 
    # date line crosses. The following blocks handle frames with buffers
    # that cross the date line.
    
    # Case where the actual TIRS frame crosses the date line
    # - In this case, viirs_min_lon_ix_ai will be near or less than 0 (i.e.
    #   much less than 10000) and viirs_max_lon_ix_ai will be near 43200
    #   (i.e. much greater than 10000)
    if np.abs(viirs_max_lon_ix_ai - viirs_min_lon_ix_ai) > 10000:            
        # Start lon buffer W of the minimum lon ix that is located in EH
        lon_start_ix = np.min(
            viirs_lon_ixs[ai,:][np.where(viirs_lon_ixs[ai,:] > 10000)]
            ) - lon_buffer_npts
        # End lon buffer E of the maximum lon ix that is located in WH
        lon_end_ix = np.max(
            viirs_lon_ixs[ai,:][np.where(viirs_lon_ixs[ai,:] < 10000)]
            ) + lon_buffer_npts
        lon_ixs_buffer = np.concatenate((
            np.arange(lon_start_ix, len(viirs_lons), 1),
            np.arange(0, lon_end_ix+1, 1)
        ))
        
        viirs_st_buffer = np.concatenate((
            np.ravel(viirs_st_global[lat_start_ix_n:lat_end_ix_s+1,
                                     lon_start_ix:len(viirs_lons)]),
            np.ravel(viirs_st_global[lat_start_ix_n:lat_end_ix_s+1,
                                     0:lon_end_ix+1])
            ))
        
    # Case where the TIRS frame is entirely in WH and doesn't cross the
    # date line, but the surrounding buffer does
    elif ((viirs_min_lon_ix_ai - lon_buffer_npts) < 0) and \
        (viirs_max_lon_ix_ai < 10000):
        # lon buffer starts in EH            
        lon_start_ix = (len(viirs_lons) + (viirs_min_lon_ix_ai - lon_buffer_npts))
        # lon buffer ends in WH
        lon_end_ix = viirs_max_lon_ix_ai + lon_buffer_npts
        lon_ixs_buffer = np.concatenate((
            np.arange(lon_start_ix, len(viirs_lons), 1),
            np.arange(0, lon_end_ix+1, 1)
        ))
        
        viirs_st_buffer = np.concatenate((
            np.ravel(viirs_st_global[lat_start_ix_n:lat_end_ix_s+1,
                                     lon_start_ix:len(viirs_lons)]),
            np.ravel(viirs_st_global[lat_start_ix_n:lat_end_ix_s+1,
                                     0:lon_end_ix+1])
            ))
        
    # Case where the TIRS frame is entirely in EH and doesn't cross the
    # date line, but the surrounding buffer does
    elif ((viirs_max_lon_ix_ai + lon_buffer_npts) >= len(viirs_lons)) and \
        (viirs_min_lon_ix_ai > 10000):
        # lon buffer starts in EH            
        lon_start_ix = viirs_min_lon_ix_ai - lon_buffer_npts
        # lon buffer ends in WH
        lon_end_ix = (viirs_max_lon_ix_ai + lon_buffer_npts) - len(viirs_lons)
        lon_ixs_buffer = np.concatenate((
            np.arange(lon_start_ix, len(viirs_lons), 1),
            np.arange(0, lon_end_ix+1, 1)
        ))
        
        viirs_st_buffer = np.concatenate((
            np.ravel(viirs_st_global[lat_start_ix_n:lat_end_ix_s+1,
                                     lon_start_ix:len(viirs_lons)]),
            np.ravel(viirs_st_global[lat_start_ix_n:lat_end_ix_s+1,
                                     0:lon_end_ix+1])
            ))
        
    # Case where the neither the TIRS frame nor the buffer cross the date line
    else:
        # lon buffer starts and ends in the same hemisphere            
        lon_start_ix = viirs_min_lon_ix_ai - lon_buffer_npts
        lon_end_ix = viirs_max_lon_ix_ai + lon_buffer_npts
        lon_ixs_buffer = np.arange(lon_start_ix, lon_end_ix+1, 1)
        
        viirs_st_buffer = np.ravel(
            viirs_st_global[lat_start_ix_n:lat_end_ix_s+1,
                            lon_start_ix:lon_end_ix+1]
            )
    
    # Get VIIRS lat/lon coordinates at buffer indices
    lons_buffer_1d = viirs_lons[lon_ixs_buffer]
    lats_buffer_1d = viirs_lats[lat_ixs_buffer]

    # Create 2D array of all possible pairs of lon/lat coordinates within
    # the buffer, then flatten to 1D for STRtree creation
    lons_buffer_2d, lats_buffer_2d = np.meshgrid(lons_buffer_1d, lats_buffer_1d)
    lons_flat_1d = np.ravel(lons_buffer_2d)
    lats_flat_1d = np.ravel(lats_buffer_2d)
    
    # Create STRtree of coordinates within buffer
    buffer_tree = shapely.STRtree(shapely.points(lons_flat_1d, y=lats_flat_1d))
    
    return viirs_st_buffer, buffer_tree

def _calc_tirs_intersect_ixs(TIRS_geo_data, buffer_tree, ai, xi):
    """
    Helper to get_VIIRS_IGBP_types that determines indices of a given "buffer"
    grid that are within a given TIRS scene.

    Parameters
    ----------
    TIRS_geo_data : dict
        Contains selected TIRS Level-1B geometry data.
    buffer_tree : shapely.STRtree
        STRtree containing all combinations of lat/lon coordinates in the buffer.
    ai : int
        TIRS along-track index.
    xi : int
        TIRS cross-track index.

    Raises
    ------
    Exception
        Error: TIRS scene split into more than 2 polygons.

    Returns
    -------
    tirs_intersect_ixs : numpy.ndarray
        Indices of the 1D buffer array that are within the given TIRS scene.

    """
    
    # Create TIRS polygons (including antimeridian splits) for the given scene 
    # - (Adapted from construct_tirs_geo function in Aux-Sat)
    tirs_polylist = []
    
    vlons_fov = TIRS_geo_data['vertex_lons_deg'][ai,xi,:]
    vlats_fov = TIRS_geo_data['vertex_lats_deg'][ai,xi,:]
    
    vlons_closed = np.append(vlons_fov, vlons_fov[0])
    vlats_closed = np.append(vlats_fov, vlats_fov[0])
        
    lon_diffs = [lon - vlons_closed for lon in vlons_closed]
    # Crosses antimeridian (TIRS scenes should never cross a pole), so
    # create 2 polygons split by the antimeridian
    if np.max(np.abs(lon_diffs)) > 180:
        split_polys = []
        poly_pts_unsplit = generate_gc_poly_bounds(
            vlons_closed, vlats_closed
            )                
        split_polys_pts = poly_antimeridian_split(poly_pts_unsplit)
        for poly_pts in split_polys_pts:
            split_polys.append(shapely.polygons(poly_pts))

        # Raise exception if TIRS scene split into more than 2 polygons.
        # This would indicate a bug in poly_antimeridian_split.
        if len(split_polys) > 2:
            raise Exception(
                'Error: TIRS scene split into more than 2 polygons.'
                )

        tirs_polylist.append(split_polys)
    # Doesn't cross antimeridian, so create a single polygon for the 
    # TIRS scene
    else:
        poly = shapely.polygons(
            [[lon,lat] for lon,lat in zip(vlons_closed, vlats_closed)]
            )                
        tirs_polylist.append([poly])
        
    # Get indices of TIRS polygon intersection with atrack STRtree
    for tirs_polys in tirs_polylist:
        if len(tirs_polys) == 2:
            tirs_intersect_ixs = []
            for split_poly in tirs_polys:
                ixs_split_scene = buffer_tree.query(
                    split_poly, predicate='contains'
                    )
                tirs_intersect_ixs.extend(ixs_split_scene)
        elif len(tirs_polys) == 1:
            tirs_intersect_ixs = buffer_tree.query(
                tirs_polys[0], predicate='contains'
                )
    
    return tirs_intersect_ixs


def calc_antarctic_overlap(TIRS_geo_data, antarctic_shapefile_path,
                           data_product_specs):
    """
    Calculate the fractional overlap between each South Polar TIRS scene
    and any ice shelf or ice tongue polygon(s) that it intersects.

    Parameters
    ----------
    TIRS_geo_data : dict
        Contains selected TIRS Level-1B geometry data.
    antarctic_shapefile_path : str
        Path to Antarctic coastlines + ice shelves polygons shapefile.
    data_product_specs : dict
        Dictionary of data product specifications

    Returns
    -------
    land_overlap_frac : numpy.ndarray
        Fractional overlap between each TIRS scene and any Antarctic (< -60S)
        land polygon(s) that it intersects.
    iceshelf_overlap_frac : numpy.ndarray
        Fractional overlap between each TIRS scene and any ice shelf, ice
        tongue, and/or ice rumple polygon(s) that it intersects.

    """
    
    tirs_spolar_ixs, tirs_spolar_mask_ixs, tirs_spolar_strtree, tirs_shape = \
        _build_tirs_spolar_strtree(TIRS_geo_data)
        
    land_polys, iceshelf_polys = _read_antarctic_polys(antarctic_shapefile_path)
    
    land_overlap_frac_1d = _calc_antarctic_overlap_frac(
        land_polys, tirs_spolar_ixs, tirs_spolar_strtree, tirs_shape
        )
    iceshelf_overlap_frac_1d = _calc_antarctic_overlap_frac(
        iceshelf_polys, tirs_spolar_ixs, tirs_spolar_strtree, tirs_shape
        )
        
    # In output data, mask out all TIRS scenes that are not poleward of -60S.
    land_overlap_frac_1d_masked = np.array(land_overlap_frac_1d)
    fval = data_product_specs["Aux-Met"]["antarctic_land_fraction"]["fill_value"]
    land_overlap_frac_1d_masked[tirs_spolar_mask_ixs] = fval
    iceshelf_overlap_frac_1d_masked = np.array(iceshelf_overlap_frac_1d)
    fval = data_product_specs["Aux-Met"]["antarctic_ice_shelf_fraction"]["fill_value"]
    iceshelf_overlap_frac_1d_masked[tirs_spolar_mask_ixs] = fval
    
    # Reshape to (atrack, xtrack) dimensions
    land_overlap_frac = np.reshape(land_overlap_frac_1d_masked, tirs_shape)
    iceshelf_overlap_frac = np.reshape(iceshelf_overlap_frac_1d_masked, tirs_shape)
    
    return land_overlap_frac, iceshelf_overlap_frac

def _build_tirs_spolar_strtree(TIRS_geo_data):
    """
    Helper to calc_antarctic_overlap. Create STRtree consisting of TIRS scene
    polygons that are located in the South Polar region (poleward of -60S
    latitude).

    Parameters
    ----------
    TIRS_geo_data : dict
        Contains selected TIRS Level-1B geometry data.

    Returns
    -------
    tirs_spolar_ixs : list
        1D indices (into the list of all TIRS scenes in a granule) of TIRS
        scenes that are poleward of -60S.
    tirs_spolar_mask_ixs : list
        1D indices (into the list of all TIRS scenes in a granule) of TIRS
        scenes that are *not* poleward of -60S.
    tirs_spolar_strtree : shapely.STRtree
        STRtree of TIRS scene polygons that are located in the South Polar
        region (poleward of -60S latitude).
    tirs_shape : tuple
        Shape of TIRS scenes in PREFIRE granule (atrack, xtrack).

    """
    
    # Reshape TIRS FOV vertex lats/lons from 3D to 2D arrays
    # - 3D dims: (atrack, xtrack, FOV_vertices)
    # - 2D dims: (atrack * xtrack, FOV_vertices)
    vlons_reshp = np.reshape(
        TIRS_geo_data["vertex_lons_deg"],
        (np.size(TIRS_geo_data["lons_deg"]), 4)
        )
    vlats_reshp = np.reshape(
        TIRS_geo_data["vertex_lats_deg"],
        (np.size(TIRS_geo_data["lats_deg"]), 4)
        )
    tirs_shape = TIRS_geo_data["lats_deg"].shape
    
    tirs_spolar_ixs = []
    tirs_spolar_mask_ixs = []
    tirs_spolar_polys_tf = []
    
    ccrs_epsg_3031 = ccrs.SouthPolarStereo(
        central_longitude=0, true_scale_latitude=-71.
        )
    
    # Antarctic (<-60S) TIRS scenes only
    for i, (vlons_fov, vlats_fov) in enumerate(zip(vlons_reshp, vlats_reshp)):
        maxlat = np.max(vlats_fov)
        if maxlat < -60:
            tirs_spolar_ixs.append(i)
                        
            # Transform TIRS scene corner points into the same coordinate system
            # as ice shelf polygons shapefile
            corner_pts_tf = ccrs_epsg_3031.transform_points(
                ccrs.PlateCarree(), vlons_fov, vlats_fov
                )
            
            poly_tf = shapely.polygons([[pt[0],pt[1]] for pt in corner_pts_tf])
            tirs_spolar_polys_tf.append(poly_tf)
        else:
            tirs_spolar_mask_ixs.append(i)
        
    tirs_spolar_strtree = shapely.STRtree(tirs_spolar_polys_tf)
    
    return tirs_spolar_ixs, tirs_spolar_mask_ixs, tirs_spolar_strtree, tirs_shape

def _read_antarctic_polys(antarctic_shapefile_path):
    """
    Helper to calc_antarctic_overlap. Read land and ice shelf polygons from
    Antarctic coastlines shapefile.

    Parameters
    ----------
    antarctic_shapefile_path : str
        Path to Antarctic coastlines shapefile.

    Returns
    -------
    land_polys : list
        Polygons of each individual land feature.
    iceshelf_polys : list
        Polygons of each individual ice shelf feature.

    """
    
    # The other associated files (.shx, .dbf, .prj, .cpg) must be in the same
    # directory as the .shp file that antarctic_shapefile_path points to.
    reader = shpreader.Reader(antarctic_shapefile_path)
    land_polys = []
    iceshelf_polys = []
    
    # Descriptions of Antarctic features are located at a different index
    # for v7.3 and v7_4 of the shapefile compared to later versions
    bas_antarctic_version = antarctic_shapefile_path.split('.shp')[0][-4:]
    if bas_antarctic_version in ['v7.3','v7_4']:
        descr_ix = 1
    else:
        descr_ix = 0
    
    for record in reader.records():
        if list(record.attributes.values())[descr_ix] == 'land':
            land_polys.append(record.geometry)
        # "ice shelves", "ice tongues", and "rumples" are all counted as ice shelves
        if list(record.attributes.values())[descr_ix] in ['ice shelf','ice tongue','rumple']:
            iceshelf_polys.append(record.geometry)
        
    return land_polys, iceshelf_polys

def _calc_antarctic_overlap_frac(polys, tirs_spolar_ixs, tirs_spolar_strtree, tirs_shape):
    """
    Helper to calc_antarctic_overlap. Calculates overlap fraction between
    TIRS scenes and all land or ice shelf polygons.

    Parameters
    ----------
    polys : list
        Polygons of each individual land or ice shelf feature.
    tirs_spolar_ixs : list
        1D indices (into the list of all TIRS scenes in a granule) of TIRS
        scenes that are poleward of -60S.
    tirs_spolar_strtree : shapely.STRtree
        STRtree of TIRS scene polygons that are located in the South Polar
        region (poleward of -60S latitude).
    tirs_shape : tuple
        Shape of TIRS scenes in PREFIRE granule (atrack, xtrack).

    Returns
    -------
    overlap_frac_1d : list
        Overlap fraction between each TIRS scene and any land or ice shelf
        feature.

    """
    
    # For each land or ice shelf polygon, determine which TIRS scene(s) it
    # intersects, if any.
    tirs_intersect_ixs = []
    for poly in polys:
        tirs_intersect_ixs.append(
            list(tirs_spolar_strtree.query(poly, predicate='intersects'))
            )
    
    # Loop through the list of TIRS scene intersections for each land or ice shelf
    # polygon and calculate the percentage of overlap.
    overlap_frac_1d = [0]*tirs_shape[0]*tirs_shape[1]
    for i, tirs_spolar_ixlist in enumerate(tirs_intersect_ixs):
        if len(tirs_spolar_ixlist) > 0:
            poly = polys[i]
            
            for tirs_spolar_ix in tirs_spolar_ixlist:
                tirs_poly = tirs_spolar_strtree.geometries[tirs_spolar_ix]
                tirs_scene_area = shapely.area(tirs_poly)
                overlap_area = shapely.area(shapely.intersection(tirs_poly, poly))
                frac_overlap = (overlap_area / tirs_scene_area)
    
                tirs_full_ix = tirs_spolar_ixs[tirs_spolar_ix]
                overlap_frac_1d[tirs_full_ix] += frac_overlap

    return overlap_frac_1d


def calc_sfc_type_prelim(TIRS_geo_data, interp_data, data_product_specs):
    """
    Calculate the "preliminary" PREFIRE surface type classification, which
    depends only on static products and met analysis data (e.g. GEOS-IT).

    Parameters
    ----------
    TIRS_geo_data : dict
        Contains selected TIRS Level-1B geometry data.
    interp_data : dict
        Dictionary containing arrays of met analysis data. This function can write
        data from any stage of AUX-MET processing, but in final version this should
        contain data interpolated spatially and temporally to TIRS scenes, with 
        vertical profile variables surface-corrected and interpolated to PCRTM 
        fixed-101 pressure levels.
    data_product_specs : dict
        Dictionary of data product specifications

    Returns
    -------
    sfc_type : numpy.ndarray
        The PREFIRE surface type classification of each TIRS scene.
    lf_data_source : numpy.ndarray
        Data source used to determine land fraction for each TIRS scene.
    seaice_data_source : numpy.ndarray
        Data source used to determine sea ice concentration for each TIRS scene.
    snow_data_source : numpy.ndarray
        Data source used to determine snow cover for each TIRS scene.

    """
    
    tirs_shape = TIRS_geo_data["lats_deg"].shape
    
    # Initialize output data
    sfc_type_1d = []
    lf_data_source_1d = []
    seaice_data_source_1d = []
    snow_data_source_1d = []
    
    tirs_lats = TIRS_geo_data["lats_deg"].ravel()
    lf = TIRS_geo_data["land_fraction"].ravel()
    ant_isf = interp_data['antarctic_ice_shelf_fraction'].ravel()
    ant_lf = interp_data['antarctic_land_fraction'].ravel()
    VIIRS_stype = interp_data['VIIRS_surface_type'].reshape(
        -1, interp_data['VIIRS_surface_type'].shape[-1]
        )
    auxmet_seaice = interp_data['seaice_concentration'].ravel()
    auxmet_snow = interp_data['snow_cover'].ravel()
    auxmet_lf = interp_data['land_fraction'].ravel()

    dps_group = data_product_specs["Aux-Met"]
    
    # Loop through TIRS scenes and classify surface type
    # If used, PREFIRE sea ice and snow data source is always 7 (GEOS-IT) for
    # this preliminary surface type classification.
    fval_seai = dps_group["merged_seaice_prelim_data_source"]["fill_value"]
    fval_snow = dps_group["merged_snow_prelim_data_source"]["fill_value"]
    for i, lat in enumerate(tirs_lats):
        # Antarctic scenes
        if lat < -60:
            # Land fraction data source 2 = BAS Antarctic coastline
            lf_data_source_1d.append(2)
            lf_scene = ant_lf[i]
            isf_scene = ant_isf[i]
            
            # Ice shelf
            if ((lf_scene + isf_scene) > 0.5) and (isf_scene > lf_scene):
                # PREFIRE surface type 5 = Antarctic ice shelf
                sfc_type_scene = 5
                seaice_data_source_scene = fval_seai
                snow_data_source_scene = fval_snow
            # Land
            elif ((lf_scene + isf_scene) > 0.5) and (lf_scene > isf_scene):
                sfc_type_scene = classify_land_type(VIIRS_stype[i], auxmet_snow[i])
                seaice_data_source_scene = fval_seai
                snow_data_source_scene = 7
            # Water
            else:
                sfc_type_scene = classify_water_type(auxmet_seaice[i])
                seaice_data_source_scene = 7
                snow_data_source_scene = fval_snow
        # Non-Antarctic scenes
        else:
            # Use GEOS-IT land fraction if L1B land fraction is missing
            if ma.is_masked(lf[i]):
                lf_scene = auxmet_lf[i]
                # Land fraction data source 7 = GEOS-IT (via Aux-Met)
                lf_data_source_1d.append(7)
            else:
                lf_scene = lf[i]
                # Land fraction data source 1 = Copernicus DEM (via L1B)
                lf_data_source_1d.append(1)
            
            if lf_scene >= 0.5:
                sfc_type_scene = classify_land_type(VIIRS_stype[i], auxmet_snow[i])
                seaice_data_source_scene = fval_seai
                snow_data_source_scene = 7
            else:
                sfc_type_scene = classify_water_type(auxmet_seaice[i])
                seaice_data_source_scene = 7
                snow_data_source_scene = fval_snow

        sfc_type_1d.append(sfc_type_scene)
        seaice_data_source_1d.append(seaice_data_source_scene)
        snow_data_source_1d.append(snow_data_source_scene)
            
    # Reshape to 2D
    sfc_type = np.reshape(np.array(sfc_type_1d), tirs_shape)
    lf_data_source = np.reshape(np.array(lf_data_source_1d), tirs_shape)
    seaice_data_source = np.reshape(np.array(seaice_data_source_1d), tirs_shape)
    snow_data_source = np.reshape(np.array(snow_data_source_1d), tirs_shape)
    
    return sfc_type, lf_data_source, seaice_data_source, snow_data_source


def classify_land_type(igbp_types_scene, snow_cover_scene):
    """
    Classifies land scenes into a PREFIRE scene type.

    Parameters
    ----------
    igbp_types_scene : numpy.ndarray
        Count of VIIRS pixels of each IGBP type within a single TIRS scene.
    snow_cover_scene : float
        Snow cover for a single TIRS scene.

    Returns
    -------
    sfc_type_scene : int
        The PREFIRE surface type classification of a single TIRS scene.

    """
    
    # IGBP type 15 = permanent snow / ice; array index 14 in igbp_types_scene
    # - This check is to determine if the most common *land* type within TIRS
    #   scene is permanent snow / ice, so exclude water pixels from check
    if np.argmax(igbp_types_scene[:-1]) == 14:
        # PREFIRE surface type 4 = permanent land ice
        sfc_type_scene = 4
    else:
        if snow_cover_scene >= 0.5:
            # PREFIRE surface type 6 = snow covered land
            sfc_type_scene = 6
        elif 0.01 < snow_cover_scene < 0.5:
            # PREFIRE surface type 7 = partial snow covered land
            sfc_type_scene = 7
        # Land scenes with missing snow cover data from GEOS-IT
        # (snow_cover_scene == FillValue) are classified as snow-free land
        else:
            # PREFIRE surface type 8 = snow-free land
            sfc_type_scene = 8
    
    return sfc_type_scene
    

def classify_water_type(seaice_scene):
    """
    Classifies water scenes into a PREFIRE scene type.

    Parameters
    ----------
    seaice_scene : float
        Sea ice concentration for a single TIRS scene.

    Returns
    -------
    sfc_type_scene : int
        The PREFIRE surface type classification of a single TIRS scene.

    """
    
    # PREFIRE surface type 2 = sea ice
    if seaice_scene >= 0.95:
        sfc_type_scene = 2
    # PREFIRE surface type 3 = partial sea ice
    elif 0.05 <= seaice_scene < 0.95:
        sfc_type_scene = 3
    # PREFIRE surface type 1 = open water
    # Water scenes with missing sea ice data from GEOS-IT
    # (seaice_scene == FillValue) are classified as open water
    else:
        sfc_type_scene = 1
    
    return sfc_type_scene
