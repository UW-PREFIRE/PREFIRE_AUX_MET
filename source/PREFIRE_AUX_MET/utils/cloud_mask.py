import numpy as np
import copy

FillValue = -9999.0

def cloud_mask_calc(atrack_ix,
                    xtrack_ix,
                    CF_profile_orig,
                    CM_profile_allscenes,
                    QI_profile,
                    QL_profile,
                    pres_profile,
                    surface_pres,
                    alpha=3.):
    """
    First, adjust cloud fraction to account for the previous scene's mask
    result. (For first scene in granule, use un-adjusted ("original") cloud
    fraction.)
    
    Then, determine cloud mask for each vertical layer of a single profile using a 
    max-overlap like assumption applied to cloud fraction profile:
    
    Find CFMax in profile. Draw random number for CFmax at the layer of CFmax.
    - If clear, then the entire profile is clear.
    - If cloudy, set layer of CFmax to cloudy, and then do random draws in the
      rest of the profile according to (CFprofile / CFmax).
      
    Since PCRTM only allows 36 cloud layers: after creating the intial cloud
    mask profile, check if the number of cloud levels is less than 37. If true,
    do nothing; if >= 37, change cloud layers to clear to get down to 36, by
    "removing" the cloud layers with the smallest (qi + ql) values.
    
    Since PCRTM can't handle below-surface cloud layers: find the layers in each
    profile where the pressure is greater than the surface pressure. Then zero out
    cloud mask, cloud fraction, QI, and QL on all these layers *except* the first
    below-surface layer.
    
    Parameters
    ----------
    atrack_ix : int
        Along-track index.
    xtrack_ix : int
        Cross-track index.
    CF_profile_orig : numpy.ndarray
        Cloud fraction profile for a single TIRS scene.
        Dimensions: (zlevels)
    CM_profile_allscenes : numpy.ndarray
        Cloud mask profile for all TIRS scenes.
        Dimensions: (atrack, xtrack, zlevels)
    QI_profile : numpy.ndarray
        QI profile for a single TIRS scene.
        Dimensions: (zlevels)
    QL_profile : numpy.ndarray
        QL profile for a single TIRS scene.
        Dimensions: (zlevels)
    pres_profile : numpy.ndarray
        PCRTM pressure levels.
        Dimensions: (zlevels)
    surface_pres : float
        Surface pressure for a single TIRS scene.
    alpha : float
        cloud decoupling parameter (bigger = less influence from previous footprint);
        in the limit of alpha -> infty cloud fields are randomly sampled

    Returns
    -------
    cloud_mask_profile : numpy.ndarray
        Cloud mask profile with below-surface layers zeroed out, with 
        cloud correlation scale correction applied.
        Dimensions: (zlevels)
    CF_profile_orig : numpy.ndarray
        Cloud fraction profile with below-surface layers zeroed out.
        Dimensions: (zlevels)
    CF_profile_corr : numpy.ndarray
        Cloud fraction profile with below-surface layers zeroed out, with 
        cloud correlation scale correction applied.
        Dimensions: (zlevels)
    QI_profile : numpy.ndarray
        QI profile with below-surface layers zeroed out.
        Dimensions: (zlevels)
    QL_profile : numpy.ndarray
        QL profile with below-surface layers zeroed out.
        Dimensions: (zlevels)

    """
        
    # First along-track row of scenes: we don't know what the previous scene's
    # mask result was, so don't adjust cloud fraction based on the previous
    # scene's mask result.
    if atrack_ix == 0:
        CF_profile_corr = CF_profile_orig
    # All other scenes: adjust CF to account for previous scene's mask result,
    # then clip to be [0,1]
    else:
        CM_profile_previous = CM_profile_allscenes[atrack_ix-1, xtrack_ix]
        CF_profile_corr = np.clip(
            CF_profile_orig - ((-1)**CM_profile_previous)/alpha,
            a_min=0,
            a_max=1
            )

    CF_max_layer = CF_profile_corr.argmax()
    CF_profile_max = CF_profile_corr[CF_max_layer]
    # Compute a scaled profile where the max CF is 100%.
    CF_profile_scaled = CF_profile_corr / CF_profile_max
    
    nlayers = CF_profile_corr.shape[0]
    cloud_mask_profile_full = np.zeros(CF_profile_corr.shape, dtype='int8')
    # for max overlap: draw scalar random number to decide if the profile
    # is cloudy or not....
    if np.random.uniform(0,1,1) < CF_profile_max:
        # ... if cloudy, then decide which layers have cloud by drawing
        # random numbers against the scaled profile. Since this has a peak of 100%,
        # we should automatically get at least one cloud layer at the layer that
        # has the peak CF.
        cloud_mask_profile_full[:] = np.random.uniform(0,1,(nlayers,)) < CF_profile_scaled
    # (if clear, entire cloud mask profile remains 0.)
    
    # If number of cloud levels is < 37, keep cloud mask profile as is
    if np.sum(cloud_mask_profile_full) < 37:
        cloud_mask_profile = cloud_mask_profile_full
        
    # If number of cloud levels is >= 37, remove cloud layers with the smallest
    # (qi + ql) values, to get down to 36 cloud levels.
    else:
        # In determining the 36 largest qcloud (qi + ql) values, consider *only* 
        # the values where cloud mask = 1.
        qi_mask = np.where(cloud_mask_profile_full == 1, QI_profile, 0)
        ql_mask = np.where(cloud_mask_profile_full == 1, QL_profile, 0)
        qcloud_thresh = np.sort(qi_mask + ql_mask)[::-1][35]
        cloud_mask_profile = np.where((qi_mask + ql_mask) >= qcloud_thresh, 1, 0)
        
        # Break "ties" where more than one level has qi_mask + ql_mask == qcloud_thresh
        # - Set cloud_mask equal to 0 in these cases
        if np.sum(cloud_mask_profile) > 36:
            cloud_mask_profile[(qi_mask + ql_mask) == qcloud_thresh] = 0
    
    # Keep data from the first level with pressure > surface pressure, because it 
    # contains unique data (different from first above-surface level), and shouldn't 
    # cause PCRTM to break.
    below_sfc_ixs = np.where(pres_profile > surface_pres)[0][1:]
    cloud_mask_profile[below_sfc_ixs] = 0
    CF_profile_orig[below_sfc_ixs] = 0
    CF_profile_corr[below_sfc_ixs] = 0
    QI_profile[below_sfc_ixs] = 0
    QL_profile[below_sfc_ixs] = 0
            
    return cloud_mask_profile, CF_profile_orig, CF_profile_corr, QI_profile, QL_profile


def apply_cloud_mask(data_product_specs,
                     CF_profiles_allscenes,
                     QI_profiles_allscenes,
                     QL_profiles_allscenes,
                     pres_profile,
                     surface_pres_allscenes,
                     random_seed=139):
    """
    Apply cloud mask calculation to cloud fraction profiles for all TIRS scenes
    in a given orbit file.
    
    Parameters
    ----------
    data_product_specs : dict
        Aux-Met product specs (including SimTruth group).
    CF_profs_allscenes : numpy.ndarray
        Cloud fraction profiles for all TIRS scenes in a given orbit file.
        Dimensions: (atrack, xtrack, zlevels)
    QI_profs_allscenes : numpy.ndarray
        QI profiles for all TIRS scenes in a given orbit file.
        Dimensions: (atrack, xtrack, zlevels)
    QL_profs_allscenes : numpy.ndarray
        QL profiles for all TIRS scenes in a given orbit file.
        Dimensions: (atrack, xtrack, zlevels)
    pres_profile : numpy.ndarray
        PCRTM pressure levels.
        Dimensions: (zlevels)
    surface_pres_allscenes : numpy.ndarray
        Surface pressure for all TIRS scenes in a given orbit file.
        Dimensions: (atrack, xtrack)
    random_seed : int
        Sets the random seed to generate the same set of random numbers with
        every run.

    Returns
    -------
    cloud_mask_corr_3d : numpy.ndarray
        Cloud mask profiles for all TIRS scenes in a given orbit file, with 
        cloud correlation scale correction applied.
        Dimensions: (atrack, xtrack, zlevels)
    CF_adj_3d : numpy.ndarray
        Cloud fraction adjusted with zeroed out below-surface layers.
        Dimensions: (atrack, xtrack, zlevels)
    CF_corr_adj_3d : numpy.ndarray
        Cloud fraction adjusted with zeroed out below-surface layers, with 
        cloud correlation scale correction applied.
        Dimensions: (atrack, xtrack, zlevels)
    QI_adj_3d : numpy.ndarray
        QI adjusted with zeroed out below-surface layers.
        Dimensions: (atrack, xtrack, zlevels)
    QL_adj_3d : numpy.ndarray
        QL adjusted with zeroed out below-surface layers.
        Dimensions: (atrack, xtrack, zlevels)

    """
    np.random.seed(random_seed)
    
    orbit_shape_3d = CF_profiles_allscenes.shape
    
    cloud_mask_corr_3d = np.full(
        orbit_shape_3d,
        data_product_specs["SimTruth"]["cloud_mask_profile_correlated"]["fill_value"],
        dtype=data_product_specs["SimTruth"]["cloud_mask_profile_correlated"]["np_dtype"]
        )
    CF_adj_3d = np.full(
        orbit_shape_3d,
        data_product_specs["SimTruth"]["cloud_fraction_profile"]["fill_value"],
        dtype=data_product_specs["SimTruth"]["cloud_fraction_profile"]["np_dtype"]
        )
    CF_corr_adj_3d = np.full(
        orbit_shape_3d,
        data_product_specs["SimTruth"]["cloud_fraction_profile_correlated"]["fill_value"],
        dtype=data_product_specs["SimTruth"]["cloud_fraction_profile_correlated"]["np_dtype"]
        )
    QI_adj_3d = np.full(
        orbit_shape_3d,
        data_product_specs["SimTruth"]["qi_profile"]["fill_value"],
        dtype=data_product_specs["SimTruth"]["qi_profile"]["np_dtype"]
        )
    QL_adj_3d = np.full(
        orbit_shape_3d,
        data_product_specs["SimTruth"]["ql_profile"]["fill_value"],
        dtype=data_product_specs["SimTruth"]["ql_profile"]["np_dtype"]
        )
    
    for atrack_ix, xtrack_ix in np.ndindex(orbit_shape_3d[:-1]):        
        cloud_mask_corr_3d[atrack_ix, xtrack_ix],\
        CF_adj_3d[atrack_ix, xtrack_ix],\
        CF_corr_adj_3d[atrack_ix, xtrack_ix],\
        QI_adj_3d[atrack_ix, xtrack_ix],\
        QL_adj_3d[atrack_ix, xtrack_ix] = cloud_mask_calc(
            atrack_ix,
            xtrack_ix,
            CF_profiles_allscenes[atrack_ix, xtrack_ix],
            cloud_mask_corr_3d,
            QI_profiles_allscenes[atrack_ix, xtrack_ix],
            QL_profiles_allscenes[atrack_ix, xtrack_ix],
            pres_profile,
            surface_pres_allscenes[atrack_ix, xtrack_ix]
            )
    
    return cloud_mask_corr_3d, CF_adj_3d, CF_corr_adj_3d, QI_adj_3d, QL_adj_3d


def add_estimated_cloudprops(auxmet_data):
    """
    Add estimated cloud properties to Aux-Met output data (only if Aux-Met is
    run in ANC-SimTruth mode). The estimated cloud properties added are:
    cloud flag, cloud optical depth, cloud effective diameter, and cloud 
    pressure profile.

    Parameters
    ----------
    auxmet_data : dict
        Dictionary containing Aux-Met output data.

    Returns
    -------
    auxmet_data : dict
        Dictionary containing Aux-Met output data.

    """
    
    # Get necessary inputs from Aux-Met data dictionary
    temp = auxmet_data['temp_profile'][:].astype(np.float32)
    plev = auxmet_data['pressure_profile'][:]
    q = auxmet_data['wv_profile'][:].astype(np.float32)
    cldmsk = auxmet_data['cloud_mask_profile_correlated'][:].astype(np.float32)
    qi = auxmet_data['qi_profile'][:].astype(np.float32)
    ql = auxmet_data['ql_profile'][:].astype(np.float32)
    # pressure = auxmet_data['pressure_profile'][:]
    
    # Get array shape from temperature profile data
    at,xt,lev = auxmet_data['temp_profile'].shape
    
    #create arrays for cloud variables
    cld_temp = np.zeros((at,xt,lev-1)) - 999.0
    cld_q = np.zeros((at,xt,lev-1)) - 999.0
    cld_od = np.zeros((at,xt,lev-1)) 
    cld_flag = np.zeros((at,xt,lev-1))
    cld_de = np.zeros((at,xt,lev-1)) 
    cld_dp = np.zeros((at,xt,lev-1)) - 999.0
    
    for i in range(100):
        cld_temp[:,:,i] = ((temp[:,:,i] + temp[:,:,i+1])/2.0) - 273.15
        cld_dp[:,:,i] = (plev[i] + plev[i+1])/2.0
        cld_q[:,:,i] = ((q[:,:,i] + q[:,:,i+1])/2.0)

    #find LWC in g/m^3
    Mdry= 28.9644 #g/mol
    Mh2o = 18.01528 #g/mol
    R = 8.3144598 #J/(mol*K)
    eps = Mh2o/Mdry
    Rdry = R/Mdry #J/(g*K)
    Rmoist = Rdry*(1+((1-eps)*q/(1000.0*eps))) #J/(g*K)
    
    Tv =  (cld_temp+273.15)*(1+((1-eps)*cld_q/(1000.0*eps))) # K
    
    LWC = cldmsk*ql*100.0*plev/(Rmoist*temp*1000.0) #g/m^3
    IWC = cldmsk*qi*100.0*plev/(Rmoist*temp*1000.0) #g/m^3
    
    rho_ice = 917 #kg/m^3
    rho_liq = 997 #kg/m^3
    
    #parameterized based on cloud temperature (S-C Ou, K-N. Liou, 1995);
    de_ice_all = 326.3 + 12.42*cld_temp + 0.197*cld_temp**2 + 0.0012*cld_temp**3
    #PCRTM max is 180 for ice effective diameter
    de_large_msk = de_ice_all > 180.0
    de_small_msk = de_ice_all < 10.0
    de_ice = copy.deepcopy(de_ice_all)
    de_ice[de_large_msk] = 180.0
    de_ice[de_small_msk] = 10.0
    
    #set effective diameter of liquid to 20 um
    de_liq = 20.0
    tot_clr = 0
    tot = 0
    #loop over the at and xt dimensions
    for i in range(at):
        for j in range(xt):
            tot = tot+1
            #this is for the cloud properties
            if np.sum(cldmsk[i,j,:]) > 0:
                for w in range(lev-1):
                    #find cloud properties if cldmsk =1
                    #for 101 shaped variables use the w+1 index
                    #this will shift the cloud away from the surface by a level
                    if cldmsk[i,j,w+1] == 1:
                    
                        #use hypsometric equation to get layer thickness
                        dz = 1000.0*Rdry*Tv[i,j,w]*np.log(plev[w+1]/plev[w])/9.81
                        tau_liq_all = 1000.0*3.0*LWC[i,j,w+1]*dz/(2.0*rho_liq*(de_liq/2.0))
                        tau_ice_all = 1000.0*3.0*IWC[i,j,w+1]*dz/(2.0*rho_ice*(de_ice[i,j,w]/2.0))
                        
                        #limit OD of liquid to 100 and ice to 20
                        # defined by PCRTM limits
                        if tau_liq_all > 100.0:
                            tau_liq = 100.0
                        else:
                            tau_liq = copy.deepcopy(tau_liq_all)
                        if tau_ice_all > 20.0:
                            tau_ice = 20.0
                        else:
                            tau_ice = copy.deepcopy(tau_ice_all)
                        
                        #can't model mixed clouds, so. . .
                        #if liquid optical depth dominates then set liq values
                        #use the original tau (before limits) to set the phase
                        if tau_liq_all > tau_ice_all:
                            cld_od[i,j,w] =  tau_liq
                            cld_flag[i,j,w] =  2
                            cld_de[i,j,w] = de_liq 
                        #if ice optical depth dominates then set ice values
                        else:
                            cld_od[i,j,w] =  tau_ice
                            cld_flag[i,j,w] =  1
                            cld_de[i,j,w] = de_ice[i,j,w] 
            
            # Count the total number of clear scenes in the granule
            else:
                tot_clr=tot_clr+1

    return cld_flag, cld_od, cld_de, cld_dp
