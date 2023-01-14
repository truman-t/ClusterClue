import pandas as pd
from collections import defaultdict
import pyvo as vo
from sklearn.preprocessing import RobustScaler
import hdbscan
from astropy.coordinates import SkyCoord
import numpy as np
from astropy.table import Table
import os
from astroquery.gaia import Gaia
from random import randint, shuffle
import time
import shutil
from tensorflow import device
from matplotlib.pyplot import hist2d, close


def cluster_query(ra, de, radius, distance, r_fraction=3, dist_fraction=0.4, dist_windows_cut=4000):
    """
    Returns the columns in table gedr3dist.litewithdist nedded to run the first part of ClusterClue pipeline for the specified cylinder of search.
    It is assumed that the input parameters follow the same epoch as Gaia DR3 (2016). 
    
    Arguments
    ---------
    ra: right ascension center of the cylinder (degrees)
    de: declination center of the cylinder (degrees)
    radius: radius of the cluster (degrees)
    r_fraction: fraction of radius that is used as the radius of the cylinder.
    distance: distance to the center of the cylinder (parsecs)
    dist_fraction: porcentage uncertainty in the distance to be use as half the cylinder high
    dist_windows_cut: upper floor in the distance window (parsecs)
    
    Returns
    -------
    pyvo.dal.tap.TAPResults
        a table with all the stars in litewithdist (Gaia data) within the cylinder
    """
    # Get query parameters
    r = radius*r_fraction
    d_window = distance*dist_fraction
    if d_window>dist_windows_cut:
        d_window = dist_windows_cut
    d_min = distance - d_window
    d_max = distance + d_window
    
    # define ADQL query
    query = "SELECT source_id, ra, dec, parallax, parallax_error, r_med_photogeo, pmra, pmdec FROM gedr3dist.litewithdist WHERE phot_g_mean_mag<=18 AND 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {_ra}, {_de}, {_r})) AND r_hi_photogeo>={_d_min} AND r_lo_photogeo<={_d_max}"
    query = query.format(_ra=ra, _de=de, _r=r, _d_min=d_min, _d_max=d_max)
    # define service
    gavo = vo.dal.TAPService("https://dc.zah.uni-heidelberg.de/__system__/tap/run")
    # run query
    try:
        job = gavo.submit_job(query, maxrec=10000000)
        job.run()
        while (job.phase == 'EXECUTING') or (job.phase == 'QUEUED'):
            time.sleep(0.5)
        ans = job.fetch_result()
    except:
        return False
    return ans
    
    
    
    
def preprocess(data, pos_weight=0.75):
    """
    Takes the output of a pyvo query and returns the input array to HDBSCAN
    
    Arguments
    ---------
    data: output of pyvo query (TAPResults)
    pos_weight: weight factor apply to positional data
    
    Returns
    -------
    numpy.ndarray
        input for HDBSCAN 
    """
    # Go from spherical to cartesian
    c = SkyCoord(ra=data['ra'].data, dec=data['dec'].data, distance=data['r_med_photogeo'].data, frame='icrs', unit=('deg', 'deg', 'pc'))
    c.representation_type = 'cartesian'
    
    # Scale the input data
    input_data = RobustScaler().fit_transform(np.transpose(np.array([c.x.value, c.y.value, c.z.value, data['pmra'].data, data['pmdec'].data])))
    
    # Appply weights
    w = np.array([pos_weight, pos_weight, pos_weight, 1, 1])/(((3*pos_weight)+2)/5)
    return input_data*np.sqrt(w)




def get_cmd_data_gaia(star_list):
    """
    queries Gaia DR3 catalog for a list of stars and returns CMD data.
    
    Arguments
    ---------
    star_list: list of Gaia DR3 ids 
    
    Returns
    -------
    (colours, magnitudes)
    """
    t = Table()
    t['dr3id'] = star_list
    job = Gaia.launch_job_async("select source_id, phot_g_mean_mag, bp_rp from gaiadr3.gaia_source as main join tap_upload.myTable as my on main.source_id=my.dr3id", 
                                upload_resource=t, upload_table_name="myTable", verbose=False)
    if job.is_finished:
        try:
            data = job.get_results()
        except:
            return False
    return data




def cmd_preprocess(colour, magnitude, max_mag=18, cmd_size=64):
    """
    Given CMD data (colour and magnitudes) for a list of stars, return a matrix that is a 2D histogram of the CMD nomalized logarithmically.
    
    Arguments
    ---------
    colour: list of colours of stars
    magnitudes: list of magnitudes of stars
    max_mag: maximun magnitude to display in the CMD
    cmd_size: size of the 2D histogram
    
    Outputs
    -------
    matrix
    """
    # select g magnitudes less than the maximun magnitude and not missing values
    color = np.array(colour)
    mag = np.array(magnitude)
    condition = (pd.notnull(color) & pd.notnull(mag)) & (mag<=max_mag)
    color = color[condition]
    mag = mag[condition]
    # calculate 2d histogram
    hist = hist2d(color, mag, bins=cmd_size)[0]
    # normalize histogram
    ma = max(hist.ravel())
    mi = min(hist.ravel())
    hist = (hist-mi)/(ma-mi)
    # apply logarithmic scaling
    c = 1/np.log(2)
    hist = c*np.log(hist + 1)
    close()
    return hist.transpose()
    
    
    

def clusterClue(ra, de, radius, distance, validator, r_fraction=3, dist_fraction=0.4, dist_windows_cut=4000, pos_weight=0.75, label=''):
    """
    Runs ClusterClue pipeline for the sky region specified.
    If fail during quering the data from databases, False is returned.
    If no clusters (by HDBSCAN or none validated) are found, label is returned.
    
    Arguments
    ---------
    ra: right ascension center of the cylinder (degrees)
    de: declination center of the cylinder (degrees)
    radius: radius of the cluster (degrees)
    r_fraction: fraction of radius that is used as the radius of the cylinder.
    distance: distance to the center of the cylinder (parsecs)
    validator: NN model used as validator.
    dist_fraction: porcentage uncertainty in the distance to be use as half the cylinder high
    dist_windows_cut: upper floor in the distance window (parsecs)
    pos_weight: weight factor apply to positional data
    label: string to identify the clusters found
    
    Returns
    -------
    DataFrame with information about the clusters of the sky region 
    """
    
    # Search astrometric data
    astrometric_data = cluster_query(ra, de, radius, distance, r_fraction, dist_fraction, dist_windows_cut)
    if isinstance(astrometric_data, bool):
        return False
    if len(astrometric_data)<10:
        return False

    # Preprocess data
    hdbscan_input = preprocess(astrometric_data, pos_weight)

    # Run HDBSCAN
    try:
        clusterer1 = hdbscan.HDBSCAN(min_cluster_size=80, cluster_selection_method='leaf', min_samples=10, metric='euclidean', memory='./cachedir1').fit(hdbscan_input)
        clusterer2 = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_method='leaf', min_samples=10, metric='euclidean', memory='./cachedir1').fit(hdbscan_input)
    except:
        return label + 'weird'
    shutil.rmtree('./cachedir1', ignore_errors=True)
    # check if any cluster was found 
    if len(set(clusterer1.labels_))==1 and len(set(clusterer2.labels_))==1:
        return label

    # Retrieve candidates membership list
    offset = 10000
    clusterer2.labels_ = [x + offset if x!=-1 else x for x in clusterer2.labels_]
    candidates = defaultdict(list)
    for c_label in [x for x in set(clusterer1.labels_) if x>=0]:
        c_name = str(c_label)
        candidates[c_name] = [x[0] for x in zip(astrometric_data['source_id'].tolist(), clusterer1.labels_) if x[1]==c_label]
    for c_label in [x for x in set(clusterer2.labels_) if x>=0]:
        c_name = str(c_label)
        candidates[c_name] = [x[0] for x in zip(astrometric_data['source_id'].tolist(), clusterer2.labels_) if x[1]==c_label]
    # Free field stars
    astrometric_data = astrometric_data.to_table().to_pandas()
    astrometric_data['prob80'] = clusterer1.probabilities_
    astrometric_data['prob10'] = clusterer2.probabilities_
    candidates = [(cluster, star) for (cluster, v) in candidates.items() for star in v]
    candidates = pd.DataFrame(candidates, columns=['cluster', 'source_id'])
    result = astrometric_data.set_index('source_id').join(candidates.set_index('source_id'), how='inner') # free field stars
    result.reset_index(inplace=True)
    result.drop_duplicates(inplace=True)
    
    # Filter extremely incompatible candidates
    removed = []
    clusts = result.cluster.unique()
    for c in clusts:
        d = result[result.cluster==c][['parallax', 'r_med_photogeo', 'pmra', 'pmdec', 'ra', 'dec']]
        # First condition. Proper motion dispersion
        dispersion = ((d.pmra.std()**2) + (d.pmdec.std()**2))**(1/2)
        mean_parallax = d.parallax.mean()
        if mean_parallax<=0.67:
            condition = 1
        else:
            condition = 1.49 * mean_parallax
        if dispersion>condition:
            removed.append(c)
            continue
        # Second condition. r50 radius
        c_distance = np.median(d.r_med_photogeo.to_numpy())
        members = SkyCoord(ra=d.ra.tolist(), dec=d.dec.tolist(), distance=[c_distance for x in range(len(d))], frame='icrs', unit=('deg', 'deg', 'pc'))
        members.representation_type = 'cartesian'
        (c_x, c_y, c_z) = (members.x.mean().to('pc').value, members.y.mean().to('pc').value, members.z.mean().to('pc').value)
        center = SkyCoord(x=c_x, y=c_y, z=c_z, frame='icrs', unit=('pc', 'pc', 'pc'), representation_type = 'cartesian')
        distances = center.separation_3d(members)
        distances = np.sort(distances.pc)
        if distances[len(distances)//2 - 1]>20:
            removed.append(c)
    remain = [x for x in clusts if x not in removed]
    # stop if there are no candidates left
    if len(remain)==0:
        return label
    result = result[result.cluster.isin(remain)] # free incompatible candidates
    candidates = result.groupby('cluster')['source_id'].apply(list).to_dict()
    
    # Get CMD data
    cmd_data = get_cmd_data_gaia(result.source_id.tolist())
    if isinstance(cmd_data, bool):
        return False
    cmd_data = cmd_data.to_pandas()

    # Preprocess CMDs
    cmds = [] 
    for cluster, members in candidates.items():
        c_data = cmd_data[cmd_data.source_id.isin(members)][['phot_g_mean_mag', 'bp_rp']]
        cmds.append(cmd_preprocess(c_data.bp_rp.tolist(), c_data.phot_g_mean_mag.tolist()))
    cmds = np.array(cmds).reshape(-1, 64, 64, 1)

    # Validate cluster candidates
    with device('/cpu:0'):
        val_result = validator.predict(cmds, verbose=False)
    val_result = val_result.reshape(-1).tolist()
    clusters_validation = {x[0]:x[1] for x in zip(candidates.keys(), val_result) if x[1]>=0.5} # select validated clusters and their validation score
    # return label if there is no validated clusters
    if len(clusters_validation.keys())==0:
        return label
    result = result[result.cluster.isin(clusters_validation.keys())] # free not validated candidates
    candidates = result.groupby('cluster')['source_id'].apply(list).to_dict()
    
    # Merge clusters
    clusters80 = [str(x) for x in set(clusterer1.labels_) if str(x) in candidates.keys()]
    clusters10 = [str(x) for x in set(clusterer2.labels_) if str(x) in candidates.keys()]
    if len(clusters80)==0 or len(clusters10)==0:
        need_subset_search = False
    else:
        need_subset_search = True
    if need_subset_search:
        clean10_clusters = [x for x in clusters10 if sum([set(candidates[x]).issubset(candidates[y]) for y in clusters80])==0]
        merged_clusters = clean10_clusters + clusters80
        result = result[result.cluster.isin(merged_clusters)]
    
    # Return candidate clusters
    result = result.set_index('source_id').join(cmd_data.set_index('source_id'), how='inner')
    result['validation'] = result.cluster.apply(lambda x: clusters_validation[x])
    result['membership_prob'] = result.apply(lambda row: row['prob80'] if int(row['cluster'])<1000 else row['prob10'], axis=1)
    result['cluster'] = result.cluster.apply(lambda x: label + "_p:" + str(round(ra, 5)) + ',' + str(round(de, 5)) + '_r:' + str(round(radius, 3)) + '_number:' + x)
    result.reset_index(inplace=True)
    result.drop(['prob80', 'prob10'], axis=1, inplace=True)
    result.drop_duplicates(inplace=True)
    return result
