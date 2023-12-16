import numpy as np
from pathlib import Path
import datetime
import csv
import itertools
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist

_MAGIC_RAND = np.uint64(2654435761)

def _insert_at_index(pattern, hash_index, table):
    """Inserts to table with quadratic probing. Returns table index where pattern was inserted."""
    max_ind = np.uint64(table.shape[0])
    hash_index = np.uint64(hash_index)
    for c in itertools.count():
        c = np.uint64(c)
        i = (hash_index + c*c) % max_ind
        if all(table[i, :] == 0):
            table[i, :] = pattern
            return i

def pack_properties(pattern_mode, pattern_size, pattern_bins, pattern_max_error, max_fov, min_fov,
                         star_catalog, epoch_equinox, epoch_proper_motion, pattern_stars_per_fov,
                         verification_stars_per_fov, star_max_magnitude, simplify_pattern, range_ra, range_dec,
                         presort_patterns):
    
    """Packs the provided database properties into a NumPy structured array.

    Args:
        pattern_mode: String specifying the pattern mode (e.g., edge_ratio).
        pattern_size: Integer representing the pattern size.
        ... (remaining arguments)

    Returns:
        A NumPy structured array containing the packed database properties.

    Notes:
        Each property is assigned a specific data type for efficient storage and retrieval.
        The returned array can be easily saved and loaded using NumPy functions.
    """

    dtype = [('pattern_mode', 'U64'),
              ('pattern_size', np.uint16),
              ('pattern_bins', np.uint16),
              ('pattern_max_error', np.float32),
              ('max_fov', np.float32),
              ('min_fov', np.float32),
              ('star_catalog', 'U64'),
              ('epoch_equinox', np.uint16),
              ('epoch_proper_motion', np.float32),
              ('pattern_stars_per_fov', np.uint16),
              ('verification_stars_per_fov', np.uint16),
              ('star_max_magnitude', np.float32),
              ('simplify_pattern', bool),
              ('range_ra', np.float32, (2,)),
              ('range_dec', np.float32, (2,)),
              ('presort_patterns', bool)]

    return np.array((pattern_mode, pattern_size, pattern_bins, pattern_max_error, max_fov, min_fov,
                      star_catalog, epoch_equinox, epoch_proper_motion, pattern_stars_per_fov,
                      verification_stars_per_fov, star_max_magnitude, simplify_pattern, range_ra, range_dec,
                      presort_patterns), dtype=dtype)
                    
def _key_to_index(key, bin_factor, max_index):
    """Get hash index for a given key. Can be length p list or n by p array."""
    key = np.uint64(key)
    bin_factor = np.uint64(bin_factor)
    max_index = np.uint64(max_index)
    # If p is the length of the key (default 5) and B is the number of bins (default 50,
    # calculated from max error), this will first give each key a unique index from
    # 0 to B^p-1, then multiply by large number and modulo to max index to randomise.
    if key.ndim == 1:
        hash_indices = np.sum(key*bin_factor**np.arange(len(key), dtype=np.uint64),
                              dtype=np.uint64)
    else:
        hash_indices = np.sum(key*bin_factor**np.arange(key.shape[1], dtype=np.uint64)[None, :],
                              axis=1, dtype=np.uint64)
    with np.errstate(over='ignore'):
        hash_indices = (hash_indices*_MAGIC_RAND) % max_index
    return hash_indices

    
def generate_database(max_fov, min_fov=None, save_as=None,
                          star_catalog='hip_main.csv', pattern_stars_per_fov=10,
                          verification_stars_per_fov=30, star_max_magnitude=7,
                          pattern_max_error=.005, simplify_pattern=False,
                          range_ra=None, range_dec=None,
                          presort_patterns=True, save_largest_edge=False,
                          multiscale_step=1.5, epoch_proper_motion='now'):
    
    max_fov = np.deg2rad(float(max_fov))
    if min_fov is None:
        min_fov = max_fov
    else:
        min_fov = np.deg2rad(float(min_fov))
    pattern_stars_per_fov = int(pattern_stars_per_fov)
    verification_stars_per_fov = int(verification_stars_per_fov)
    star_max_magnitude = float(star_max_magnitude)
    pattern_size = 4
    pattern_bins = round(1/4/pattern_max_error)
    presort_patterns = bool(presort_patterns)
    save_largest_edge = bool(save_largest_edge)   

    if epoch_proper_motion is None or str(epoch_proper_motion).lower() == 'none':
        epoch_proper_motion = None
        print('Proper motions will not be considered')
    elif str(epoch_proper_motion).lower() == 'now':
        epoch_proper_motion = datetime.date.today().year
        print('Proper motion epoch set to now: ' + str(epoch_proper_motion))
    else:
        raise ValueError('epoch_proper_motion value %s is forbidden' % epoch_proper_motion)

    #ensure the code does not proceed without necessary catalog file
    catalog_file_full_pathname = Path(__file__).parent / star_catalog
    assert catalog_file_full_pathname.exists(), 'No star catalogue found at ' + str(catalog_file_full_pathname)

    # Calculate number of star catalog entries:
    num_entries = sum(1 for _ in open(catalog_file_full_pathname))
    epoch_equinox = 2000
    pm_origin = 1991.25

    print('Loading catalogue %s with %s star entries.' %(star_catalog, num_entries))

    if epoch_proper_motion is None:
        # If pm propagation was disabled, set end date to origin
        epoch_proper_motion = pm_origin
        print('Using catalog RA/Dec %s epoch; not propagating proper motions from %s.' %
                            (epoch_equinox, pm_origin))
    else:
        print('Using catalog RA/Dec %s epoch; propagating proper motions from %s to %s.' %
                            (epoch_equinox, pm_origin, epoch_proper_motion))

        # Preallocate star table:
    star_table = np.zeros((num_entries, 6), dtype=np.float32)
    # Preallocate ID table
    star_catID = np.zeros(num_entries, dtype=np.uint32)
    # Read magnitude, RA, and Dec from star catalog:
    # The Hipparcos and Tycho catalogs uses International Celestial
    # Reference System (ICRS) which is essentially J2000. See
    # https://cdsarc.u-strasbg.fr/ftp/cats/I/239/version_cd/docs/vol1/sect1_02.pdf
    # section 1.2.1 for details.
    with open(catalog_file_full_pathname, 'r') as star_catalog_file:
        reader = itertools.islice(csv.reader(star_catalog_file, delimiter=','), 1, None)
        incomplete_entries = 0
        for (i, entry) in enumerate(reader):
            # Skip this entry if mag, ra, or dec are empty.
            if entry[5].isspace() or entry[5] == '' or entry[8].isspace() or entry[8] == '' or entry[9].isspace() or entry[9] == '':
                incomplete_entries += 1
                continue
            # If propagating, skip if proper motions are empty.
            if epoch_proper_motion != pm_origin and (entry[12].isspace() or entry[13].isspace()):
                incomplete_entries += 1
                continue
            mag = float(entry[5])
            if mag > star_max_magnitude:
                continue
            # RA/Dec in degrees at 1991.25 proper motion start.
            alpha = float(entry[8])
            delta = float(entry[9])
            cos_delta = np.cos(np.deg2rad(delta))

            mu_alpha = 0
            mu_delta = 0
            if epoch_proper_motion != pm_origin:
                # Pick up proper motion terms. Note that the pmRA field is
                # "proper motion in right ascension"; see
                # https://en.wikipedia.org/wiki/Proper_motion; see also section
                # 1.2.5 in the cdsarc.u-strasbg document cited above.

                # The 1000/60/60 term converts milliarcseconds per year to
                # degrees per year.
                mu_alpha_cos_delta = float(entry[12])/1000/60/60
                mu_delta = float(entry[13])/1000/60/60

                # Divide the pmRA field by cos_delta to recover the RA proper
                # motion rate. Note however that near the poles (delta near plus
                # or minus 90 degrees) the cos_delta term goes to zero so dividing
                # by cos_delta is problematic there.
                # Section 1.2.9 of the cdsarc.u-strasbg document cited above
                # outlines a change of coordinate system that can overcome
                # this problem; we simply punt on proper motion near the poles.
                if cos_delta > 0.1:
                    mu_alpha = mu_alpha_cos_delta / cos_delta
                else:
                    # abs(dec) > ~84 degrees. Ignore proper motion.
                    mu_alpha = 0
                    mu_delta = 0

            ra  = np.deg2rad(alpha + mu_alpha * (epoch_proper_motion - pm_origin))
            dec = np.deg2rad(delta + mu_delta * (epoch_proper_motion - pm_origin))
            star_table[i,:] = ([ra, dec, 0, 0, 0, mag])
            # Find ID, depends on the database
            star_catID[i] = np.uint32(entry[1])
            
        if incomplete_entries:
            print('Skipped %i incomplete entries.' % incomplete_entries)

    # Remove entries in which RA and Dec are both zero
    # (i.e. keep entries in which either RA or Dec is non-zero)
    kept = np.logical_or(star_table[:, 0]!=0, star_table[:, 1]!=0)
    star_table = star_table[kept, :]
    brightness_ii = np.argsort(star_table[:, -1])
    star_table = star_table[brightness_ii, :]  # Sort by brightness
    num_entries = star_table.shape[0]
    # Trim and order catalogue ID array to match
    star_catID = star_catID[kept][brightness_ii]
    print('Loaded ' + str(num_entries) + ' stars with magnitude below ' \
        + str(star_max_magnitude) + '.')

    # If desired, clip out only a specific range of ra and/or dec for a partial coverage database
    if range_ra is not None:
        range_ra = np.deg2rad(range_ra)
        if range_ra[0] < range_ra[1]: # Range does not cross 360deg discontinuity
            kept = np.logical_and(star_table[:, 0] > range_ra[0], star_table[:, 0] < range_ra[1])
        else:
            kept = np.logical_or(star_table[:, 0] > range_ra[0], star_table[:, 0] < range_ra[1])
        star_table = star_table[kept, :]
        num_entries = star_table.shape[0]
        # Trim down catalogue ID to match
        star_catID = star_catID[kept]
        print('Limited to RA range ' + str(np.rad2deg(range_ra)) + ', keeping ' + str(num_entries) + ' stars.')
    if range_dec is not None:
        range_dec = np.deg2rad(range_dec)
        if range_dec[0] < range_dec[1]: # Range does not cross +/-90deg discontinuity
            kept = np.logical_and(star_table[:, 1] > range_dec[0], star_table[:, 1] < range_dec[1])
        else:
            kept = np.logical_or(star_table[:, 1] > range_dec[0], star_table[:, 1] < range_dec[1])
        star_table = star_table[kept, :]
        num_entries = star_table.shape[0]
        # Trim down catalogue ID to match
        star_catID = star_catID[kept]
        print('Limited to DEC range ' + str(np.rad2deg(range_dec)) + ', keeping ' \
            + str(num_entries) + ' stars.')

    # Calculate star direction vectors:
    for i in range(0, num_entries):
        vector = np.array([np.cos(star_table[i, 0])*np.cos(star_table[i, 1]),
                            np.sin(star_table[i, 0])*np.cos(star_table[i, 1]),
                            np.sin(star_table[i, 1])])
        star_table[i, 2:5] = vector
    # Insert all stars in a KD-tree for fast neighbour lookup
    print('Trimming database to requested star density.')
    all_star_vectors = star_table[:, 2:5]
    vector_kd_tree = KDTree(all_star_vectors)

    # Bool list of stars, indicating it will be used in the database
    keep_for_patterns = np.full(num_entries, False)
    # Keep the first one and skip index 0 in loop
    keep_for_patterns[0] = True    

    # Calculate set of FOV scales to create patterns at
    fov_ratio = max_fov/min_fov
    def logk(x, k):
        return np.log(x) / np.log(k)
    fov_divisions = np.ceil(logk(fov_ratio, multiscale_step)).astype(int) + 1
    if fov_ratio < np.sqrt(multiscale_step):
        pattern_fovs = [max_fov]
    else:
        pattern_fovs = np.exp2(np.linspace(np.log2(min_fov), np.log2(max_fov), fov_divisions))
    print('Generating patterns at FOV scales: ' + str(np.rad2deg(pattern_fovs)))

    # List of patterns found, to be populated in loop
    pattern_list = set([])
    # initialize pattern, which will contain pattern_size star ids
    pattern = [None] * pattern_size
    for pattern_fov in reversed(pattern_fovs):
        keep_at_fov = np.full(num_entries, False)
        if fov_divisions == 1:
            # Single scale database, trim to min_fov, make patterns up to max_fov
            pattern_stars_separation = .6 * min_fov / np.sqrt(pattern_stars_per_fov)
        else:
            # Multiscale database, trim and make patterns iteratively at smaller FOVs
            pattern_stars_separation = .6 * pattern_fov / np.sqrt(pattern_stars_per_fov)

        print('At FOV ' + str(round(np.rad2deg(pattern_fov), 5)) + ' separate stars by ' \
            + str(np.rad2deg(pattern_stars_separation)) + 'deg.')
        # Loop through all stars in database, create set of of pattern stars
        # Note that each loop just adds stars to the previous version (between old ones)
        # so we can skip all indices already kept
        for star_ind in range(num_entries):
            vector = all_star_vectors[star_ind, :]
            # Check if any kept stars are within the pattern checking separation
            within_pattern_separation = vector_kd_tree.query_ball_point(vector,
                pattern_stars_separation)
            occupied_for_pattern = np.any(keep_at_fov[within_pattern_separation])
            # If there isn't a star too close, add this to the table and carry on
            # Add to both this FOV specifically and the general table.
            if not occupied_for_pattern:
                keep_for_patterns[star_ind] = True
                keep_at_fov[star_ind] = True

        print('Stars for patterns at this FOV: ' + str(np.sum(keep_at_fov)) + '.')
        print('Stars for patterns total: ' + str(np.sum(keep_for_patterns)) + '.')
        # Clip out table of the kept stars
        pattern_star_table = star_table[keep_at_fov, :]
        # Insert into KD tree for neighbour lookup
        pattern_kd_tree = KDTree(pattern_star_table[:, 2:5])
        # List of stars available (not yet used to create patterns)
        available_stars = [True] * pattern_star_table.shape[0]
        # Index conversion from pattern_star_table to main star_table
        pattern_index = np.nonzero(keep_at_fov)[0].tolist()

        # Loop through all pattern stars
        for pattern[0] in range(pattern_star_table.shape[0]):
            # Remove star from future consideration
            available_stars[pattern[0]] = False
            # Find all neighbours within FOV, keep only those not removed
            vector = pattern_star_table[pattern[0], 2:5]
            if simplify_pattern:
                neighbours = pattern_kd_tree.query_ball_point(vector, pattern_fov/2)
            else:
                neighbours = pattern_kd_tree.query_ball_point(vector, pattern_fov)
            available = [available_stars[i] for i in neighbours]
            neighbours = np.compress(available, neighbours)
            # Check all possible patterns
            for pattern[1:] in itertools.combinations(neighbours, pattern_size - 1):
                if simplify_pattern:
                    # Add to database
                    pattern_list.add(tuple(pattern_index[i] for i in pattern))
                    if len(pattern_list) % 1000000 == 0:
                        print('Generated ' + str(len(pattern_list)) + ' patterns so far.')
                else:
                    # Unpack and measure angle between all vectors
                    vectors = pattern_star_table[pattern, 2:5]
                    dots = np.dot(vectors, vectors.T)
                    if dots.min() > np.cos(pattern_fov):
                        # Maximum angle is within the FOV limit, append with original index
                        pattern_list.add(tuple(pattern_index[i] for i in pattern))
                        if len(pattern_list) % 1000000 == 0:
                            print('Generated ' + str(len(pattern_list)) + ' patterns so far.')
    print('Found ' + str(len(pattern_list)) + ' patterns in total.')

    # Repeat process, add in missing stars for verification task
    verification_stars_separation = .6 * min_fov / np.sqrt(verification_stars_per_fov)
    keep_for_verifying = keep_for_patterns.copy()
    for star_ind in range(1, num_entries):
        vector = all_star_vectors[star_ind, :]
        # Check if any kept stars are within the pattern checking separation
        within_verification_separation = vector_kd_tree.query_ball_point(vector,
            verification_stars_separation)
        occupied_for_verification = np.any(keep_for_verifying[within_verification_separation])
        if not occupied_for_verification:
            keep_for_verifying[star_ind] = True
    print('Total stars for verification: ' + str(np.sum(keep_for_verifying)) + '.')

    # Trim down star table and update indexing for pattern stars
    star_table = star_table[keep_for_verifying, :]
    pattern_index = (np.cumsum(keep_for_verifying)-1)
    pattern_list = pattern_index[np.array(list(pattern_list))].tolist()
    # Trim catalogue ID to match
    star_catID = star_catID[keep_for_verifying]
    # Create all pattens by calculating and sorting edge ratios and inserting into hash table
    print('Start building catalogue.')
    catalog_length = 2 * len(pattern_list)
    # Determine type to make sure the biggest index will fit, create pattern catalogue
    max_index = np.max(np.array(pattern_list))
    if max_index <= np.iinfo('uint8').max:
        pattern_catalog = np.zeros((catalog_length, pattern_size), dtype=np.uint8)
    elif max_index <= np.iinfo('uint16').max:
        pattern_catalog = np.zeros((catalog_length, pattern_size), dtype=np.uint16)
    else:
        pattern_catalog = np.zeros((catalog_length, pattern_size), dtype=np.uint32)
    print('Catalog size ' + str(pattern_catalog.shape) + ' and type ' + str(pattern_catalog.dtype) + '.')

    pattern_largest_edge = np.zeros(catalog_length, dtype=np.float16)
    print('Storing largest edges as type ' + str(pattern_largest_edge.dtype))

    # Indices to extract from dot product matrix (above diagonal)
    upper_tri_index = np.triu_indices(pattern_size, 1)

    # Go through each pattern and insert to the catalogue
    for (index, pattern) in enumerate(pattern_list):
        if index % 1000000 == 0 and index > 0:
            print('Inserting pattern number: ' + str(index))

        # retrieve the vectors of the stars in the pattern
        vectors = star_table[pattern, 2:5]

        # implement more accurate angle calculation
        edge_angles_sorted = np.sort(2 * np.arcsin(.5 * pdist(vectors)))
        edge_ratios = edge_angles_sorted[:-1] / edge_angles_sorted[-1]

        # convert edge ratio float to hash code by binning
        hash_code = tuple((edge_ratios * pattern_bins).astype(int))
        hash_index = _key_to_index(hash_code, pattern_bins, catalog_length)

        if presort_patterns:
            # find the centroid, or average position, of the star pattern
            pattern_centroid = np.mean(vectors, axis=0)
            # calculate each star's radius, or Euclidean distance from the centroid
            pattern_radii = cdist(vectors, pattern_centroid[None, :]).flatten()
            # use the radii to uniquely order the pattern, used for future matching
            pattern = np.array(pattern)[np.argsort(pattern_radii)]

        table_index = _insert_at_index(pattern, hash_index, pattern_catalog)
        # Store as milliradian to better use float16 range
        pattern_largest_edge[table_index] = edge_angles_sorted[-1]*1000

    print('Finished generating database.')
    print('Size of uncompressed star table: %i Bytes.' %star_table.nbytes)
    print('Size of uncompressed pattern catalog: %i Bytes.' %pattern_catalog.nbytes)


    db_props = {}

    star_table = star_table
    star_catalog_IDs = star_catID
    pattern_catalog = pattern_catalog
    pattern_largest_edge = pattern_largest_edge
    db_props['pattern_mode'] = 'edge_ratio'
    db_props['pattern_size'] = pattern_size
    db_props['pattern_bins'] = pattern_bins
    db_props['pattern_max_error'] = pattern_max_error
    db_props['max_fov'] = np.rad2deg(max_fov)
    db_props['min_fov'] = np.rad2deg(min_fov)
    db_props['star_catalog'] = star_catalog
    db_props['epoch_equinox'] = epoch_equinox
    db_props['epoch_proper_motion'] = epoch_proper_motion
    db_props['pattern_stars_per_fov'] = pattern_stars_per_fov
    db_props['verification_stars_per_fov'] = verification_stars_per_fov
    db_props['star_max_magnitude'] = star_max_magnitude
    db_props['simplify_pattern'] = simplify_pattern
    db_props['range_ra'] = range_ra
    db_props['range_dec'] = range_dec
    db_props['presort_patterns'] = presort_patterns
    print(db_props)

    props_packed = pack_properties(**db_props)

    if save_as is not None:
        print('Saving generated database as: ' + str(save_as))
        """Save database to file.

    Args:
        path (str or pathlib.Path): The file to save to. If given a str, the file will be saved
            in the tetra3/data directory. If given a pathlib.Path, this path will be used
            unmodified. The suffix .npz will be added.
    """
        path = ""
        if isinstance(path, str):
            print('String given, append to tetra3 directory')
            path = (Path(__file__).parent / 'data' / path).with_suffix('.npz')
        else:
            print('Not a string, use as path directly')
            path = Path(path).with_suffix('.npz')
        
        print('Saving database to: ' + str(path))

        
        print('Packed properties into: ' + str(props_packed))
        print('Saving as compressed numpy archive')

        to_save = {'star_table': star_table,
            'pattern_catalog': pattern_catalog,
            'props_packed': props_packed}
        to_save['pattern_largest_edge'] = pattern_largest_edge
        to_save['star_catalog_IDs'] = star_catalog_IDs

        np.savez_compressed(path, **to_save)
    else:
        print('Skipping database file generation.')
        
   
def main():
    generate_database(max_fov=30, min_fov=10, star_max_magnitude=7, save_as='default_database')

if __name__ == "__main__":
    main()

