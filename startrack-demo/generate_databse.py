"""
This example generates a tetra3 database from a star catalogue. You must have the catalogue file
hip_main.dat in the same directory as tetra3.py to run this example. You can download it from
https://cdsarc.u-strasbg.fr/ftp/cats/I/239/
"""

import sys
#sys.path.append('..')

import starTrack

# Create instance without loading any database.
t3 = starTrack.Tetra3(load_database=None)

# Generate and save database.
t3.generate_database(save_as='t3_fov20-30_mag8', max_fov=30, min_fov=20,
                     star_max_magnitude=8, star_catalog='hip_main')