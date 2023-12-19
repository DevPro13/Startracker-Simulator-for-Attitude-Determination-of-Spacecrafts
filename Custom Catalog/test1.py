import numpy as np

# Open the npz file
with np.lib.npyio.NpzFile('data.npz') as f:
    # Get list of stored arrays
    files = f.files

    # Access specific array by name
    array1 = f['star_table']

    # Looping through all stored arrays
    for file in files:
        data = f[file]
        print(f"File: {file}, Data: {data}")

# Close the file automatically at the end of the block
