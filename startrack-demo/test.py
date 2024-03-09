import starTrack

from time import perf_counter as precision_timestamp
from PIL import Image
from PIL import ImageGrab
from pathlib import Path
import outputpresentation, quarterneonCalc
import matplotlib.pyplot as plt
import cv2
import numpy as np

EXAMPLES_DIR = Path(__file__).parent

demo = starTrack.STARTRACK()

path = EXAMPLES_DIR / 'images'

#for centroid matching
a_distortion=[-.2, .1]

optional_features = {'min_sum': 250, 'max_axis_ratio': 1.5}

def centroiding_img(pointz):
        image = cv2.imread("test69.png")
        plt.imshow(image)
        plt.scatter(pointz[:, 1], pointz[:, 0], c="red", marker="o", s=5, alpha=0.3)
        plt.show()
        plt.savefig("test.png")
    
for impath in path.glob('*'):
    print('Solving for image at: ' + str(impath))
    with Image.open(str(impath)) as img:
        assert demo.has_database, 'No database loaded'
        demo._logger.debug('Got solve from image with input: ' + str((img, {**optional_features})))
        (width, height) = img.size[:2]
        demo._logger.debug('Image (height, width): ' + str((height, width)))
        
        # Run star extraction, passing optional_features along
        t0_extract = precision_timestamp()
        centr_data = starTrack.get_centroids_from_image(img, **optional_features)
        t_extract = (precision_timestamp() - t0_extract)*1000
       
        # If we get a tuple, need to use only first element and then reassemble at return
        if isinstance(centr_data, tuple):
            centroids = centr_data[0]
        else:
            centroids = centr_data
        demo._logger.debug('Found this many centroids, in time: ' + str((len(centroids), t_extract)))
        
        # Run centroid solver, passing arguments along (could clean up with optional_features handler)
        solution = demo.solve_from_centroids(centroids, (height, width), 
            fov_estimate=None, fov_max_error=None,
            pattern_checking_stars=8, match_radius=0.01,
            match_threshold=1e-3, solve_timeout=None,
            target_pixel=None, distortion=a_distortion,
            return_matches=False, return_visual=False)
        # Add extraction time to results and return
        solution['T_extract'] = t_extract
        if isinstance(centr_data, tuple):
            final = (solution,) + centr_data[1:]
        else:
            final = solution

        print(solution)
        print('Solution: ')
        print('RA: '+ str(final['RA']))
        print('DEC: '+ str(final['Dec']))
        print('Roll: '+ str(final['Roll']))
        print('FOV: '+ str(final['FOV']))

        centroiding_img(centr_data)
        q = quarterneonCalc.radec_to_quarterneon(final['RA'], final['Dec'], final['Roll'])
        #outputpresentation.presentoutput(final['RA'], final['Dec'], final['Roll'],q)
        break
    