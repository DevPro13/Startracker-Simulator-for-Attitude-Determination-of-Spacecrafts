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

def centroiding_img(img, pointz):
    image = cv2.imread(str(img))
    # The kernel to be used for dilation purpose 
    kernel = np.ones((5, 5), np.uint8) 
    
    # converting the image to HSV format 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    
    # defining the lower and upper values of HSV, 
    # this will detect yellow colour 
    Lower_hsv = np.array([20, 70, 100]) 
    Upper_hsv = np.array([30, 255, 255]) 
    
    # creating the mask by eroding,morphing, 
    # dilating process 
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv) 
    Mask = cv2.erode(Mask, kernel, iterations=1) 
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel) 
    Mask = cv2.dilate(Mask, kernel, iterations=1) 
    
    # Inverting the mask by 
    # performing bitwise-not operation 
    Mask = cv2.bitwise_not(Mask) 
    
    # Displaying the image 
    plt.imshow(Mask)
    plt.axis('off')
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.tight_layout()
    plt.savefig("testinv.png")
    plt.close()

    fig, ax = plt.subplots(facecolor='whitesmoke')
    plt.imshow(image)
    plt.scatter(pointz[:, 1], pointz[:, 0], c="green", marker="o", s=5, alpha=0.3)
    plt.axis('off')
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.tight_layout()
    plt.savefig("test.png")
    plt.show()
    plt.close()

for impath in path.glob('*'):
    print('Solving for image at: ' + str(impath))
    with Image.open(str(impath)) as img:
        assert demo.has_database, 'No database loaded'
        print('Got solve from image with input: ' + str((img, {**optional_features})))
        (width, height) = img.size[:2]
        print('Image (height, width): ' + str((height, width)))
        
        # Run star extraction, passing optional_features along
        t0_extract = precision_timestamp()
        centr_data = starTrack.get_centroids_from_image(img, **optional_features)
        t_extract = (precision_timestamp() - t0_extract)*1000
       
        # If we get a tuple, need to use only first element and then reassemble at return
        if isinstance(centr_data, tuple):
            centroids = centr_data[0]
        else:
            centroids = centr_data
        print('Found this many centroids, in time: ' + str((len(centroids), t_extract)))
        
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
        print(centr_data)
        print(solution)
        print('Solution: ')
        print('RA: '+ str(final['RA']))
        print('DEC: '+ str(final['Dec']))
        print('Roll: '+ str(final['Roll']))
        print('FOV: '+ str(final['FOV']))

        centroiding_img(impath, centr_data)
        q = quarterneonCalc.radec_to_quaternion(final['RA'], final['Dec'], final['Roll'])
        print(q)
        outputpresentation.presentoutput(final['RA'], final['Dec'], final['Roll'],q)
        break
    