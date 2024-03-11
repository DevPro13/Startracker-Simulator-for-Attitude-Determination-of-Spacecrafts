import os
from PIL import Image

from centroiding import get_centroids_from_image
from centroiding import overlay_spots
from centroiding import crop_and_downsample_image

# Paths for input and output folders
input_folder = "Centroiding_Media/real_images/"
potential_centroids_folder = "Centroiding_Media/potential_centroids/"
annotated_images_folder = "Centroiding_Media/annotated_images/"

# Iterate through files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # Process only PNG images
        image_path = os.path.join(input_folder, filename)

        try:
            # Load the image
            image = Image.open(image_path)

            # Perform centroid detection and image generation
            centr_data = get_centroids_from_image(image, min_area=7)  # Assuming this function is defined elsewhere
            centroids = centr_data[0] if isinstance(centr_data, tuple) else centr_data
            print('Found ' + str(len(centroids)) + ' centroids.')
            print(centroids)
            print("----------------------------------------------")
            labelled_regions = centr_data[1]['labelled_regions']
            final_centroids = centr_data[1]['final_centroids']

            # Create and save the first output image (potential centroids)
            potential_centroids_image = overlay_spots(image, labelled_regions, alpha=1)  # Assuming this function is defined elsewhere
            output_path = os.path.join(potential_centroids_folder, "pot_" + filename)
            potential_centroids_image.save(output_path)

            # Create and save the second output image (annotated image)
            annotated_image = overlay_spots(final_centroids, labelled_regions, alpha=2/3)
            output_path = os.path.join(annotated_images_folder, "ann_" + filename)
            annotated_image.save(output_path)

            print(f"Processed image {filename}, saved outputs to respective folders.")

        except Exception as e:
            print(f"Error processing image {filename}: {e}")
