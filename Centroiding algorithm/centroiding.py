import cv2
import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, mu, sigma):
  """
  Defines the Gaussian function for fitting.
  """
  return np.exp(-(x - mu)**2 / (2 * sigma**2))

def centroid_gaussian_fit(image):
  """
  Calculates the centroid of the brightest object in the image using Gaussian fitting.

  Args:
      image: A numpy array representing the grayscale image.

  Returns:
      A tuple (x, y) representing the centroid coordinates, or None if no bright object is found.
  """
  # Convert image to grayscale if it's not already grayscale
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply thresholding to isolate the brightest object
  threshold, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

  # Find contours of the object
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Check if any contours were found
  if not contours:
    return None

  # Find the largest contour
  largest_contour = max(contours, key=cv2.contourArea)

  # Get moments of the largest contour
  moments = cv2.moments(largest_contour)

  # Check if moments are valid (avoid division by zero)
  if moments['m00'] == 0:
    return None

  # Calculate initial guess for centroid based on moments
  centroid_x = int(moments['m10'] / moments['m00'])
  centroid_y = int(moments['m01'] / moments['m00'])

  # Define a window around the initial guess for fitting
  window_size = 20  # Adjust window size as needed
  x_min = max(0, centroid_x - window_size // 2)
  x_max = min(image.shape[1], centroid_x + window_size // 2)
  y_min = max(0, centroid_y - window_size // 2)
  y_max = min(image.shape[0], centroid_y + window_size // 2)

  # Extract image data within the window
  windowed_image = image[y_min:y_max, x_min:x_max]

  # Flatten the extracted image data for fitting
  data = windowed_image.flatten()

  # Define x-coordinates for the flattened data (assuming uniform spacing)
  x = np.arange(len(data)) + x_min

  # Try-except block to handle potential fitting errors
  try:
    # Perform the Gaussian fit using curve_fit from scipy.optimize
    popt, _ = curve_fit(gaussian, x, data, p0=[centroid_x - x_min, 5])  # Adjust initial guess for sigma
  except RuntimeError as e:
    print("Gaussian fitting failed:", e)
    return None

  # Extract fitted centroid from the results
  fitted_centroid_x = popt[0]
  fitted_centroid_y = centroid_y  # Assuming negligible y-axis shift due to windowing

  return (int(fitted_centroid_x), int(fitted_centroid_y))

# Example usage
image = cv2.imread("../plot.png")
centroid_coordinates = centroid_gaussian_fit(image)

if centroid_coordinates:
  print("Centroid:", centroid_coordinates)
  # Draw a circle at the centroid location (optional)
  cv2.circle(image, centroid_coordinates, 5, (0, 0, 255), -1)
  cv2.imshow("Image with Centroid", image)
  cv2.waitKey(0)
else:
  print("No bright object found in the image")
