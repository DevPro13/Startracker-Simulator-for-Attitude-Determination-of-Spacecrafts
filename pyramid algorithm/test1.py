import numpy as np
import math

def calculate_angle(vector1, vector2):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1,vector2)

    # Calculate the magnitudes (lengths) of the vectors
    magnitude_v1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude_v2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the cosine of the angle between the vectors
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Calculate the angle in radians
    angle_radians = math.acos(cosine_angle)

    # Convert the angle from radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

# Example usage:
vector1 = [0, 1]
vector2 = [4, 0]

angle_between_vectors = calculate_angle(vector1, vector2)
print("Angle between the two vectors:", angle_between_vectors, "degrees")
