from math import radians,degrees,sin,cos,tan,sqrt,atan,pi,exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
def create_M_matrix(alpha_0,delta_0,phi_0):
    """[summary]

    Args:
        ra ([int]): [right ascension of sensor center]
        de ([int]): [declination of sensor center]
        roll ([int]): [roll angle of star sensor]
        method ([int]): [1 for method 1(Calculating each elements),2 for method 2(calculating rotation matrices)]
    """
    # if method == 1:
    #     a1 = (sin(ra)*cos(roll)) - (cos(ra)*sin(de)*sin(roll))
    #     a2 = -(sin(ra)*sin(roll)) - (cos(ra)*sin(de)*cos(roll))
    #     a3 = -(cos(ra)*cos(de))
    #     b1 = -(cos(ra)*cos(roll)) - (sin(ra)*sin(de)*sin(roll))
    #     b2 = (cos(ra)*sin(roll)) - (sin(ra)*sin(de)*cos(roll))
    #     b3 = -(sin(ra)*cos(de))
    #     c1 = (cos(ra)*sin(roll))
    #     c2 = (cos(ra)*cos(roll))
    #     c3 = -(sin(de))
    #     M = np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
    # if method == 2:
    ra_exp = alpha_0 - (np.pi/2)
    de_exp = delta_0 + (np.pi/2)
    M1 = np.array([[np.cos(ra_exp),-np.sin(ra_exp),0],[np.sin(ra_exp),np.cos(ra_exp),0],[0,0,1]])
    M2 = np.array([[1,0,0],[0,np.cos(de_exp),-np.sin(de_exp)],[0,np.sin(de_exp),np.cos(de_exp)]])
    M3 = np.array([[np.cos(phi_0),-np.sin(phi_0),0],[np.sin(phi_0),np.cos(phi_0),0],[0,0,1]])
    first_second = np.matmul(M1,M2)
    M = np.matmul(first_second,M3)
    return np.round(M.transpose(),decimals=5)


def dir_vector_to_star_sensor(ra,de,M_transpose):
    """[Converts direction vector to star sensor coordinates]

    Args:
        ra ([int]): [right ascension of the object vector]
        de ([int]): [desclination of the object vector]
        M_transpose ([numpy array]): [rotation matrix from direction vector to star sensor transposed]
    """
    print(ra,de,roll)    
    x_dir_vector = (cos(ra)*cos(de))
    y_dir_vector = (sin(ra)*cos(de))
    z_dir_vector = (sin(de))
    dir_vector_matrix = np.array([[x_dir_vector],[y_dir_vector],[z_dir_vector]])
    return M_transpose.dot(dir_vector_matrix)


def draw_star(x,y,magnitude,gaussian,background,ROI=5):
    """[Draws the star in the background image]

    Args:
        x ([int]): [The x coordinate in the image coordinate system (starting from left to right)]
        y ([int]): [The y coordinate in the image coordinate system (starting from top to bottom)]
        magnitude ([float]): [The stellar magnitude]
        gaussian ([bool]): [True if using the gaussian function, false if using own function]
        background ([numpy array]): [background image]
        ROI ([int]): [The ROI of each star in pixel radius]
    """
    if gaussian:
        H = 2000*exp(-magnitude+1)
        sigma = 5
        for u in range(x-ROI,x+ROI+1):
            for v in range(y-ROI,y+ROI+1):
                dist = ((u-x)**2)+((v-y)**2)
                diff = (dist)/(2*(sigma**2))
                exponent_exp = 1/(exp(diff))
                raw_intensity = int(round((H/(2*pi*(sigma**2)))*exponent_exp))
                if u == x and v == y:
                    print(raw_intensity)
                background[v,u] = raw_intensity
    else:
        mag = abs(magnitude-7) #1 until 9
        radius = int(round((mag/9)*(5)+2))
        color = int(round((mag/9)*(155)+100))
        cv2.circle(background,(x,y),radius,color,thickness=-1)
    return background

def add_noise(low,high,background):
    """[Adds noise to an image]

    Args:
        low ([int]): [lower threshold of the noise generated]
        high ([int]): [maximum pixel value of the noise generated]
        background ([numpy array]): [the image that is put noise on]
    """
    row,col = np.shape(background)
    background = background.astype(int)
    noise = np.random.randint(low,high=high,size=(row,col))
    noised_img = cv2.addWeighted(noise,0.1,background,0.9,0)
    return noised_img

#Right ascension, declination and roll input prompt from user
ra0 = 12#30#249.2104#input("Enter the right ascension angle in degrees:\n")
de0 = 58#70#-12.0386#input("Enter the declination angle in degrees:\n")
roll0 =90#30# 13.3845#input("Enter the roll angle in degrees:\n")

ra = radians(float(ra0))
de = radians(float(de0))
roll = radians(float(roll0))

#length/pixel
myu = 1.12*(10**-6)

#Focal length prompt from user
#f =0.00584563#0.00304

#Star sensor pixel
l = 1024#3280
w = 1024#2464
print("Resolution length: {}".format(l))
print("Resolution width: {}".format(w))

#Star sensor FOV
# FOVy = degrees(2*atan((myu*w/2)/f))
# FOVx = degrees(2*atan((myu*l/2)/f))
FOVx=FOVy=20
f=myu*l/(2*tan(np.deg2rad(FOVx/2)))
print("FOV y: {}".format(FOVy))
print("FOV x: {}".format(FOVx))

#STEP 1: CONVERSION OF CELESTIAL COORDINATE SYSTEM TO STAR SENSOR COORDINATE SYSTEM
M = create_M_matrix(ra,de,roll)
print("*"*80)
print(f"Matrix M:\n {M}")

#Check if matrix is orthogonal
M_inverse = np.round(np.linalg.inv(M),decimals=5)
M_transpose = np.round(np.matrix.transpose(M),decimals=5)
print(f"Transpose: {M_transpose}")
orthogonal_check = []
for row in range(3):
    for column in range(3):
        element_check = M_inverse[row,column] == M_transpose[row,column]
        orthogonal_check.append(element_check)

if all(orthogonal_check):
    print("Matrix M is orthogonal...\nMoving on to next calculation\n")
else:
    print("WARNING: Matrix M is not orthogonal")

#Search for image-able stars
print("Reading in CSV file...\n")
col_list = ["Star ID","RA","DE","Magnitude"]
star_catalogue = pd.read_csv('./Below_6.0_SAO.csv',usecols=col_list)
R = (sqrt((radians(FOVx)**2)+(radians(FOVy)**2))/2)
alpha_start = (ra - (R/cos(de)))
alpha_end = (ra + (R/cos(de)))
delta_start = (de - R)
delta_end = (de + R)
star_within_ra_range = (alpha_start <= star_catalogue['RA']) & (star_catalogue['RA'] <= alpha_end)
star_within_de_range = (delta_start <= star_catalogue['DE']) & (star_catalogue['DE'] <= delta_end)
star_in_ra = star_catalogue[star_within_ra_range]
star_in_de = star_catalogue[star_within_de_range]
star_in_de = star_in_de[['Star ID']].copy()
stars_within_FOV = pd.merge(star_in_ra,star_in_de,on="Star ID")

#Converting to star sensor coordinate system
ra_i = list(stars_within_FOV['RA'])
de_i = list(stars_within_FOV['DE'])
star_sensor_coordinates = []
for i in range(len(ra_i)):
    coordinates = dir_vector_to_star_sensor(ra_i[i],de_i[i],M_transpose=M_transpose)
    star_sensor_coordinates.append(coordinates)

#STEP 2: CONVERSION OF STAR SENSOR COORDINATE SYSTEM TO IMAGE COORDINATE SYSTEM
star_loc = []
for coord in star_sensor_coordinates:
    x = f*(coord[0]/coord[2])
    y = f*(coord[1]/coord[2])
    star_loc.append((x,y))

pixel_per_length = 1/myu

magnitude_mv = list(stars_within_FOV['Magnitude'])
print(magnitude_mv)
filtered_magnitude = []

#Rescaling to pixel sizes
pixel_coordinates = []
delete_indices = []
for i,(x1,y1) in enumerate(star_loc):
    x1 = float(x1)
    y1 = float(y1)
    x1pixel = round(pixel_per_length*x1)
    y1pixel = round(pixel_per_length*y1)
    print([x1pixel,y1pixel])
    if abs(x1pixel) > l/2 or abs(y1pixel) > w/2:
        delete_indices.append(i)
        continue
    pixel_coordinates.append((x1pixel,y1pixel))
    filtered_magnitude.append(magnitude_mv[i])
print(filtered_magnitude)
background = np.zeros((w,l))
for i in range(len(filtered_magnitude)):
    x = round(l/2 + pixel_coordinates[i][0])
    y = round(w/2 - pixel_coordinates[i][1])
    print(f"The coordinates are({x},{y}.")
    background = draw_star(x,y,filtered_magnitude[i],False,background)

#Adding noise
background = add_noise(0,50,background=background)
file_name = f"../Media/Github_Validation_ra{ra0}_de{de0}_roll{roll0}.jpg"
cv2.imwrite(file_name,background)
#cv2.imshow("Image",background)