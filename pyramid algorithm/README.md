# Pyramid StarID Technique
## Algorithm
n = number of stars in the frame
- if n<3, a triangle can't be built so an error code is displayed
- if n=3, the pyramid can't be built, check for a unique solution of the triangle
  if the condition <br>
  <p align="center">
  $sign[b_i^T(b_j \times b_k)] = sign[r_i^T(r_j \times r_k)]$
  </p>    
  <br> is satisfied, the catalog triangles are not specular <br>
  NOTE: If more than one nonspecular cataloged triangle is found to match the measured triangle within the measurement tolerance, the star identification is not     accepted as unique (Display error code in this case too)
- if n>3, the algo looks for a unique triangle [i j k] by scanning the smart triad indices
  then a pyramid is identified having indices [i j k r]
  if the confirming star r is not found, another basic triangle[i j k] is selected by choosing another smart triad indices, and the r step is repeated again. **to be further added**

  The Flowchart for the Pyramid Star Identification is given as follows: [from **The Pyramid Star Identification Technique (Mortari)**]<p align="center">
![star_pyramid_flow](https://github.com/DevPro13/Startracker-Simulator-for-Attitude-Determination-of-Spacecrafts/assets/72692293/9faec76b-6604-4d0f-af96-fbf1af734627)
  </p>
## Interstar angle
It is essential to get the interstar angle in order to match the triad or pyramid structure to the star catalog. Let us calculate the angle part and collect resources for this part.<br>
From **Star Identification and Attitude Determination With Projective Cameras** by JOHN A. CHRISTIAN AND JOHN L. CRASSIDIS<p align="center">
![image](https://github.com/DevPro13/Startracker-Simulator-for-Attitude-Determination-of-Spacecrafts/assets/72692293/547198df-9eb3-4c68-a620-4b2dd78bacba)
![image](https://github.com/DevPro13/Startracker-Simulator-for-Attitude-Determination-of-Spacecrafts/assets/72692293/1ffdee3d-8f37-4cfb-823f-25c8e1b8b8f0)
</p>
<p align="center">
  
  ![image](https://github.com/DevPro13/Startracker-Simulator-for-Attitude-Determination-of-Spacecrafts/assets/72692293/5c07cfa3-86f1-4392-982b-63d2d11846ed)
</p>
This is the vector that interprets the direction of to star by a pair of angles (e.g., right ascension and declination) in the International Celestial Reference Frame (ICRF), where &delta;<sub>i</sub> is declination and &alpha;<sub>i</sub> is right ascension.
