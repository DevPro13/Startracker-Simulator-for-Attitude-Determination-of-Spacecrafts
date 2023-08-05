# Pyramid StarID Technique
## Algorithm
n = number of stars in the frame
- if n<3, a triangle can't be built so an error code is displayed
- if n=3, the pyramid can't be built, check for a unique solution of the triangle
  if the condition 
      b_i^T(b_j * b_k) = r_I^T(r_j * r_K)
  is satisfied, the catalog triangles are not specular
- if n>3, the algo looks for a unique triangle [i j k] by scanning the smart triad indices
  then a pyramid is identified having indices [i j k r]
  if the confirming star r is not found, another basic triangle[i j k] is selected by choosing another smart triad indices, and the r step is repeated again.
