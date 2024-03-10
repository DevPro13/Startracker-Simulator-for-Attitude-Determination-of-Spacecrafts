import math

def radec_to_quaternion(ra,dec,roll):
    #initializing the quarterneon
    q=[0,0,0,0]

    #convering degrees to radians
    ra = (ra * math.pi)/180
    dec = (dec * math.pi)/180
    roll = (roll * math.pi)/180

    #calculating the direction cosines l,m,n of roll angle 
    l = math.cos(dec)*math.cos(ra)
    m = math.cos(dec)*math.sin(ra)
    n = math.sin(dec)

    #half-angle of roll
    D = roll/2
    vert_comp = math.sin(D)
    horz_comp = math.cos(D)

    #quarterneon components calculation
    q[0]=l*vert_comp
    q[1]=m*vert_comp
    q[2]=n*vert_comp
    q[3]=horz_comp
    return q




