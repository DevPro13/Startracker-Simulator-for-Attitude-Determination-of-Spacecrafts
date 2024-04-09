import numpy as np
import matplotlib.pyplot as plt
class SimulatorModel:
    def __init__(self,res_x,res_y,FOVx,FOVy,Mv,sigma,noiseLevel):
        self.res_x=res_x
        self.res_y=res_y
        self.FOVx=FOVx
        self.FOVy=FOVy
        self.Mv=Mv
        self.sigma=sigma
        self.noiseLevel=noiseLevel
    def getRotationalMat(self,alpha_0,delta_0,phi_0):
        """Gets the ascension, declination and role angle as inputs and return a rotational matrix"""
        # a1=(np.sin(alpha_0)*np.cos(phi_0))-(np.cos(alpha_0)*np.sin(delta_0)*np.sin(phi_0))
        # a2=(-np.sin(alpha_0)*np.sin(phi_0))-(np.cos(alpha_0)*np.sin(delta_0)*np.cos(phi_0))
        # a3=-np.cos(alpha_0)*np.cos(delta_0)
        # b1=(-np.cos(alpha_0)*np.cos(phi_0))-(np.sin(alpha_0)*np.sin(delta_0)*np.sin(phi_0))
        # b2=(np.cos(alpha_0)*np.sin(phi_0))-(np.sin(alpha_0)*np.sin(delta_0)*np.cos(phi_0))
        # b3=-np.sin(alpha_0)*np.cos(delta_0)
        # c1=np.cos(alpha_0)*np.sin(phi_0)
        # c2=np.cos(alpha_0)*np.cos(phi_0)
        # c3=-np.sin(delta_0)
        # M=np.array([
        #     [a1,a2,a3],
        #     [b1,b2,b3],
        #     [c1,c2,c3]
        # ],dtype="float64")
        ra_exp = alpha_0 - (np.pi/2)
        de_exp = delta_0 + (np.pi/2)
        M1 = np.array([[np.cos(ra_exp),-np.sin(ra_exp),0],[np.sin(ra_exp),np.cos(ra_exp),0],[0,0,1]])
        M2 = np.array([[1,0,0],[0,np.cos(de_exp),-np.sin(de_exp)],[0,np.sin(de_exp),np.cos(de_exp)]])
        M3 = np.array([[np.cos(phi_0),-np.sin(phi_0),0],[np.sin(phi_0),np.cos(phi_0),0],[0,0,1]])
        first_second = np.matmul(M1,M2)
        M = np.matmul(first_second,M3)
        return np.round(M.transpose(),decimals=5)
    def obtainStarSensorCoordinateSystem(self,M_T,alpha_i,delta_i):
        """
        M_T=Transpose of orthogonal matrix. ie. Rotational Matrix
        alpha_i=ith ascension
        delta_i=ith declination
        """
        direction_vector_of_stars_in_celestial_coord_sys=np.array([
            [np.cos(alpha_i)*np.cos(delta_i)],
            [np.sin(alpha_i)*np.cos(delta_i)],
            [np.sin(delta_i)]
        ],dtype="float64")
        return np.matmul(M_T,direction_vector_of_stars_in_celestial_coord_sys)
    def checkStarsAlphaDeltaWithInFOV(self,alpha_i,delta_i,alpha_0,delta_0):
        """
        Takes star positions, camera position and FOV as input and checks whether
        stars position is within camera FOV frame
        If returns True, stars position is within FOV
        """
        R=np.sqrt((self.FOVx**2)+(self.FOVy**2))/2
        R=np.deg2rad(R)
        rangeOfAlpha=np.array([alpha_0-(R/np.cos(delta_0)),alpha_0+(R/np.cos(delta_0))])
        rangeOfDelta=np.array([delta_0-R,delta_0+R])
        return np.logical_and(rangeOfAlpha[0]<alpha_i<rangeOfAlpha[1],rangeOfDelta[0]<delta_i<rangeOfDelta[1])
    def getPixelXY(self,sensor_coord_mat):
        fx=self.res_x/(2*np.tan(self.FOVx/2))
        fy=self.res_y/(2*np.tan(self.FOVy/2))
        proj_mat=np.array([
            [fx,0,self.res_x/2],
            [0,fy,self.res_y/2],
            [0,0,1]
        ])
        return np.matmul(proj_mat,sensor_coord_mat)+np.array([
            [self.res_x/2],
            [self.res_y/2],
            [0]
        ])
    def getPixelXYAxis(self,sensor_coord_mat):
        myu=1.12e-6
        fx=self.res_x/(2*np.tan(np.deg2rad(self.FOVx/2)))*myu
        fy=self.res_y/(2*np.tan(np.deg2rad(self.FOVy/2)))*myu
        x=fx*(sensor_coord_mat[0]/sensor_coord_mat[2])
        y=fy*(sensor_coord_mat[1]/sensor_coord_mat[2])
        x,y=float(x),float(y)
        x1Pix=round(x/myu)
        y1Pix=round(y/myu)
        cx=self.res_x/2
        cy=self.res_y/2
        return (x1Pix+cx,cy-y1Pix)