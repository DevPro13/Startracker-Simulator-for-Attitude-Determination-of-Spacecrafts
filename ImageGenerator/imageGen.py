from SimulatorModel import SimulatorModel
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
class GenerateStarImage:
    def __init__(self,description):
        self.RA=float(description["RA"])
        self.DEC=float(description["DEC"])
        self.ROLL=float(description["ROLL"])
        self.resX=int(description["RES_X"])
        self.resY=int(description["RES_Y"])
        self.FOVx=int(description["FOVx"])
        self.FOVy=int(description["FOVy"])
        self.sigma=float(description["sigma"])
        self.Mv=float(description["Mv"])
        self.Noise=float(description["noise"])
    def generateCatalog(self):
        data=np.load("../Custom Catalog/data.npz")
        df=pd.DataFrame(data["star_table"],columns=["RA","DEC","COSCOS","SINCOS","SIN","MV"])
        df.drop(columns=["COSCOS","SINCOS","SIN"],inplace=True)
        star_coords_in_array=df.to_numpy()
        return star_coords_in_array
    def generateImage(self,star_coords_in_array_from_catalog):
        simulator=SimulatorModel(self.resX,self.resY,self.FOVx,self.FOVy,self.Mv,self.sigma,self.Noise)
        #make ort mat
        M_T=simulator.getRotationalMat(np.deg2rad(self.RA),np.deg2rad(self.DEC),np.deg2rad(self.ROLL))
        sensor_coords_in_array=[]
        Mvs=[]
        for i in star_coords_in_array_from_catalog:
            if simulator.checkStarsAlphaDeltaWithInFOV(i[0],i[1],np.deg2rad(self.RA),np.deg2rad(self.DEC)):
                Mvs.append(i[2])
                sensor_coords_in_array.append(simulator.obtainStarSensorCoordinateSystem(M_T,i[0],i[1]))
        pixels=[]
        for i in sensor_coords_in_array:
            pixels.append(simulator.getPixelXYAxis(i)[:2])
        x = np.linspace(0,self.resX,self.resY)
        y = np.linspace(0,self.resX,self.resY)
        X, Y = np.meshgrid(x,y)
        PSF=np.zeros_like(Y)
        for i in range(len(pixels)):
            PSF +=self.psfGaussianDistribution(X-pixels[i][0],Y-pixels[i][1],Mvs[i])#gaussian_psf(X - X_points[i], Y - Y_points[i])
        filename = f"../Media/ra{self.RA}_de{self.DEC}_roll{self.ROLL}_FOV{self.FOVx}_Res_{self.resX}x{self.resY}.png"
        PSFnoisy=self.Add_Noise(PSF)
        self.SaveImage(PSFnoisy,filename)
        return filename
    def SaveImage(self,PSF,filename):
        # Normalize PSF for visualization
        PSF_normalized = (PSF - PSF.min()) / (PSF.max() - PSF.min()) * 255
        # Convert PSF array to uint8 for OpenCV
        PSF_uint8 = PSF_normalized.astype(np.uint8)
        #save the image
        cv2.imwrite(filename,PSF_uint8)
    def psfGaussianDistribution(self,x,y,Mv):
        k1=1000
        k2=k3=1
        H=k1*np.exp(-k2*Mv+k3)
        num=np.exp(-((x**2)+(y**2))/(2*(self.sigma**2)),dtype="float128")
        den=2*3.14*(self.sigma**2)
        return H*num/den
    def Add_Noise(self,image):
        noise=np.random.normal(0,self.Noise,size=(self.resX,self.resY))
        return image+noise
    def description(self):
        return f"""
        <pre>
        RA={self.RA}   DEC={self.DEC}     ROLL={self.ROLL}<br>    
        Resolution={self.resX}*{self.resY} pixels         FOV={self.FOVx}*{self.FOVy} degrees<br>
        <br><br>
        Sigma={self.sigma}<br>
        Noise Level={self.Noise}
        </pre>
        """