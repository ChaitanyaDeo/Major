import pandas as pd
import numpy as np 
import time 
from CBDNetProcess import processUsingCBD

class Denoise:
    
    def CBDdenoise(self, imagePath, choice):

        denoised = processUsingCBD(imagePath, choice)
        
        return denoised

    def UNetdenoise(self, imagePath):
        pass