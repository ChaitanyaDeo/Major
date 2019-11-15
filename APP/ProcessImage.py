import cv2
import os
from DenoiseImage import Denoise


class Process:

    def __init__(self, imagePath):

        self.__imageName = imagePath.split("/")[-1]
        
        try:
            self.__toBeProcessed = cv2.imread(imagePath)
            #os.remove(imagePath)
            if self.__toBeProcessed is None:
                raise Exception() 
        except:
            print('UNSUCCESSFUL in reading image // PATH may be wrong') 
    
        self.__denoise = Denoise()

    def startProcessing(self):

        processedImage = self.__denoise.denoise(self.__toBeProcessed)

        pathToSave = os.getcwd() + '/static/images/cleaned'+self.__imageName
        cv2.imwrite(pathToSave, processedImage)
        print('DONE')
        return