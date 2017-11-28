import cv2
import numpy as np
import os
import argparse
import shutil
import random
import sys

def create_database():

    """
    Reads the files created by the reconstruction API to create a flat hierarchy
    Each folder generated contains one original image + configured number of random samples
    from the generated set + configured number of images where random noise has been applied to
    the original image.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir',type=str,default ='recon')
    parser.add_argument('--outDir',type=str,default = 'database')
    parser.add_argument('--numNoisy',type=int,default=2)
    args = parser.parse_args()

    if (os.path.exists(args.outDir)):
        # Remove it
        shutil.rmtree(args.outDir)

    os.makedirs(args.outDir)

    image_dirs = [os.path.join(args.inputDir,d) for d in os.listdir(args.inputDir) if os.path.isdir(os.path.join(args.inputDir,d))]

    for idx,imgDir in zip(range(len(image_dirs)),image_dirs):
        dumpDir = os.path.join(args.outDir,'{}'.format(idx))
        os.makedirs(dumpDir)
        original_image = os.path.join(imgDir,'original.jpg')
        image_location = os.path.join(imgDir,'gen_images')
        # Select last 3 iterations of generated images
        sampleFiles = []
        for step in range(3):
            sampleFiles.append(os.path.join(image_location,'gen_{}.jpg'.format(1850 + step*50)))
        for imgFile in sampleFiles:
            shutil.copy2(imgFile,dumpDir)
        shutil.copy2(original_image,dumpDir)
        noisyImages = add_noise(original_image,noiseType = "blur", numNoisy=3)
        for idx,nImg in zip(range(len(noisyImages)),noisyImages):
            cv2.imwrite(os.path.join(dumpDir,'noisy_{}.jpg'.format(idx)),nImg)


def add_noise(imagePath,noiseType = "blur",numNoisy = 3):
    trueImage = cv2.imread(imagePath)
    noisyImages = []
    if noiseType == "random":
        for steps in range(numNoisy):
            noisyImages.append(add_random_noise(trueImage,prob = 0.02))
    elif noiseType == "blur":
        noisyImages.append(add_blur(trueImage,5))
        noisyImages.append(add_blur(trueImage,9))
    else:
        print('Noise type not supported')
        sys.exit()
    return noisyImages

def add_blur(image,kernelSize):
    return cv2.GaussianBlur(image,(kernelSize,kernelSize),0)

def add_random_noise(image,prob):
    noisyImage = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for k in range(image.shape[2]): # Color Channel
        for i in range(image.shape[0]): # Width
            for j in range(image.shape[1]): # Height
                rdn = random.random()
                if rdn < prob:
                    noisyImage[i][j][k] = 0
                elif rdn > thres:
                    noisyImage[i][j][k] = 255
                else:
                    noisyImage[i][j][k] = image[i][j][k]
    return noisyImage



if __name__ == '__main__':
    create_database()



