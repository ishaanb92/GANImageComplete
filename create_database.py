import cv2
import numpy as np
import os
import argparse
import batch_preprocess
import shutil
import random

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
    parser.add_argument('-numGen',type=int,default=10)
    parser.add_argument('--numNoisy',type=int,default=5)
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
        sampleFiles = batch_preprocess.create_file_list(image_location,num_samples = args.numGen,sample = True)
        for imgFile in sampleFiles:
            shutil.copy2(imgFile,dumpDir)
        shutil.copy2(original_image,dumpDir)
        noisyImages = add_noise(original_image,numNoisy=3)
        for idx,nImg in zip(range(len(noisyImages)),noisyImages):
            cv2.imwrite(os.path.join(dumpDir,'noisy_{}.jpg'.format(idx)),nImg)


def add_noise(imagePath,numNoisy):
    trueImage = cv2.imread(imagePath)
    noisyImages = []
    for steps in range(numNoisy):
        noisyImages.append(add_random_noise(trueImage,prob = 0.02))
    return noisyImages


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



