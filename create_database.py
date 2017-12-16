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
    parser.add_argument('--outDir',type=str,default = 'database')
    parser.add_argument('--numNoisy',type=int,default=2)
    args = parser.parse_args()

    if (os.path.exists(args.outDir)):
        # Remove it
        shutil.rmtree(args.outDir)

    os.makedirs(args.outDir)
    image_files_gan = 'recons_mturk'
    image_files_ce = '/home/ibhat/context_enc/Context-Encoder/src/test_dump'
    image_files_ae = '/home/ibhat/context_enc/Context-Encoder/src_vanilla/test_dump'

    image_dirs_gan = [os.path.join(image_files_gan,d) for d in os.listdir(image_files_gan) if os.path.isdir(os.path.join(image_files_gan,d))]
    image_dirs_gan = sorted(image_dirs_gan)

    image_dirs_ce = [os.path.join(image_files_ce,d) for d in os.listdir(image_files_ce) if os.path.isdir(os.path.join(image_files_ce,d))]
    image_dirs_ce = sorted(image_dirs_ce)


    image_dirs_ae = [os.path.join(image_files_ae,d) for d in os.listdir(image_files_ae) if os.path.isdir(os.path.join(image_files_ae))]
    image_dirs_ae = sorted(image_dirs_ae)

    for idx,gan_dir,ce_dir,ae_dir in zip(range(len(image_dirs_gan)),image_dirs_gan,image_dirs_ce,image_dirs_ae):
        dumpDir = os.path.join(args.outDir,'{}'.format(idx))
        os.makedirs(dumpDir)
        # Make the sub-folder
        genDir = os.path.join(dumpDir,'gen')
        os.makedirs(genDir)
        original_image = os.path.join(gan_dir,'original.jpg')
        # Copy over the original image
        shutil.copy2(original_image,dumpDir)

        # Get paths from each model dump
        gan_image = os.path.join(gan_dir,'gen_images','completed_3950.jpg')
        ce_image = os.path.join(ce_dir,'recon.jpg')
        ae_image = os.path.join(ae_dir,'recon.jpg')

        # Create destination paths
        gan_dump = os.path.join(genDir,'gan.jpg')
        ce_dump = os.path.join(genDir,'ce.jpg')
        ae_dump = os.path.join(genDir,'ae.jpg')

        #Copy
        shutil.copy(gan_image,gan_dump)
        shutil.copy(ce_image,ce_dump)
        shutil.copy(ae_image,ae_dump)

        noisyImages = add_noise(original_image,noiseType = "blur", numNoisy=1)
        for idx,nImg in zip(range(len(noisyImages)),noisyImages):
            cv2.imwrite(os.path.join(genDir,'noisy_{}.jpg'.format(idx)),nImg)


def add_noise(imagePath,noiseType = "blur",numNoisy = 3):
    trueImage = cv2.imread(imagePath)
    noisyImages = []
    if noiseType == "random":
        for steps in range(numNoisy):
            noisyImages.append(add_random_noise(trueImage,prob = 0.02))
    elif noiseType == "blur":
        noisyImages.append(add_blur(trueImage,5))
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



