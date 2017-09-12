import cv2
import os
import glob
import numpy as np
import collections
import matplotlib.pylab as plt
import argparse
import shutil

# Author : Ishaan Bhat
# Email : i.r.bhat@student.tue.nl


def computeMatchesORB(img1,img2):
    orb = cv2.ORB_create()
    sim_metric = 0
    # Keypoints and descriptors for the objects
    kp_ref, des_ref = orb.detectAndCompute(img1,None)
    kp_test, des_test = orb.detectAndCompute(img2,None)
    # Use Brute Force matcher + Hamming distance since binary features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des_ref,des_test)
    # Need to draw only good matches, so create a mask
    goodMatches = []
    # ratio test as per Lowe's paper
    for i,m in enumerate(matches):
        # For threshold see: https://stackoverflow.com/questions/22857398/matching-orb-features-with-a-threshold
        if m.distance < 64:
            goodMatches.append(m)
            sim_metric = sim_metric + 1
    img3 = cv2.drawMatches(img1,kp_ref,img2,kp_test,goodMatches,outImg=None,flags=2)
    return sim_metric

def computeMatchesSIFT(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
    sim_metric = 0
    # Keypoints and descriptors for the objects
    kp_ref, des_ref = sift.detectAndCompute(img1,None)
    kp_test, des_test = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des_ref,des_test,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            sim_metric = sim_metric + 1
    return sim_metric

# Functions to compute metrics to be added here ...

def returnScore():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",type=str,default="completions",help="Folder name for images")
    parser.add_argument("--num_images",type=int,default=10,help="Number of images to be analyzed")
    args = parser.parse_args()

    # Create a dir to store results
    if not os.path.exists(os.path.join(os.getcwd(),args.images,'results')):
        os.makedirs(os.path.join(os.getcwd(),args.images,'results'))
    else:
        shutil.rmtree(os.path.join(os.getcwd(),args.images,'results'))
        os.makedirs(os.path.join(os.getcwd(),args.images,'results'))

    resultPath = os.path.join(os.getcwd(),args.images,'results')

    # For each image completed
    for img in range(0,args.num_images):
        imagePath = os.path.join(os.getcwd(),args.images,'{:04d}'.format(img),'completed','*.jpg')
        refFile = open(os.path.join(os.getcwd(),args.images,'before_{:04d}.jpg'.format(img)))
        refImg = cv2.imread(refFile.name)
        refMatches = computeMatchesSIFT(refImg,refImg)
        images = glob.glob(imagePath)
        metricDict = {}
        print('Collecting metrics for {} image'.format(img))
        # For each checkpointed iteration of completion (per-image)
        for image in images:
            testImg = cv2.imread(image)
            # Computing the similarity metric
            simSIFT = computeMatchesSIFT(refImg,testImg)
            simScore = float(simSIFT)/float(refMatches)
            pathSet = image.split(os.sep)
            idx = pathSet[-1].split(".")
            # Creating a dictionary with key:value as  IterationID:SimilarityScore
            metricDict[int(idx[0])] = simScore

            # Add code to compute alternative metrics here (PSNR, MSE etc ..)
            # Store these in a dictionary similar to as done above
            # Order it once metrics for all intermediate images are calculated (as shown below)

        # Sort the dict based on key which is the iteration ID
        orderedMetrics = sorted(metricDict.items())
        x,y = zip(*orderedMetrics)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot the ordered dictionary
        ax.plot(x,y)
        plt.ylabel('Similarity Score (SIFT)')
        plt.xlabel('Iteration')

        # Save the graph as an image
        figPath = os.path.join(resultPath,'res_{:04d}.png'.format(img))
        plt.savefig(figPath)

if __name__ == "__main__":
    returnScore()



