import cv2
import os
import glob
import numpy as np
import collections
import matplotlib.pylab as plt

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

def returnScore():
    imagePath = os.path.join(os.getcwd(),'completions','completed','*.jpg')
    refFile = open(os.path.join(os.getcwd(),'completions','before.jpg'))
    refImg = cv2.imread(refFile.name)
    refMatches = computeMatchesSIFT(refImg,refImg)
    images = glob.glob(imagePath)
    metricDict = {}
    for image in images:
        testImg = cv2.imread(image)
        simSIFT = computeMatchesSIFT(refImg,testImg)
        simScore = float(simSIFT)/float(refMatches)
        pathSet = image.split(os.sep)
        idx = pathSet[-1].split(".")
        metricDict[int(idx[0])] = simScore
    # Sort the dict based on key which is the iteration ID
    orderedMetrics = sorted(metricDict.items())
    x,y = zip(*orderedMetrics)
    plt.plot(x,y)
    plt.ylabel('Similarity Score (SIFT)')
    plt.xlabel('Iteration')
    plt.show()

if __name__ == "__main__":
    returnScore()



