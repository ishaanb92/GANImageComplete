import cv2
import os
import numpy as np

def computeMatchesORB(img1,img2):
    orb = cv2.ORB_create()
    sim_metric = 0
    # Keypoints and descriptors for the objects
    kp_ref, des_ref = orb.detectAndCompute(img1,None)
    print(des_ref)
    kp_test, des_test = orb.detectAndCompute(img2,None)
    # Use Brute Force matcher + Hamming distance since binary features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des_ref,des_test)
    print(matches)
    # Need to draw only good matches, so create a mask
    goodMatches = []
    # ratio test as per Lowe's paper
    for i,m in enumerate(matches):
        # For threshold see: https://stackoverflow.com/questions/22857398/matching-orb-features-with-a-threshold
        if m.distance < 64:
            goodMatches.append(m)
            sim_metric = sim_metric + 1
    print(goodMatches)
    img3 = cv2.drawMatches(img1,kp_ref,img2,kp_test,goodMatches,outImg=None,flags=2)
    return sim_metric,img3

def returnScore():
    initial = cv2.imread('before.jpg')
    final = cv2.imread('final.jpg')
    sim,res = computeMatchesORB(initial,initial)
    print('Metric obtained from ORB : {0}'.format(sim))
    cv2.imwrite('result.png',res)

if __name__ == "__main__":
    returnScore()



