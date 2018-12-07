import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.decomposition import PCA
import cv2
import pickle as pkl
from tqdm import tqdm

def drawLips(keypoints, new_img, c = (255, 255, 255), th = 1, show = True):

	keypoints = np.float32(keypoints)

	for i in range(48, 59):
		cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
	for i in range(60, 67):
		cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)

	if (show == True):
		cv2.imshow('Test', new_img)
		cv2.waitKey(100)

def getOriginalKeypoints(kp_features_mouth, N, tilt, mean):
	# Denormalize the points
	kp_dn = N * kp_features_mouth
	# Add the tilt
	x, y = kp_dn[:, 0], kp_dn[:, 1]
	c, s = np.cos(tilt), np.sin(tilt)
	x_dash, y_dash = x * c + y * s, - x * s + y * c
	kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1,1))))
	# Shift to the mean
	kp = kp_tilt + mean
	return kp

print "Starting"
with open('Kp_25.pickle', 'rb') as pklFile:
	kp = pkl.load(pklFile)
print "Loaded"

for l in range(0, 10):
	for idx, k in enumerate(kp[l]):
		#print x
		unit_mouth_kp, N, tilt, mean, unit_kp, keypoints = k[0], k[1], k[2], k[3], k[4], k[5]
		kps = getOriginalKeypoints(unit_mouth_kp, N, tilt, mean)
		keypoints[48:68] = kps

		imgfile = 'Data/' + str(l + 1).rjust(3, '0') + '/' + str(idx+1).rjust(4, '0') + '.jpg'
		im = cv2.imread(imgfile)
		drawLips(keypoints, im, c = (255, 255, 255), th = 1, show = True)

	# make it pix2pix style
	# print('Shape: ', im1.shape)
