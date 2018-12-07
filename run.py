from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Lambda, TimeDistributed
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import pickle as pkl
from keras.callbacks import TensorBoard
from time import time
import cv2
import scipy.io.wavfile as wav
from python_speech_features import logfbank, mfcc
import subprocess
import argparse

#########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--sf", help="path to wav file")
parser.add_argument("--mf", help="path to model")
parser.add_argument("--lb", help="look back")
a = parser.parse_args()

keyAudio = a.sf 
modelFile = a.mf
timeDelay = 20
lookBack = int(a.lb)
nEpoch = 25
outputFolder = 'output/'

#########################################################################################

cmd = 'rm -rf '+ outputFolder + '&& mkdir ' + outputFolder
subprocess.call(cmd ,shell=True)

#########################################################################################

model = load_model(modelFile)

#########################################################################################

def subsample(y, fps_from = 100.0, fps_to = 30):
	factor = int(np.ceil(fps_from / fps_to))
	# Subsample the points
	new_y = np.zeros((int(y.shape[0] / factor), 20, 2)) #(timesteps, 20) = (500, 20x2)
	for idx in range(new_y.shape[0]):
		if not (idx * factor > y.shape[0] - 1):
			new_y[idx, :, 0] = y[idx * factor, 0:20]
			new_y[idx, :, 1] = y[idx * factor, 20:]
		else:
			break
	new_y = [np.array(each) for each in new_y.tolist()]
	return new_y

def drawLips(keypoints, new_img, c = (255, 255, 255), th = 1, show = False):

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
		cv2.imshow('lol', new_img)
		cv2.waitKey(10000)

def getOriginalKeypoints(kp_features_mouth, N, tilt, mean):
	kp_dn = N * kp_features_mouth * 1.5
	x, y = kp_dn[:, 0], kp_dn[:, 1]
	c, s = np.cos(tilt), np.sin(tilt)
	x_dash, y_dash = x*c + y*s, -x*s + y*c
	kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1, 1))))
	kp = kp_tilt + mean
	return kp


def join_features(mfcc, fbank):
    features = np.concatenate((mfcc, fbank), axis=1)
    return features

#########################################################################################

with open('PCA_reducedKp.pickle', 'rb') as pkl_file:
	video_kp = pkl.load(pkl_file)
with open('TestVideo/kp_test.pickle', 'rb') as pkl_file:
	kp = pkl.load(pkl_file)
with open('PCA.pickle', 'rb') as pkl_file:
	pca = pkl.load(pkl_file)

# Get the data
X, y = [], [] # Create the empty lists

print keyAudio

(rate, sig) = wav.read(keyAudio)
print rate
mfccFeat = mfcc(sig, rate)
fbankFeat = logfbank(sig, rate)
audio = join_features(mfccFeat, fbankFeat) 

video = video_kp[0]


start = (timeDelay - lookBack) if (timeDelay - lookBack > 0) else 0
for i in range(start, len(audio) - lookBack):
	a = np.array(audio[i:i + lookBack])
	X.append(a)

for i in range(start, len(video) - lookBack):
	v = np.array(video[i + lookBack - timeDelay]).reshape((1, -1))
	y.append(v)

X = np.array(X)
y = np.array(y)
shapeX = X.shape
shapey = y.shape
print('Shapes:', X.shape)
X = X.reshape(-1, X.shape[2])
y = y.reshape(-1, y.shape[2])
print('Shapes:', X.shape)

scalerX = MinMaxScaler(feature_range=(0, 1))
scalery = MinMaxScaler(feature_range=(0, 1))

X = scalerX.fit_transform(X)
y = scalery.fit_transform(y)


X = X.reshape(shapeX)

y_pred = model.predict(X)

# Scale it up
y_pred = scalery.inverse_transform(y_pred)
y_pred = pca.inverse_transform(y_pred)

print('Upsampled number:', len(y_pred))

y_pred = subsample(y_pred, 100, 30)

print('Subsampled number:', len(y_pred))

if (len(kp) < len(y_pred)):
	n = len(kp)
	y_pred = y_pred[:n]
else:
	n = len(y_pred)
	kp = kp[:n]


for idx, (x, k) in enumerate(zip(y_pred, kp)):

	#print x
	unit_mouth_kp, N, tilt, mean, unit_kp, keypoints = k[0], k[1], k[2], k[3], k[4], k[5]
	kps = getOriginalKeypoints(x, N, tilt, mean)
	keypoints[48:68] = kps

	imgfile = 'TestVideo/images/' + str(idx + 1).rjust(4, '0') + '.png'
	im = cv2.imread(imgfile)
	drawLips(keypoints, im, c = (255, 255, 255), th = 1, show = False)

	im_out = np.zeros_like(im)
	im1 = np.hstack((im, im_out))
	cv2.imwrite(outputFolder + str(idx) + '.png', im1)

print('Done writing', n, 'images')
