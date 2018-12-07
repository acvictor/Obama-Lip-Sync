import numpy as np
import sys
import os
from sklearn.decomposition import PCA
import pickle as pkl
from tqdm import tqdm

np.set_printoptions(suppress=True)

def getTilt(keypointsMean):
	# Remove in plane rotation using the eyes
	eyes = np.array(keypointsMean[36:48])
	x = eyes[:, 0]
	y = -1 * eyes[:, 1]
	# print('X:', x)
	# print('Y:', y)
	m = np.polyfit(x, y, 1)
	tilt = np.degrees(np.arctan(m[0]))
	return tilt

def getKeypointFeatures(keypoints):
	# Mean Normalize the keypoints wrt the center of the mouth
	# Leads to face position invariancy
	mouth_kp_mean = np.average(keypoints[48:68])
	keypoints_mn = keypoints - mouth_kp_mean
	
	# Remove tilt
	x_dash = keypoints_mn[:, 0]
	y_dash = keypoints_mn[:, 1]
	theta = np.deg2rad(getTilt(keypoints_mn))
	c = np.cos(theta);	s = np.sin(theta)
	x = x_dash * c - y_dash * s	# x = x'cos(theta)-y'sin(theta)
	y = x_dash * s + y_dash * c # y = x'sin(theta)+y'cos(theta)
	keypoints_tilt = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))

	# Normalize
	N = np.linalg.norm(keypoints_tilt, 2)
	return [keypoints_tilt/N, N, theta, mouth_kp_mean]

d = {}
saveFilename = 'Kp.pickle'

fileDir = os.path.dirname(os.path.realpath('__file__'))

for h in range(1, 26):
	file = sys.argv[h]
	print(file)

	bigList = []

	input = np.array(np.loadtxt(open(file, "rb"), delimiter=",", skiprows=0)).astype("float")

	for i in range(0, input.shape[0]):
		temp = input[i, :]

		keypoints = temp.reshape(68, 2)

		#print keypoints

		mouthMean = np.average(keypoints[48:68], 0)
		keypointsMean = keypoints - mouthMean

		xDash = keypointsMean[:, 0]
		yDash = keypointsMean[:, 1]

		theta = np.deg2rad(getTilt(keypointsMean))

		c = np.cos(theta);	
		s = np.sin(theta)

		x = xDash * c - yDash * s	# x = x'cos(theta)-y'sin(theta)
		y = xDash * s + yDash * c   # y = x'sin(theta)+y'cos(theta)

		keypointsTilt = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1))))

		# Normalize
		N = np.linalg.norm(keypointsTilt, 2)
		
		#print N
		keypointsNorm = keypointsTilt / N

		kpMouth = keypointsNorm[48:68]
		storeList = [kpMouth, N, theta, mouthMean, keypointsNorm, keypoints]
		prev_storeList = storeList
		bigList.append(storeList)

	d[h - 1] = bigList

with open(saveFilename, "wb") as outputFile:
	pkl.dump(d, outputFile)

bigList = []
newList = []

if (os.path.exists(saveFilename)):
	with open(saveFilename, 'rb') as outputFile:
		bigList = pkl.load(outputFile)

for key in tqdm(sorted(bigList.keys())):
	for frameKp in bigList[key]:
		kpMouth = frameKp[0]
		x = kpMouth[:, 0].reshape((1, -1))
		y = kpMouth[:, 1].reshape((1, -1))
		X = np.hstack((x, y)).reshape((-1)).tolist()
		newList.append(X)

X = np.array(newList)

pca = PCA(n_components = 8)
pca.fit(X)
with open('PCA.pickle', 'wb') as file:
	pkl.dump(pca, file)
	
with open('PCA_explanation.pickle', 'wb') as file:
	pkl.dump(pca.explained_variance_ratio_, file)

print('Explanation for each dimension:', pca.explained_variance_ratio_)
print('Total variance explained:', 100 * sum(pca.explained_variance_ratio_))

upsampledKp = {}
for key in tqdm(sorted(bigList.keys())):
	print('Key:', key)
	nFrames = len(bigList[key])
	factor = int(np.ceil(100/25))
	# Create the matrix
	newUnitKp = np.zeros((int(factor * nFrames), bigList[key][0][0].shape[0], bigList[key][0][0].shape[1]))
	newKp = np.zeros((int(factor*nFrames), bigList[key][0][-1].shape[0], bigList[key][0][-1].shape[1]))

	print('Shape of newUnitKp:', newUnitKp.shape, 'newKp:', newKp.shape)
	for idx, frame in enumerate(bigList[key]):
		newKp[(idx*(factor)), :, :] = frame[-1]
		newUnitKp[(idx*(factor)), :, :] = frame[0]

		if (idx > 0):
			start = (idx - 1) * factor + 1
			end = idx * factor
			for j in range(start, end):
				newKp[j, :, :] = newKp[start-1, :, :] + ((newKp[end, :, :] - newKp[start-1, :, :]) * (np.float(j+1-start)/np.float(factor)))
				l = getKeypointFeatures(newKp[j, :, :])
				newUnitKp[j, :, :] = l[0][48:68, :]
		
	upsampledKp[key] = newUnitKp


# Use PCA to de-correlate the points
up = {}
reduced = {}
keys = sorted(upsampledKp.keys())
for key in tqdm(keys):
	x = upsampledKp[key][:, :, 0]
	y = upsampledKp[key][:, :, 1]
	X = np.hstack((x, y))
	up[key] = X
	XTrans = pca.transform(X)
	reduced[key] = XTrans

with open('upsampledKp.pickle', 'wb') as file:
	pkl.dump(up, file)

with open('PCA_reducedKp.pickle', 'wb') as file:
	pkl.dump(reduced, file)

print('Saved Everything')

							
