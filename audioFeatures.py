import soundfile as sf
import pyworld as pw
import scipy.io.wavfile as wav
import sys
import pickle as pkl
import numpy as np
from python_speech_features import logfbank, mfcc

# Normalize audio files before passing

def join_features(mfcc, fbank):
    features = np.concatenate((mfcc, fbank), axis=1)
    return features

if len(sys.argv) < 2:
    print(
        "Usage \n"
        "audioFeatures.py file1.wav file2.wav...\n")
    exit()

d = {}
saveFilename = 'AudioKp.pickle'

for h in range(1, 26):

	file = sys.argv[h]

	(rate, sig) = wav.read(file)
	print rate
	mfccFeat = mfcc(sig, rate)
	#print mfccFeat.shape

	fbankFeat = logfbank(sig, rate)
	#print fbankFeat.shape

	acousticFeatures = join_features(mfccFeat, fbankFeat) 
	#print acousticFeatures.shape"""

	d[h - 1] = acousticFeatures

with open(saveFilename, "wb") as outputFile:
	pkl.dump(d, outputFile)

