from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Lambda, TimeDistributed
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import pickle as pkl
from keras.callbacks import TensorBoard
from time import time

#########################################################################################

timeDelay = 20 
lookBack = 10
n_epoch = 20
n_videos = 12
tbCallback = TensorBoard(log_dir="logs/{}".format(time())) # TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=n_batch, write_graph=True, write_images=True)

#########################################################################################,

# Load the files
with open('data/AudioKp.pickle', 'rb') as pklFile:
	audioKp = pkl.load(pklFile)
with open('data/PCA_reducedKp.pickle', 'rb') as pklFile:
	videoKp = pkl.load(pklFile)


# Get the data

X, y = [], [] 

keysAudio = audioKp.keys()
keysVideo = videoKp.keys()
keys = sorted(list(set(keysAudio).intersection(set(keysVideo))))

for key in tqdm(keys[0: n_videos]):
	audio = audioKp[key]
	video = videoKp[key]
	if (len(audio) > len(video)):
		audio = audio[0: len(video)]
	else:
		video = video[0: len(audio)]
	start = (timeDelay - lookBack) if (timeDelay - lookBack > 0) else 0
	for i in range(start, len(audio) - lookBack):
		a = np.array(audio[i: i + lookBack])
		v = np.array(video[i + lookBack - timeDelay]).reshape((1, -1))
		X.append(a)
		y.append(v)

X = np.array(X)
y = np.array(y)
shapeX = X.shape
shapey = y.shape
print('Shapes:', X.shape, y.shape)
X = X.reshape(-1, X.shape[2])
y = y.reshape(-1, y.shape[2])
print('Shapes:', X.shape, y.shape)

scalerX = MinMaxScaler(feature_range=(0, 1))
scalery = MinMaxScaler(feature_range=(0, 1))

X = scalerX.fit_transform(X)
y = scalery.fit_transform(y)


X = X.reshape(shapeX)
y = y.reshape(shapey[0], shapey[2])

print('Shapes:', X.shape, y.shape)

split1 = int(0.8 * X.shape[0])
split2 = int(0.9 * X.shape[0])

trainX = X[0: split1]
trainy = y[0: split1]
valX = X[split1: split2]
valy = y[split1: split2]
testX = X[split2:]
testy = y[split2:]


# Initialize the model

model = Sequential()
model.add(LSTM(25, input_shape = (lookBack, 39)))
model.add(Dropout(0.25))
model.add(Dense(8))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())

filepath="checkpoint.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallback = [checkpoint]

# model = load_model('obama.h5')

for i in tqdm(range(n_epoch)):
	print('Epoch', (i+1), '/', n_epoch, ' - ', int(100 * (i + 1) / n_epoch))
	model.fit(trainX, trainy, epochs = 1, batch_size = 1, 
		verbose = 1, shuffle = True, callbacks = tbCallback, validation_data = (valX, valy))
	# model.reset_states()
	test_error = np.mean(np.square(testy - model.predict(testX)))
	# model.reset_states()
	print('Test Error: ', test_error)

# Save the model
model.save('obama.h5')
model.save_weights('obama_weights.h5')
print('Saved Model')

