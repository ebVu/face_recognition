'''
- It need to aovid hard code user 1
- It need to save numpy file to save time
''' 
####
import cv2
import numpy as np
import sys
import os
from numpy import load
from numpy import expand_dims
from keras.models import load_model
# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from keras.models import load_model
# develop a classifier for the 5 Celebrity Faces Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice

sys.path.append(os.getcwd())
from models import base_server
from configs import configs

# Define a Base Server
srv = base_server.BaseServer(model_fp=configs.face_describer_model_fp,
							input_tensor_names=configs.face_describer_input_tensor_names,
							output_tensor_names=configs.face_describer_output_tensor_names,
							device=configs.face_describer_device)

# get the face embedding for one face
def get_embedding(face_pixels):
    # Read example image
    face_pixels = cv2.resize(face_pixels, configs.face_describer_tensor_shape)
    face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Define input tensors feed to session graph
    dropout_rate = 0.5
    input_data = [np.expand_dims(face_pixels, axis=0), dropout_rate]

    # Run prediction
    prediction = srv.inference(data=input_data)

    return prediction[0][0]

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

if (True != os.path.exists('5-celebrity-faces-dataset.npz')):
    # load train dataset
    trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
    print(trainX.shape, trainy.shape)
    # load test dataset
    testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
    # save arrays to one file in compressed format
    savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)

# load the face dataset
data = load('5-celebrity-faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)
# save arrays to one file in compressed format
savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

############
# load dataset
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

# test model on a random example from the test dataset
data = load('5-celebrity-faces-dataset.npz')
testX_faces = data['arr_2']
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])