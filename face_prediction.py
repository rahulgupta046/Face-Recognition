import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

#Initialize Camera

vid = cv2.VideoCapture(0)

#classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip = 0
dataset_path = './face_data/'

face_data = []
label = []

class_id = 0 # labels of different names
names = {} #dictionary to map


#preparing data
for fx in os.listdir(dataset_path):
	if fx.endswith('npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		#for labels
		ids = class_id*np.ones(data_item.shape[0])
		class_id +=1
		label.append(ids)

#cannot directly make into numpy array, to include all faces we need to concat along axis
face_data = np.concatenate(face_data,axis = 0)
label = np.concatenate(label, axis = 0)
# print(face_data.shape)
# print(label.shape)
#training_set = np.concatenate((face_data,label),axis = 1)


#Testing - take data from video stream
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(face_data,label)

while True:
	ret, frame = vid.read()
	if ret == False:
		continue

	faces = classifier.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face
		offset = 10
		cropped = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		cropped = cv2.resize(cropped,(100,100))

		prediction = model.predict([cropped.flatten()])

		#Display name and rectaingle around frame
		pred_name = names[int(prediction)]
		cv2.putText(frame, pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow("faces",frame)

	key = cv2.waitKey(1) &0xFF
	if key == ord('q'):
		break
vid.release()
cv2.destroyAllWindows()




