import cv2
import numpy as np

vid = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
crop_face = [] #to store every 10th cropped faced
data = './face_data/'
counter = 0
file_name = input("Enter name of person : ")
while True:
	ret,frame = vid.read()
	
	if ret == False:
		continue

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = classifier.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key = lambda face:face[2]*face[3])

	for face in faces[-1:]:
		x,y,w,h =face
		offset =10
		cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
		crop = frame[y-offset:y+offset+h,x-offset:x+w+offset]
		crop = cv2.resize(crop,(100,100))

		counter +=1
		if counter%10 ==0:
			crop_face.append(crop)
			print(len(crop_face))
		cv2.imshow('crop',crop)
		
	
	cv2.imshow('stream',frame)


	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

#store into numpy array

crop_face = np.asarray(crop_face)
crop_face = crop_face.reshape(crop_face.shape[0],-1)

#store captured data into .npy file
np.save(data+file_name+'.npy',crop_face)
print("data saved at "+data+file_name+'.npy')

vid.release()
cv2.destroyAllWindows()

