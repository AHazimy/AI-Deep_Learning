# Importing all necessary libraries
import cv2
import os

from api import face_extractor

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
	if not os.path.exists('data/train'):
		os.makedirs('data/train')
except OSError:
	print ('Error: Creating directory of data')


data_path = './db_train_real/'
videos = os.listdir(data_path)

for video in videos:

	cam = cv2.VideoCapture(os.path.join(data_path,video))


	count = 0

	while(True):
	
		ret,frame = cam.read()

		if ret:
			if face_extractor(frame) is not None:
				count+=1
				face = cv2.resize(face_extractor(frame), (200,200))
				face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)    
				file_name_path = f'data/train/client001_{str(count)}.jpg'
				cv2.imwrite(file_name_path, face)
				cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
				cv2.imshow('Face Cropper', face)
    
			else:
				print("Face not found")
				pass
		else:
			cam.release()
			cv2.destroyAllWindows()




