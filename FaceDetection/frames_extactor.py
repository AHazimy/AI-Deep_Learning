# Importing all necessary libraries
import cv2
import os

from api import face_extractor

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the video from specified path
cam = cv2.VideoCapture("./db_train_real/client001_session01_webcam_authenticate_adverse_1.mov")

try:
	
	# creating a folder named data
	if not os.path.exists('data/train'):
		os.makedirs('data/train')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of data')


count = 0

while(True):
	
	# reading from frame
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
		# Release all space and windows once done
		cam.release()
		cv2.destroyAllWindows()




