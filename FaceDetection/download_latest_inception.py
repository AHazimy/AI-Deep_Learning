from keras.applications.inception_v3 import InceptionV3

inc = InceptionV3(input_shape=(224,224,3), weights='imagenet', include_top=False)
