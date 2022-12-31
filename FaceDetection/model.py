from keras.applications import VGG16
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchedNormalization
from keras.models import Model

img_rows = 224
img_cols = 224

model = VGG16(weights = 'imagenet',
              include_top = False,
              input_shape = (img_rows, img_cols, 3))

for (i, layer) in enumerate(model.layers):
    print(str(i), layer.__class__.__name__, layer.trainable)
    
    
def addTopModel(bottom_model, num_classes, D=256):
    top_model = bottom_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(D, activation="relu")(top_model)
    top_model = Dense(D, activation="relu")(top_model)
    top_model = Dense(D, activation="relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "sigmoid")(top_model)
    
    return top_model

num_classes = 2

FC_Head = addTopModel(model, num_classes)

modelnew = Model(inputs=model.input, outputs=FC_Head)

print(modelnew.summary())

from keras.preprocessing.image import ImageDataGenerator

dir_train = ''
dir_test = ''

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_batchsize = 32
val_batchsize = 16
img_rows = 224
img_cols = 224

train_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(img_rows, img_cols),
    bath_size=train_batchsize,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    dir_test,
    target_size = (img_rows, img_cols),
    batch_size = val_batchsize,
    class_mode = 'categorical',
    shuffle = False
)


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
                "family_vgg.h5",
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
             )

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
             )

callbacks = [earlystop, checkpoint]

modelnew.compile(loss='binary_crossentropy',
                 optimizer=RMSprop(lr=0.001),
                 metrics=['accuracy']
             )

nb_train_samples = 800
nb_validation_samples = 200
epochs = 5
batch_size = 16

history = modelnew.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

modelnew.save("family_vgg.h5")