#model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


def get_model():
    model = Sequential()
    model.add(Conv2D(32,  3, padding='valid',input_shape=( 150, 150,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    return model

model=get_model()
model.load_weights('first_try.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory('../data/test',
        target_size=(150, 150),  
        batch_size=600,
        class_mode='categorical')
score=model.evaluate_generator(test_generator,600//16,use_multiprocessing=True)
print(score[1])