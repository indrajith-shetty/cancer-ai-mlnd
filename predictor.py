#model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array


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
test_generator=test_datagen.flow_from_directory(
        '../data/test',
        target_size=(150, 150),  
        batch_size=16,
        class_mode='categorical')

#img =load_img("/Users/Indra/Documents/Online courses/Udacity/paid courses/machine-learning-engineer/GitHub-Repo/data/test/nevus/ISIC_0012551.jpg", target_size=(150, 150))
#x = img_to_array(img)
#x=x.reshape((1,150,150,3))
#pred=model.predict(x)
#print(pred)
score=model.evaluate_generator(test_generator)
print(score[1])