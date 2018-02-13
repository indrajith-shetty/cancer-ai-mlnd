class MyModel:

    def get_model(self,input_h):
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D
        from keras.layers import Activation, Dropout, Flatten, Dense
        model = Sequential()
        model.add(Conv2D(30,  5, padding='same',activation='relu',input_shape=( input_h, input_h,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(40, 5,activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(60, 3,activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(60,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model
