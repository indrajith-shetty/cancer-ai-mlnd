'''this file doesn't contain the updated code'''


def main():
    from keras.preprocessing.image import ImageDataGenerator
    from my_model import MyModel

    batch_size = 100
    input_h=150

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is a generator that will read pictures found in
    #  of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        '../data/train',  # this is the target directory
        target_size=(input_h, input_h),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')


    #print(train_generator.class_indices)

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    # this is a similar generator, for validation data
    validation_generator = val_datagen.flow_from_directory(
        '../data/valid',
        target_size=(input_h, input_h),
        batch_size=batch_size,
        class_mode='categorical')


    
    model = MyModel().get_model(input_h)
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(filepath='first_try.h5', verbose=1, save_best_only=True)
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=150 // batch_size,
        callbacks=[checkpointer])

    #model.save_weights('first_try.h5')


if __name__ == '__main__':
    main()
