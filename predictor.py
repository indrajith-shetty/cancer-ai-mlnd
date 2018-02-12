'''this file doesn't contain the updated code'''

from my_model import MyModel
#model
imput_h=15
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
#my_object = MyModel
model=  MyModel().get_model(imput_h)
model.load_weights('first_try.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory(
        '../data/valid',
        target_size=(imput_h, imput_h),
        batch_size=100,
        class_mode=None)

#img =load_img("../data/test/nevus/ISIC_0012551.jpg", target_size=(150, 150))
#x = img_to_array(img)
#x=x.reshape((1,150,150,3))
#pred=model.predict(x)
#print(pred)
#score=model.evaluate_generator(test_generator)

#print(score[1])
preds=model.predict_generator(test_generator,verbose=0)
ans_1st=zip(test_generator.filenames,preds)
import pandas as pd
#df=pd.read_table(preds)
#print(df)
print(preds)
