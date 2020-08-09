import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#이전에 학습완료된 모델 불러오기
model = tf.keras.models.load_model('MNISTmodel.h5')
model.summary()
     
# new test data
test_num = plt.imread('Park.png')
test_num = test_num[:,:,0]
test_num = (test_num > 0.1) * test_num
test_num = test_num.astype('float32')

# plt.imshow(test_num, cmap ="Greys", interpolation='nearest')
plt.imshow(test_num, cmap ="Greys", interpolation='nearest')
test_num = test_num.reshape((1, 28, 28, 1))

plt.show()
print('당신이 쓴 숫자는', model.predict_classes(test_num),'입니다')
