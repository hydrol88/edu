import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

tf.random.set_seed(777)  # for reproducibility
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("Fashion MNIST images train data의 크기는 {} 이다".format(train_images.shape))
print("Fashion MNIST images train label의 크기는 {} 이다".format(train_labels.shape))
print("Fashion MNIST images test data의 크기는 {} 이다".format(test_images.shape))
print("Fashion MNIST images test label의 크기는 {} 이다".format(test_labels.shape))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

np.set_printoptions(linewidth = 150)
print(train_images[0]) # 넘파이 배열로 출력한 결과(0~255 사이의 값으로 부츠형상)
print("품목 이름은 {}이고 Label 번호는 {} 이다.".format(class_names[train_labels[0]],train_labels[0])) # Ankle boot에 해당하는 라벨번호 9번

plt.figure()
plt.imshow(train_images[0], cmap = 'gray') # 이미지로 출력한 결과
plt.colorbar()
plt.grid(False)

# NN 모형에 들어가기 전에 0~255으로 포함된 이미지의 픽셀값이 0~1사이의 값을 가지도록 조정
train_images = train_images / 255.0
test_images = test_images / 255.0
print(np.round(train_images[0], 2)) # 0~1사이 값으로 변환 후, 자리수 반올림 조정을 통해 부츠 형상 출력

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 입력레이어는 크기 28x28인 이미지를 입력으로 받아 크기 784(=28x28)인 1차원 배열로 변경, 입력 레이어는 노드 개수는 784개, 외부입력을 신경망으로 가져오는 역할
    keras.layers.Dense(128, activation=tf.nn.relu), # 히든레이어는 128개를 Dense 레이어 사용
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax) # 출력레이어는 신경망의 출력을 외부로 전달역할로 10개 사용, Softmax를 사용하면 10개의 노드값에 적용되는 합이 1이 됨, 10개의 노드값 중 큰 값이 신경망이 예측하는 결과값
])

model.compile(optimizer='adam', #옵티마이져는 학습 데이터셋과 손실함수를 사용하여 모델의 가중치를 업데이트하는 방법 결정
              loss='sparse_categorical_crossentropy',  # 학습시키는 동안 손실함수를 최소화하도록 모델의 가중치 조정
              metrics=['accuracy']) # 평가지표(metrics)는 학습과 평가시 모델 성능을 측정하기 위해 사용, 전체 데이터셋에서 올바르게 분류된 이미지 비율을 표시하는 정확도 사용
model.fit(train_images, train_labels, epochs=10)  # model.fit 메소드를 사용하여 모델을 학습시킴

test_loss, test_acc = model.evaluate(test_images, test_labels)

#print('Test cost:{}, Test accuracy: {}'.format(test_loss, test_acc))

model.save('Fashion_MNISTmodel.h5')
