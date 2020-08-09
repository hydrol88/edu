import tensorflow as tf
import numpy as np
tf.random.set_seed(777)  # 실행할때 마다 일정한 결과값 출력을 위한 Seed 설정
print("TensorFlow ver",format(tf.__version__)) # 2.2.0

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

x_test = [[0.476,0.588,0.131,0,-0.251,0.501,0.2132,-0.508]]
y_test = [[1.]]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# NN에서는 backpropagation에서는 바람직하지 않는 조건
# W = tf.Variable(tf.zeros([8,1]), name='weight')
# b = tf.Variable(tf.zeros([1]), name='bias')
W = tf.Variable(tf.random.normal([8,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# TensorFlow v.1:  tf.train is delected in v.2.0
#optimizer =  tf.train.GradientDescentOptimizer(learning_rate=0.01)    
# TensorFlow v.2:  tf.keras.optimizers.SGD is called.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def logistic_regression(features):
 #hypothesis  = tf.divide(1., 1. + tf.exp(-tf.matmul(features, W) + b))
    hypothesis = tf.sigmoid(tf.matmul(features, W) + b) 
    return hypothesis

def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

def grad(features, labels):
    with tf.GradientTape() as tape:
        hypothesis = logistic_regression(features)
        loss_value = loss_fn(hypothesis,labels)
    return tape.gradient(loss_value, [W,b])

EPOCHS = 100000

for step in range(EPOCHS):
    for features, labels  in iter(dataset.batch(len(x_train))):
        #print(dataset.batch(len(x_train)))
        hypothesis = logistic_regression(features)
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
        if step % 10000 == 0:
            predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
            accuracy = accuracy_fn(hypothesis, labels) 
            print("Iter: {}, Loss: {:.4f}, Accuracy: {}, Hypothesis: {:7.4f}, Predicted {:7.4f}".format(step, loss_fn(hypothesis,labels), accuracy, hypothesis.numpy()[5][0], predicted.numpy()[5][0]))     

accuracy = accuracy_fn(hypothesis, labels) 
print("Accuracy of Total Model: {}".format(accuracy))
test_acc = accuracy_fn(logistic_regression(x_test),y_test)
print("Test Result = {}".format(tf.cast(logistic_regression(x_test) > 0.5, dtype=tf.int32)))  
print("Test Set Accuracy: {:.4f}".format(test_acc))
