import tensorflow as tf
import numpy as np
tf.executing_eagerly()
tf.random.set_seed(777)  # 실행할때 마다 일정한 결과값 출력을 위한 Seed 설정

xy = np.loadtxt('data_food_est.csv', delimiter=',', dtype=np.float32)

# slice data
X = xy[:, 0:-1]
Y = xy[:, [-1]]

W = tf.Variable(tf.random.normal([4, 1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.001
# hypothesis, prediction function
def predict(X):
    hypothesis = tf.matmul(X, W) + b
    return hypothesis 

n_epochs = 50000

for i in range(n_epochs+1):
    # record the gradient of the cost function
    with tf.GradientTape() as tape: 
        hypothesis = predict(X)
        cost = tf.reduce_mean((tf.square(hypothesis - Y)))
        # calculates the gradients of the loss
        W_grad, b_grad = tape.gradient(cost, [W, b])
        # updates parameters (W and b)
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)
        
        if i % 5000 == 0: 
             print("{:5} | 비용 {:10.4e} | 배추가격 {:10.4f} |".format(i, cost.numpy(), hypothesis.numpy()[0][0]))
