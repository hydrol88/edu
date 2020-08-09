import tensorflow as tf
import numpy as np
tf.executing_eagerly()
tf.random.set_seed(777)  # 실행할때 마다 일정한 결과값 출력을 위한 Seed 설정

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

data = np.array([
# X1,  X2,  X3,  y
[ 73., 80., 75., 152. ], [ 93., 88., 93., 185. ], [ 89., 91., 90., 180. ], [ 96., 98., 100., 196. ],
[ 73., 66., 70., 142. ]], dtype=np.float32)

# slice data
X = data[:, :-1]
y = data[:, [-1]]

W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.00001
# hypothesis, prediction function
def predict(X):
    hypothesis = tf.matmul(X, W) + b
    return hypothesis 

n_epochs = 100000
print("epoch |    cost    | hypothesis1| hypothesis2| hypothesis3| hypothesis4| hypothesis5|")

for i in range(n_epochs+1):
    # record the gradient of the cost function
    with tf.GradientTape() as tape: 
        hypothesis = predict(X)
        cost = tf.reduce_mean((tf.square(hypothesis - y)))
        # calculates the gradients of the loss
        W_grad, b_grad = tape.gradient(cost, [W, b])
        # updates parameters (W and b)
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)
        
        if i % 1000 == 0: 
             print("{:5} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} |".format(i, cost.numpy(), 
             hypothesis.numpy()[0][0], hypothesis.numpy()[1][0], hypothesis.numpy()[2][0], hypothesis.numpy()[3][0], hypothesis.numpy()[4][0]))
