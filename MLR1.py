import tensorflow as tf
tf.executing_eagerly()
tf.random.set_seed(777)  # 실행할때 마다 일정한 결과값 출력을 위한 Seed 설정

# data and label
x1 = [ 73., 93., 89., 96., 73.]
x2 = [ 80., 88., 91., 98., 66.]
x3 = [ 75., 93., 90., 100., 70.]
Y = [152., 185., 180., 196., 142.]

# random weights
w1 = tf.Variable(tf.random.normal([1]))
w2 = tf.Variable(tf.random.normal([1]))
w3 = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.00001

print("epoch |    cost    | hypothesis1| hypothesis2| hypothesis3| hypothesis4| hypothesis5|") 
for i in range(10000+1):
    # tf.GradientTape() to record the gradient of the cost function
    with tf.GradientTape() as tape:
        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        
        # calculates the gradients of the cost
        w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
        
        # update w1,w2,w3 and b
        w1.assign_sub(learning_rate * w1_grad)
        w2.assign_sub(learning_rate * w2_grad)
        w3.assign_sub(learning_rate * w3_grad)
        b.assign_sub(learning_rate * b_grad)
        
        if i % 1000 == 0:
             print("{:5} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} |".format(i, cost.numpy(), 
             hypothesis.numpy()[0], hypothesis.numpy()[1], hypothesis.numpy()[2], hypothesis.numpy()[3], hypothesis.numpy()[4]))
