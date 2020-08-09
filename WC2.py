import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3])
Y = np.array([1,2,3])

def cost_function(W,X,Y):
# Cost function 
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))

# Variables for plotting cost function
W_val = []
cost_val = []

W_values = np.linspace(-3, 5, num=15)
for feed_W in W_values:
    curr_cost = cost_function(feed_W, X, Y)
    print('{:6.3f} | {:10.5f}'.format(feed_W, curr_cost))
    
    # Save the W and Cost 
    W_val.append(feed_W)
    cost_val.append(curr_cost)

# Show the cost function
plt.plot(W_val, cost_val) 
plt.xlabel('W'), plt.ylabel('Cost'), plt.title('W-Cost function')
plt.show()
