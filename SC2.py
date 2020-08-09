import tensorflow as tf
import numpy as np
print("TensorFlow ver",format(tf.__version__))
tf.random.set_seed(777)  # for reproducibility

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, -1]

# The number of animals
nb_classes = 7  # 0 ~ 6 

# Make Y data as one-hot shape
Y_one_hot = tf.one_hot(y_data.astype(np.int32), nb_classes)
#print(x_data.shape, Y_one_hot.shape)

X = x_data
Y = Y_one_hot

#Weight and bias setting
W = tf.Variable(tf.random.normal((16, nb_classes)), name='weight')
b = tf.Variable(tf.random.normal((nb_classes,)), name='bias')
variables = [W, b]

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
def logit_fn(X):
    return tf.matmul(X, W) + b

def hypothesis(X):
    return tf.nn.softmax(logit_fn(X))

def cost_fn(X, Y):
    logits = logit_fn(X)
    cost_i = tf.keras.losses.categorical_crossentropy(y_true=Y, y_pred=logits, 
                                                      from_logits=True)    
    cost = tf.reduce_mean(cost_i)    
    return cost

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)
        return grads
    
def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), 1)
    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

epochs=10000
optimizer =  tf.keras.optimizers.SGD(learning_rate=0.1)
for i in range(epochs):
    grads = grad_fn(X, Y)
    optimizer.apply_gradients(zip(grads, variables))
    if i % 1000 == 0:
#        print('Loss at epoch %d: %f' %(i+1, cost_fn(X, Y).numpy()))
        acc = prediction(X, Y).numpy()
        loss = cost_fn(X, Y).numpy() 
        print('Steps: {} |  Loss: {:8.5f} |  Accuracy: {:8.5f} |'.format(i, loss, acc))

# Prediction: The data of test animal 
sample_data_x = [[0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,0]]
sample_data_x = np.asarray(sample_data_x, dtype=np.float32)

a = hypothesis(sample_data_x)
sample_animal = tf.argmax(a, 1) # One-hot Encoding (index: 0 or 1)
print("This animal will expected to {}".format(sample_animal))

