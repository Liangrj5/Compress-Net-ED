from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


x=tf.placeholder(tf.float32,[None,784],name='xxx')
y_=tf.placeholder(tf.float32,[None,10],name='yyy')


W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)


cross_entropy=-tf.reduce_sum(y_ * tf.log(y))

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
