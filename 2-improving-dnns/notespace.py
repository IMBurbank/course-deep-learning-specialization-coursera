import numpy as np
import tensorflow as tf

# initialize weights
w = tf.Variable(0, dtype=tf.float32)

# define cost function
cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)

# define learning algorithm
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# set up session
init = tf.global_variables_initializer()
session = tf.session()

# initialize variables
session.run(init)

# evaluate a variable
print(session.run(w))
# >> 0.0
# w is still zero as defined above

# run one step of gradient descent
session.run(train)

# and evaluate w again
print(session.run(w))
# >> 0.1

# we can also see what happens if we run 1,000 iterations
for i in range(1000):
        session.run(train)
print(session.run(w))        
# >> 4.99999
# which is very close to the optimal solution of 5
