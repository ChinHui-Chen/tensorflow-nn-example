import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_round = 5

# Network Parameters
n_input    = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 300 # 1st layer num features
n_hidden_2 = 100 # 2nd layer num features
n_classes  = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
w1 = tf.Variable( tf.random_normal([n_input, n_hidden_1])    )
w2 = tf.Variable( tf.random_normal([n_hidden_1, n_hidden_2]) )
w3 = tf.Variable( tf.random_normal([n_hidden_2, n_classes])  )
    
b1 = tf.Variable( tf.random_normal([n_hidden_1]) )
b2 = tf.Variable( tf.random_normal([n_hidden_2]) )
b3 = tf.Variable( tf.random_normal([n_classes])  )

# Create model
layer_1 = tf.nn.relu( tf.add( tf.matmul(      x,  w1 ), b1 ))      #Hidden layer with RELU activation
layer_2 = tf.nn.relu( tf.add( tf.matmul( layer_1, w2 ), b2 ))      #Hidden layer with RELU activation
pred    =             tf.add( tf.matmul( layer_2, w3 ), b3 )

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    batch_size = 100
    for epoch in range(training_round):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
