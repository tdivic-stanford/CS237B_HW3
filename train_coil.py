import numpy as np
import tensorflow as tf1
import tensorflow.compat.v2 as tf
import argparse
from utils import *

tf1.compat.v1.enable_eager_execution()

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the CoIL network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT 1: An example of this was given to you in Homework 1's Problem 1 in svm_tf.py. Now you will implement a multi-layer version.
        # HINT 2: You should use either of the following for weight initialization:
        #           - tf1.contrib.layers.xavier_initializer (this is what we tried)
        #           - tf.keras.initializers.GlorotUniform (supposedly equivalent to the previous one)
        #           - tf.keras.initializers.GlorotNormal
        #           - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        # create the initializer
        initializer = tf.keras.initializers.GlorotUniform()
        layer_size = 16

        # initialize the first layer
        self.W1 = tf.Variable(initializer(shape=(in_size, layer_size)), name="weights1")
        self.b1 = tf.Variable(tf.zeros([layer_size, ]), name="bias1")

        # # initialize the second layer
        # self.W2 = tf.Variable(initializer(shape=(layer_size, layer_size)), name="weights2")
        # self.b2 = tf.Variable(tf.zeros([layer_size, ]), name="bias2")

        # initialize the left branch
        self.Wleft = tf.Variable(initializer(shape=(layer_size, out_size)), name="weights_left")
        self.bleft = tf.Variable(tf.zeros([out_size, ]), name="bias_left")

        # initialize the right branch
        self.Wright = tf.Variable(initializer(shape=(layer_size, out_size)), name="weights_right")
        self.bright = tf.Variable(tf.zeros([out_size, ]), name="bias_right")

        # initialize the straight branch
        self.Wstraight = tf.Variable(initializer(shape=(layer_size, out_size)), name="weights_straight")
        self.bstraight = tf.Variable(tf.zeros([out_size, ]), name="bias_straight")
        
        ########## Your code ends here ##########

    def call(self, x, u):
        x = tf.cast(x, dtype=tf.float32)
        u = tf.cast(u, dtype=tf.int8)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for (x,u) where:
        # - x is a (? x |O|) tensor that keeps a batch of observations
        # - u is a (? x 1) tensor (a vector indeed) that keeps the high-level commands (goals) to denote which branch of the network to use 
        # FYI: For the intersection scenario, u=0 means the goal is to turn left, u=1 straight, and u=2 right. 
        # HINT 1: Looping over all data samples may not be the most computationally efficient way of doing branching
        # HINT 2: While implementing this, we found tf.math.equal and tf.cast useful. This is not necessarily a requirement though.
        layer1 = tf.math.tanh(tf.add(tf.matmul(x, self.W1), self.b1))

        # determine which branch we go to using a truth lookup
        left = tf.constant(0, dtype=tf.int8)
        straight = tf.constant(1, dtype=tf.int8)
        right = tf.constant(2, dtype=tf.int8)

        # get the truth vectors as integer vectors
        left_toggle = tf.cast(tf.math.equal(u, left), tf.float32)
        straight_toggle = tf.cast(tf.math.equal(u, straight), tf.float32)
        right_toggle = tf.cast(tf.math.equal(u, right), tf.float32)

        # calculate the output of each branch
        left_output = tf.multiply(left_toggle, tf.add(tf.matmul(layer1, self.Wleft), self.bleft))
        straight_output = tf.multiply(straight_toggle, tf.add(tf.matmul(layer1, self.Wstraight), self.bstraight))
        right_output = tf.multiply(right_toggle, tf.add(tf.matmul(layer1, self.Wright), self.bright))
        output = tf.add(left_output, tf.add(right_output, straight_output))

        return output
        ########## Your code ends here ##########


def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations & goals,
    # - y is the actions the expert took for the corresponding batch of observations & goals
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally
    # setup scalars to penalize the action errors differently
    weights = [10.0, 0.5]

    # set up loss list for each dimension
    losses = [0, 0]

    # calculate loss for each dimension
    for i in range(len(weights)):
        # calculate the l2-norm loss for each element, multiplied by appropriate weight
        loss_vec = tf.math.scalar_mul(weights[i], tf.math.squared_difference(y_est[:, i], y[:, i]))
        losses[i] = tf.reduce_mean(loss_vec)

    # add the two losses for our final scalar loss
    return sum(losses)
    ########## Your code ends here ##########
   

def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y, u):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model (note both x and u are inputs now)
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        # HINT: You did the exact same thing in Homework 1! It is just the networks weights and biases that are different.
        # setup the gradient tape
        with tf.GradientTape() as tape:
            # make a forward pass
            y_est = nn_model.call(x, u)

            # calculate loss for output of the forward pass
            current_loss = loss(y_est, y)

        # calculate the gradient for all weights based on the loss
        grads = tape.gradient(current_loss, nn_model.trainable_variables)

        # run an optimization step on the weights
        optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))

        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y, u in train_data:
            train_step(x, y, u)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'], data['u_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    args.goal = 'all'
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
