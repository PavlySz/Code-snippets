'''Code snippets for CNN layers using Tensorflow'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Init weights
def init_weights(shape):
    '''
    Initialize weight of a layer using a normal distribution

    Args:
        - shape: shape of the weights

    Returns:
        - W (tf.Variable): initialized weights of shape `input_shape`
    '''
    init_weight = tf.truncated_normal(shape=shape, stddev=0.1)
    W = tf.Variable(init_weight)

    return W


# Init bias
def init_bias(shape):
    '''
    Initialize bias of a layer to 0.1

    Args:
        - shape: shape of the bias...

    Returns:
        - b (tf.Variable): initialized bias of shape `input_shape`
    '''
    bias = tf.constant(0.1, shape=shape)
    b = tf.Variable(bias)
    return b


# Conv2D
def conv2d(x, shape):
    '''
    tf.nn.Conv2D wrapper

    Args:
        - x: input of shape [batch, H, W, C]
        - shape: shape of the weights..?
        [filter H, filter W, Channels IN, Channels OUT]

    Returns:
        - Conv2D layer; where:
            = Stride = 1 i.e. [1, 1, 1, 1]
            = Padding = 0 (SAME)
            = Activation function: ReLU
    '''
    W = init_weights(shape)
    b = init_bias([shape[3]])
    conv = lambda x, W: tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    feats = conv(x, W) + b
    return tf.nn.relu(feats)


# MaxPool
def max_pool(x):
    '''
    tf.nn.max_pool wrapper

    Args:
        - x: input of shape [batch, H, W, C]

    Returns:
        - MaxPool layer; where:
            = Stride = 2 i.e. [1, 2, 2, 1]
            because e only apply the maxpool function to the H and W of the input
            = Kernel = 2 i.e. [1, 2, 2, 1]
    '''
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Dense
def dense(input_layer, out_size):
    '''
    Dense (fully-connected) layer

    Args:
        - input_layer: input layer
        - out_size: shape of the next layer... I think

    Returns:
        - Dense layer
    '''
    # Get number of neurons in the input layer
    input_size = int(input_layer.get_shape()[1])

    # Initialize weights and biases
    W = init_weights([input_size, out_size])
    b = init_bias([out_size])

    # Perform matrix multiplication
    # i.e. connect every IN neuron with every OUT neuron
    return tf.matmul(input_layer, W) + b
