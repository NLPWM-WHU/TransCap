import tensorflow as tf
import numpy as np

class CapsLayer(object):
    def __init__(self, aspect, batch_size, num_outputs, vec_len, iter_routing, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type
        self.batch_size = batch_size
        self.iter_routing = iter_routing
        self.aspect = aspect

    def __call__(self, input, mode=None, kernel_size=None, stride=None, embedding_dim=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                '''
                Feature Capsules
                '''
                w, b = get_weights_and_biases([self.kernel_size, embedding_dim, 1, self.num_outputs * self.vec_len],
                                                [self.num_outputs * self.vec_len], 'pc1{}'.format(self.kernel_size))
                input_len = input.shape[1].value
                capsule_len = input_len - self.kernel_size + 1#76
                context_conv = tf.nn.conv2d(
                    input=tf.reshape(input, [self.batch_size, input_len, embedding_dim, 1]),
                    filter=w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="capsules_context{}".format(self.kernel_size)) # [b, 78, 1, 30*16]
                context_conv = tf.nn.bias_add(context_conv,b)
                '''
                Semantic Capsules
                '''
                w_asp, b_asp = get_weights_and_biases([self.kernel_size, embedding_dim, 1, self.num_outputs],
                                                [self.num_outputs], 'pc2{}'.format(self.kernel_size))
                aspect_info = tf.contrib.layers.fully_connected(self.aspect, 1, weights_initializer=
                                tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=0.05), activation_fn=None)

                aspect_conv = tf.nn.conv2d(
                    input=tf.reshape(input, [self.batch_size, input_len, embedding_dim, 1]),
                    filter=w_asp,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="capsules_aspect{}".format(self.kernel_size)) # [b, 78, 1, 30]

                aspect_gate = tf.nn.sigmoid(aspect_conv + tf.tile(tf.expand_dims(aspect_info,1),[1,capsule_len,1,self.num_outputs]))
                aspect_gate = tf.reshape(tf.tile(tf.expand_dims(aspect_gate, -1), [1,1,1,1,self.vec_len]), [self.batch_size, -1, 1, self.num_outputs * self.vec_len])

                # Aspect Routing
                capsules_ASC = tf.expand_dims(context_conv * aspect_gate, -1)
                capsules_DSC = tf.expand_dims(context_conv, -1)
                capsules_concat = tf.concat([capsules_ASC, capsules_DSC], -1) # b, 78, 1, 256, 2
                mode = tf.tile(tf.reshape(mode, [-1, 1, 1, 1, 2]), [1, capsules_concat.shape[1].value, 1, capsules_concat.shape[3].value, 1]) # b, 78, 1, 256, 2
                capsules = tf.reduce_sum(capsules_concat * mode, -1)

                # element-wise maximum
                capsules = tf.transpose(tf.reduce_max(tf.transpose(capsules, [0, 3, 2, 1]), -1, keepdims=True), [0, 3, 2, 1])
                capsules = tf.reshape(capsules, (self.batch_size, self.num_outputs, self.vec_len, 1))  # [b, 16, 16, 1]

                capsules = squash(capsules)
                return (capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                '''
                Class Capsules
                '''
                incap_num = input.shape[1].value
                incap_dim = input.shape[-2].value
                self.input = tf.reshape(input, shape=(self.batch_size, -1, 1, input.shape[-2].value, 1))
                with tf.variable_scope('routing'):
                    b_IJ = tf.constant(np.zeros([1, incap_num, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = dynamic_routing(self.input, b_IJ, self.num_outputs, self.vec_len, self.iter_routing, incap_num, incap_dim)
                    capsules = tf.squeeze(squash(capsules), axis=1)
            return(capsules)


def get_weights_and_biases(w_shape,b_shape,name = None):
    with tf.name_scope(name):
        w_form = tf.truncated_normal(shape = w_shape,stddev= 0.1)
        b_form = tf.constant(0.1,shape = b_shape)
        return [tf.Variable(w_form),tf.Variable(b_form)]

def squash(vector):
    epsilon = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

def dynamic_routing(input, b_IJ, num_caps_j, len_v_j, iter_routing, incap_num, incap_dim):
    batch_size = tf.shape(input)[0]
    pc_num = incap_num #16
    pc_dim = incap_dim #10
    sc_num = num_caps_j
    sc_dim = len_v_j

    W = tf.get_variable('RoutingWeight', shape=(1, pc_num, sc_num, pc_dim, sc_dim), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))
    biases = tf.get_variable('bias', shape=(1, 1, num_caps_j, len_v_j, 1))

    input = tf.tile(input, [1, 1, sc_num, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])

    u_hat = tf.matmul(W, input, transpose_a=True) # [batch_size, pc_num, sc_num, sc_dim, 1]
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop-gradient')

    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            c_IJ = tf.nn.softmax(b_IJ, axis=2)
            if r_iter == iter_routing -1:
                s_J = tf.multiply(c_IJ, u_hat)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

            elif r_iter < iter_routing - 1:
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                v_J_tiled = tf.tile(v_J, [1, pc_num, 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                b_IJ += u_produce_v

    return(v_J)
