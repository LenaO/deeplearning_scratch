# This is the tensorflow implementation of the siamese network
# with hooks for tensorboard
from __future__ import print_function, division
import tensorflow as tf
from keras.layers import Dense, Conv2D, Input, BatchNormalization, MaxPooling2D, Lambda, Add, Flatten, Conv2DTranspose, Concatenate, Cropping2D, Dropout, Activation, Permute
from keras.models import Model
from keras.layers.merge import concatenate
import h5py
import logging
import time
from math import sqrt


def put_kernels_on_grid(kernel, pad=1):
    """Visualize conv filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
        kernel: tensor of shape [Y, X, NumChannels, NumKernels]
        pad:    number of black pixels around each filter (between them)
    Return:
        Tensor of shape [NumChannels, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, 1].
    """
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant([[pad,pad],[pad, pad],[0,0],[0,0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (2, 0, 1, 3))

    # scaling to [0, 255] is not necessary for tensorboard
    return x


def variable_summary(var, name, scalar=True, histogram=True, image=False):
    """Attach summaries for tensorboard"""
    with tf.name_scope(name):
        if scalar:
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
        if histogram:
            tf.summary.histogram('histogram', var)
    if image:
        w_img = tf.squeeze(var)
        shape = w_img.get_shape().as_list()
        if len(shape) == 2:  # dense layer kernel case
            if shape[0] > shape[1]:
                w_img = tf.transpose(w_img)
                shape = w_img.get_shape().as_list()
                w_img = tf.reshape(w_img, [1,shape[0],shape[1],1])
        elif len(shape) == 1:  # bias case
            w_img = tf.reshape(w_img, [1,shape[0],1,1])
        else:  # convnet case
            w_img = put_kernels_on_grid(var, pad=1)
            # switch to channels_first to display every kernel as a separate image
            #w_img = tf.transpose(w_img, perm=[2, 0, 1])
            #shape = w_img.get_shape().as_list()
            #w_img = tf.reshape(w_img, [shape[0],shape[1],shape[2],1])
        tf.summary.image(name, w_img, max_outputs=3, collections=['kernel_summary'])


def make_parallel(model, devices=2):
    """
    Makes a model parallel by copying the graph of the model to each given device. Incoming batches will be split, each split
    gets passed through one of the created network instances. At the end, the two branches will be merged. This allows for
    larger batch sizes.

    Args:
        model (keras.models.Model): Model to make parallel.
        devices (int or list of str): Specify which devices to use. If this is an integer, the given number of GPUs will be used.
                                      Alternatively, you can pass a list of device names (e.g. ["/cpu:0", "/gpu:0"]), to specify
                                      in a more detailed way which devices to use.
    Returns:
        keras.models.Model, split over the given number of GPUs.
    Remarks:
        Based on https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
        (Github link: https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py)
    """
    def get_slice(data, idx, parts):
        """
        Gets a slice of the batch for a given index.
        """
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for _ in range(len(model.outputs)):
        outputs_all.append([])
    if isinstance(devices, int):
        # Assume that we want to use GPUs
        devices = ["/gpu:{}".format(gpu) for gpu in range(devices)]
    # Place a copy of the model on each GPU, each getting a slice of the batch
    for index, device in enumerate(devices):
        with tf.device(device):
            with tf.name_scope('tower_{}'.format(index)):
                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': index, 'parts': len(devices)})(x)
                    inputs.append(slice_n)
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])
    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        # Concatenate the outputs along the batch axis
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))
        return Model(inputs=model.inputs, outputs=merged)


class Network(object):
    def __init__(self, input_shapes, summary=True):
        # general neural network attributes - to be filled by implementing classes
        self.inputs = []
        self.perm_inputs = []
        for i, shape in enumerate(input_shapes):
            with tf.name_scope('input%d'%i):
                self.inputs.append(Input(batch_shape=shape, dtype='float32', name='input%d'%i))
                self.perm_inputs.append(Permute((2,3,1), name='%d_permute'%i)(self.inputs[-1]))
        self.outputs = []
        self.layers = []
        # counters for naming of layers
        self.num_conv_layers = 0
        self.num_pool_layers = 0
        self.num_batch_norm_layers = 0
        self.num_dense_layers = 0
        self.num_deconv_layers = 0
        self.num_concat_layers = 0
        self.num_dropout_layers = 0
        self.num_activation_layers = 0
        self.model = None
        self.model_to_train = None
        self.calculate_summary = summary
        self.is_parallel = False
        self.log = logging.getLogger(self.__class__.__name__)

    def make_parallel(self, num_gpus):
        self.model_to_train = make_parallel(self.model, num_gpus)
        self.outputs = self.model_to_train.outputs
        self.inputs = self.model_to_train.inputs
        self.layers = []
        for l in self.model_to_train.layers:
            if hasattr(l, 'layers'):
                for l in l.layers:
                    self.layers.append(l)
            else:
                self.layers.append(l)
        self.is_parallel = True

    def get_train_updates(self):
        update_ops = []
        for layer in self.layers:
            for update in layer.updates:
                if not self.is_parallel or 'tower' in update.name:
                    update_ops.append(update)
        return update_ops

    # --- Network creation ---
    @staticmethod
    def apply_layer(layer, inputs):
        outputs = []
        if len(inputs) == 1:
            outputs.append(layer(inputs[0]))
        else:
            name = layer.name
            for i,input_tensor in enumerate(inputs):
                layer.name = name+'_%s'%chr(97+i)
                outputs.append(layer(input_tensor))
        return outputs

    def get_conv_layer(self, inputs, num_filters=16, filter_size=3, stride=1, batch_norm=True, activation=True, summary=True, summary_activation=True):
        self.num_conv_layers += 1
        name = 'conv_%d'%self.num_conv_layers
        with tf.name_scope(name):
            conv = Conv2D(filters=num_filters, kernel_size=filter_size, strides=stride, activation=None, kernel_initializer='he_normal', name=name)
            outputs = Network.apply_layer(conv, inputs)
            self.add_summary(conv, summary)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, conv.kernel)
        if batch_norm:
            outputs = self.get_batch_norm_layer(outputs, summary=summary)
        if activation:
            outputs = self.get_activation_layer(outputs, summary_activation=summary_activation)
        return outputs

    def get_deconv_layer(self, inputs, num_filters=16, filter_size=3, stride=1, batch_norm=True, activation=True, summary=True):
        self.num_deconv_layers += 1
        name = 'deconv_%d'%self.num_deconv_layers
        with tf.name_scope(name):
            deconv = Conv2DTranspose(filters=num_filters, kernel_size=filter_size, strides=stride, padding='valid', activation=None, kernel_initializer='he_normal', name=name)
            output_shapes = [deconv.compute_output_shape(t.shape) for t in inputs]
            outputs = Network.apply_layer(deconv, inputs)
            for t,shape in zip(outputs, output_shapes):
                t.set_shape(shape)
            self.add_summary(deconv, summary)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, deconv.kernel)
        if batch_norm:
            outputs = self.get_batch_norm_layer(outputs, summary=summary)
        if activation:
            outputs = self.get_activation_layer(outputs)
        return outputs

    def get_batch_norm_layer(self, inputs, summary=True):
        self.num_batch_norm_layers += 1
        outputs = []
        name = 'batch_norm_%d'%self.num_batch_norm_layers
        with tf.name_scope(name):
            batch_norm = BatchNormalization(name=name)
            outputs = Network.apply_layer(batch_norm, inputs)
            self.add_summary(batch_norm, summary)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, batch_norm.gamma)
        return outputs

    def get_activation_layer(self, inputs, summary_activation=True):
        self.num_activation_layers += 1
        outputs = []
        name = 'activation_%d'%self.num_activation_layers
        with tf.name_scope(name):
            activation = Activation('relu', name=name)
            outputs = Network.apply_layer(activation, inputs)
            if summary_activation:
                activation.activation_summary = True
                #tf.add_to_collection('activation_summary', outputs[0])
        return outputs

    def get_pool_layer(self, inputs, pool_size=2):
        self.num_pool_layers += 1
        name = 'pool_%d'%self.num_pool_layers
        with tf.name_scope(name):
            pool = MaxPooling2D(pool_size=pool_size, name=name)
            outputs = Network.apply_layer(pool, inputs)
        return outputs

    def get_dense_layer(self, inputs, num_units, activation='sigmoid', summary=True):
        self.num_dense_layers += 1
        name = 'dense_%d'%self.num_dense_layers
        with tf.name_scope(name):
            dense = Dense(units=num_units, activation=activation, kernel_initializer='he_normal', name=name)
            outputs = []
            for input_tensor in inputs:
                if len(input_tensor.shape) >= 3:
                    outputs.append(Flatten()(input_tensor))
                else:
                    outputs.append(input_tensor)
            outputs = Network.apply_layer(dense, outputs)
            self.add_summary(dense, summary)
        return outputs

    def get_concat_layer(self, inputs):
        self.num_concat_layers += 1
        name = 'concat_%d'%self.num_concat_layers
        with tf.name_scope(name):
            cropped = []
            min_h = min([t.shape.as_list()[1] for t in inputs])
            min_w = min([t.shape.as_list()[2] for t in inputs])
            for input_tensor in inputs:
                h,w = input_tensor.shape.as_list()[1:3]
                if not (h == min_h and w == min_w):
                    crop = Cropping2D(cropping=(((h-min_h)//2, (h-min_h+1)//2),((w-min_w)//2,(w-min_w+1)//2)))
                    cropped.append(crop(input_tensor))
                else:
                    cropped.append(input_tensor)

        return Concatenate(axis=-1)(cropped)

    def get_dropout_layer(self, inputs, dropout_rate=0):
        self.num_dropout_layers += 1
        name = 'dropout_%d'%self.num_dropout_layers
        with tf.name_scope(name):
            dropout = Dropout(dropout_rate)
            outputs = Network.apply_layer(dropout, inputs)
        return outputs

    def make_output(self, tensor):
        if isinstance(tensor, list):
            self.outputs = tensor + self.outputs
        else:
            self.outputs = [tensor] + self.outputs
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model_to_train = self.model
        self.model.layers[1].needs_activation_summary = True
        self.layers = self.model.layers

    # --- Logging, Saving and Restoring ---
    def add_summary(self, layer, summary=True):
        # add variable summaries for tensorboard
        if summary and self.calculate_summary:
            with tf.name_scope('summaries'):
                for weight in layer.weights:
                    name = weight.name.split('/')[-1].split(':')[0]
                    if name == 'kernel':
                        variable_summary(weight, name, image=True)
                    else:
                        variable_summary(weight, name, image=False)

    def save_weights(self, fname):
        self.log.info('Saving weights to %s'%fname)
        with h5py.File(fname, 'w') as f:
            f.attrs['info'] = 'Weights saved by %s at %s'%(self.__class__.__name__, time.strftime('%Y-%m-%d %H:%M:%S'))
            for layer in self.model.layers:
                for weight in layer.weights:
                    name = weight.name
                    value = weight.eval()
                    dset = f.require_dataset(name, value.shape, dtype=value.dtype)
                    dset[:] = value

    def load_weights(self, fname):
        self.log.info('Loading weights from %s'%fname)
        with h5py.File(fname, 'r') as f:
            for layer in self.model.layers:
                weights = []
                for i in range(len(layer.weights)):
                    name = layer.weights[i].name
                    try:
                        weights.append(f[name][:])
                    except KeyError:
                        pass
                if len(weights) == len(layer.weights):
                    layer.set_weights(weights)
                    self.log.debug('Restored weights for %s'%layer.name)


class SiameseNetwork(Network):
    def __init__(self, input_shape, summary=True):
        super(SiameseNetwork, self).__init__([input_shape, input_shape], summary)
        # initialize variables
        self.net = None
        self.net1 = self.perm_inputs[0]
        self.net2 = self.perm_inputs[1]
        self.log = logging.getLogger(self.__class__.__name__)

    def add_conv_layer(self, num_filters=16, filter_size=3, stride=1, batch_norm=True, summary=True):
        outputs = self.get_conv_layer([self.net1, self.net2], num_filters, filter_size, stride, batch_norm, summary)
        self.net1, self.net2 = outputs

    def add_pool_layer(self):
        outputs = self.get_pool_layer([self.net1, self.net2])
        self.net1, self.net2 = outputs

    def add_dropout_layer(self, dropout=0):
        self.net1, self.net2 = self.get_dropout_layer([self.net1, self.net2], dropout_rate=dropout)

    def add_block(self, width=2, num_filters=16, batch_norm=True, pool=True, summary=True):
        for i in range(width):
            self.add_conv_layer(num_filters=num_filters, filter_size=3, stride=1, batch_norm=batch_norm, summary=summary)
        if pool:
            self.add_pool_layer()

    def add_dense_layer(self, num_units, activation='sigmoid', summary=True):
        outputs = self.get_dense_layer([self.net1, self.net2], num_units=num_units, summary=summary, activation=activation)
        self.net1, self.net2 = outputs

    def add_branch_output(self, num_labels=3, summary=True):
        with tf.name_scope('branch_prediction'):
            outputs = self.get_dense_layer([self.net1, self.net2], num_units=num_labels, summary=summary, activation=None)
            output = Concatenate(axis=-1)(outputs)
            self.outputs.append(output)

    def combine_branches(self, num_labels=1, distance_function='L1', weights='vector'):
        # Distance function allowed values: 'L1', 'L2'
        # weights allowed values: 'vector', 'matrix', None
        if weights is None:
            assert num_labels == 1
        with tf.name_scope('prediction'):
            # calculate distance between net1 and net2 depending on distance_function
            combined = Add()([Lambda(lambda x: -1*x)(self.net1), self.net2])
            if distance_function == 'L1':
                distance = Lambda(lambda x: abs(x))(combined)
            if distance_function == 'L2':
                distance = Lambda(lambda x: x**2)(combined)
            # reduce distance to one value
            if weights is None:
                reduced = Lambda(lambda x: tf.reduce_sum(x, axis=[-1]))(distance)
            if weights == 'vector':
                dense = Dense(units=num_labels, activation=None, kernel_initializer='he_normal', name='output')
                reduced = dense(distance)
            reduced = Lambda(lambda x: tf.reshape(x, [-1, num_labels]))(reduced)
            self.net = reduced
            outputs = []
            with tf.name_scope('distance'):
                outputs.append(Lambda(lambda x: x[:,0:1])(self.net))
            if num_labels > 1:
                with tf.name_scope('direction'):
                    outputs.append(Lambda(lambda x: x[:,1:4])(self.net))
        self.make_output(outputs)


class MultibranchUNet(Network):
    def __init__(self, input_shapes, summary=True):
        super(MultibranchUNet, self).__init__(input_shapes, summary)
        self.branches = self.perm_inputs[:]
        self.net = None
        self.saved_layers = [[] for _ in range(len(self.branches))]
        self.log = logging.getLogger(self.__class__.__name__)

    def add_block(self, branch=0, width=1, num_filters=16, filter_size=3, stride=1, batch_norm=True, pool=True, summary=True, summary_activation=True):
        for i in range(width):
            summ = summary if i == width-1 else False
            self.branches[branch] = self.get_conv_layer([self.branches[branch]], num_filters, filter_size, stride, batch_norm, summ, summary_activation)[0]
        if pool:
            self.branches[branch] = self.get_pool_layer([self.branches[branch]])[0]

    def add_dropout_layer(self, branch=0, dropout=0):
        self.branches[branch] = self.get_dropout_layer([self.branches[branch]], dropout_rate=dropout)[0]

    def add_unet_down(self, branch=0, depth=4, width=2, filters=32, max_filters=128, summary=True, summary_activation=True, name='gray'):
        """Create downscaling branch of U-Net.
        Args:
            depth (int): depth of the net (number of blocks)
            width (int or list): width of each block
            filters (int or list): number of filters at each block.
                If int, filters for block i are filters*2**i.
            summary (bool or list): summary option for each block
        """
        if not isinstance(width, list):
            width = [width for _ in range(depth)]
        if not isinstance(filters, list):
            filters = [min(filters*2**i, max_filters) for i in range(depth)]
        if not isinstance(summary, list):
            summary = [summary for _ in range(depth)]

        network = self.branches[branch]
        # --------------- Down --------------------------------
        for j in range(depth-1):
            with tf.name_scope('%s_block_%d'%(name, j)):
                for i in range(width[j]):
                    summ = summary[j] if i == width[j]-1 else False
                    network = self.get_conv_layer([network], num_filters=filters[j], filter_size=3, stride=1, batch_norm=True, summary=summ, summary_activation=summary_activation)[0]
                self.saved_layers[branch].append(network)
                network = self.get_pool_layer([network])[0]
        # --------------- Bottom ------------------------------
        with tf.name_scope('%s_bottom'%name):
            for i in range(width[-1]):
                summ = summary[-1] if i == width[-1]-1 else False
                network = self.get_conv_layer([network], num_filters=filters[-1], filter_size=3, stride=1, batch_norm=True, summary=summ, summary_activation=summary_activation)[0]

        self.branches[branch] = network

    def add_unet_up(self, depth=4, width=2, filters=32, max_filters=128, summary=True, summary_activation=True):
        """Create upscaling branch of U-Net.
        Args:
            depth (int): depth of the net (number of blocks)
            width (int or list): width of each block
            filters (int or list): number of filters at each block.
                If int, filters for layer i are filters*2**i.
            summary (bool or list): summary option for each block
        """
        assert self.net is not None, "Branches are not merged. Call merge_branches."
        for saved in self.saved_layers:
            assert saved is not None, "No saved layers for branch. Call add_unet_down."
            assert len(saved) == depth-1, "Depth of saved_layers is not the same as depth. Call add_unet_down with same depth as add_unet_up"
        if not isinstance(width, list):
            width = [width for _ in range(depth)]
        if not isinstance(filters, list):
            filters = [min(filters*2**i, max_filters) for i in range(depth)]
        if not isinstance(summary, list):
            summary = [summary for _ in range(depth)]

        network = self.net
        # --------------- Up ----------------------------------
        for j in range(depth-1)[::-1]:
            with tf.name_scope('merged_block_%d'%j):
                network = self.get_deconv_layer([network], num_filters=filters[j], filter_size=2, stride=2, batch_norm=True, summary=False)[0]
                network = self.get_concat_layer([network]+[saved[j] for saved in self.saved_layers])

                for i in range(width[j]):
                    summ = summary[j] if i == width[j] else False
                    network = self.get_conv_layer([network], num_filters=filters[j], filter_size=3, stride=1, batch_norm=True, summary=summ, summary_activation=summary_activation)[0]
        self.net = network

    def merge_branches(self):
        if len(self.branches) == 1:
            self.net = self.branches[0]
        else:
            self.net = self.get_concat_layer(self.branches)

    def add_output_layer(self, num_filters=16, summary=True, summary_activation=True):
        with tf.name_scope('output'):
            self.net = self.get_conv_layer([self.net], num_filters=num_filters, filter_size=3, stride=1, summary=summary, summary_activation=summary_activation)[0]
        self.make_output(self.net)

