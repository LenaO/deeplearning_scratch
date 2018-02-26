from mpi4py import MPI
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import math
from keras import backend as K
from brainmap.misc import get_logger
import os
import numpy as np
import time
from bigbrain.utils import flatten
from bigbrain.evaluation import distance_plot_for_tensorboard, confusion_matrix_for_tensorboard, coords_xyz_plot_for_tensorboard


def switch_loss(labels, predictions, weights=1.0, c1=0.999, c2=1.0, c3=1.0, switch_value=1.0,
                scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Exponential loss for labels <= switch_value, linear loss for labels > switch_value.
    Exponential function is defined by `f(x) = b*x^2` with the constraints `f(c1) = c1` and `f(c2) = c3*c2`.
    Thus, for each label `l` and prediction `p` the following is calculated:
    ```
    b*(l-p)^a             if l <= d
    |l-p|                 if l > d
    ```
    """
    def get_exp_params(c1=0.999, c2=1.0, c3=1.0):
        a = math.log(float(c1)/(c2*c3),c1)/(1-math.log(float(c3),c1))
        b = c1/c1**a
        return a, b
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    a,b = get_exp_params(c1, c2, c3)
    with ops.name_scope(scope, "switch_loss",
                        (predictions, labels, weights)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        error = math_ops.subtract(predictions, labels)
        abs_error = math_ops.abs(error)
        exp_error = b*(abs_error**a)
        losses = tf.where(labels <= switch_value, exp_error, abs_error)
        return tf.losses.compute_weighted_loss(
            losses, weights, scope, loss_collection, reduction=reduction)


def mean_squared_norm_loss(labels, predictions, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    with ops.name_scope(scope, "mean_squared_norm_loss", (predictions, labels, weights)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        divisor = tf.maximum(labels, 1.0)
        error = math_ops.square(math_ops.divide(math_ops.subtract(predictions, labels),divisor))
        return tf.losses.compute_weighted_loss(
            error, weights, scope, loss_collection, reduction=reduction)


def mean_squared_log_loss(labels, predictions, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    with ops.name_scope(scope, "mean_squared_norm_loss", (predictions, labels, weights)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        error = math_ops.square(math_ops.subtract(math_ops.log(tf.maximum(predictions, 1.0)), math_ops.log(tf.maximum(labels, 1.0))))
        return tf.losses.compute_weighted_loss(
            error, weights, scope, loss_collection, reduction=reduction)


default_train_params = {
    # --- General Options ---
    'iterations':100000,
    # When to stop training. Allowed values are 'constant' and 'max_global'
    'stop_criterium':'constant',
    # weights to finetune from
    'finetune': False,
    # Mode determines the treatment of target variables (for segmentation is NOT one-hot-encoded)
    # Allowed values are 'regression', 'segmentation'
    'mode': 'regression',

    # --- Optimizer ---
    # Optimizer to use during training. Allowed values are 'adam', 'sgd', 'rmsprop'
    'solver': 'adam',
    # Initial learning rate. For solvers 'adam' and 'sgd'
    'learning_rate':0.01,
    # only for two branch u-net, where pmaps could be learned with a larger lr. Lr for pmaps will be learning_rate*pmaps_learning_rate_factor
    'pmaps_learning_rate_factor':None,
    'rms_decay': 0.9,
    # Adaptive decreasing of learning rate. Allowed values are 'constant', 'step', 'custom'
    # For lr_policy 'step': lr = learning_rate * gamma**(iteration/stepsize)
    # For lr_policy 'custom': according to 'schedule' at specified iteration, lr is set to value
    'lr_policy':'constant',
    'gamma':0.1,
    'stepsize': 500,
    'schedule': [(1000,0.01),(2000,0.001)],  # e.g. [(1000,0.01),(2000,0.001)]
    # Momentum. For solvers 'adam' and 'sgd'
    'momentum':0.9,
    # 2nd momentum for solver 'adam'
    'momentum2':0.999,

    # --- Loss and Metrics ---
    # Weight decay factor for l2 regularization of kernels
    'weight_decay':0.005,
    # Loss function for training. Allowed values are 'huber', mean_squared', 'crossentropy', 'switch', 'linear', 'log'
    # Loss functions, weights and parameters should be a list if network has several outputs
    # Loss 'huber' has parameter huber_delta to control the transition point between squared and linear parts of the loss
    # Loss 'switch' has parameter switch_c1, switch_c2, switch_c3 to control the shape of the exponential and switch_value to control the transition between exponential and linear loss
    'loss_name': 'huber',
    'loss_weight': 1,
    'huber_delta': 30,
    'switch_c1': 5.,
    'switch_c2': 3.,
    'switch_c3': 220.,
    'switch_value': 30,
    # Metrics to be caluculated during training. Allowed values are 'inv_huber', 'inv_mean_squared', 'inv_switch', 'inv_linear', 'inv_log'.
    # Metrics are calculated by adding metric for each of the outputs together. Metrics can use loss parameters.
    'metrics_names': ['inv_huber'],

    # --- Testing ---
    # how many iterations should be tested?
    'test_iter':16,
    # at which interval should we test?
    'test_interval':200,

    # --- Display and Save ---
    'display':100,
    'snapshot':500,
    'snapshot_dir':'',
    'log_dir':'',
    # at what interval should be logged to tensorboard?
    'log_interval': 200,
    'show_activation': False,
    'labels': ['distance',],
    'eval_image_shape': (1,1000,1000,3),
    'eval_coords_image_shape': (1,500,1500,3),

}


class TFTrainer(object):
    def __init__(self, net, params, train_batch_iter, test_batch_iter, timestamp=''):
        """ Create Tensorflow Neural Network Trainer.

        Args:
            net: Keras model (created by build_net in net_definition.py)
            params (dict): training parameters (from config.py)
            train_batch_iter (BatchIterator): iterator over training data
            test_batch_iter (BatchIterator)
        """
        self.log = get_logger(self.__class__.__name__, rank=MPI.COMM_WORLD.Get_rank())
        self.log.info('Initializing NNTrainer')
        self.net = net
        self.train_batch_iter = train_batch_iter
        self.test_batch_iter = test_batch_iter
        # set params
        for name in default_train_params.keys():
            setattr(self, name, params.get(name, default_train_params[name]))
            self.log.info('Setting {} to {:.200}'.format(name, '{}'.format(getattr(self, name))))
        if not isinstance(self.loss_name, list):
            self.loss_name = [self.loss_name]
        for i, metric_name in enumerate(self.metrics_names):
            if not isinstance(metric_name, list):
                self.metrics_names[i] = [metric_name for _ in range(len(self.net.outputs))]
        for param in ('huber_delta', 'loss_weight', 'switch_c1', 'switch_c2', 'switch_c3', 'switch_value'):
            if not isinstance(getattr(self, param), list):
                setattr(self, param, [getattr(self, param)]*len(self.loss_name))
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        self.log_dir = os.path.join(self.log_dir, timestamp)

        self.train_summary = None
        self.test_summary = None
        self.global_step_var = tf.Variable(0, name='global_step', trainable=False)
        # variables and operations for training
        if self.mode == 'regression':
            self.target_var = []
            for i,output in enumerate(self.net.outputs):
                self.target_var.append(tf.placeholder(tf.float32, shape=output.shape.as_list(), name='target%d'%i))
        elif self.mode == 'segmentation':
            self.target_var = [tf.placeholder(tf.int32, shape=self.net.outputs[0].shape.as_list()[:-1], name='target')]
        else:
            raise NotImplementedError
        self.metrics_ops = self.get_metrics()
        self.loss_op = self.get_loss()
        self.train_op, self.lr_var = self.get_train_step()
        self.train_update_op = tf.group(*self.net.get_train_updates())
        self.prediction_op = self.get_prediction()
        self.init_op = None
        self.checkpoint_saver = None
        self.sess = None

        # summary operations
        self.train_summary_op = None
        self.test_summary_op = None
        self.eval_image_var = None
        self.test_metrics_vars = None
        self.test_loss_var = None
        self.add_summary()

        # initialize variables for finding best model (stop_criterium: max_global)
        self.best_metric = -1000
        self.best_iteration = 0

        # initialize variables and saver
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.checkpoint_saver = tf.train.Saver(max_to_keep=5)

    def get_loss(self):
        assert hasattr(self, 'target_var')
        with tf.name_scope('loss'):
            loss_op = 0
            for i,output in enumerate(self.net.outputs):
                if self.loss_name[i] == 'huber':
                    loss_op += tf.losses.huber_loss(self.target_var[i], output, delta=self.huber_delta[i], weights=self.loss_weight[i])
                elif self.loss_name[i] == 'mean_squared':
                    loss_op += tf.losses.mean_squared_error(self.target_var[i], output, weights=self.loss_weight[i])
                    #self.metrics_ops.append(tf.losses.mean_squared_error(self.target_var[i], output))
                elif self.loss_name[i] == 'mean_squared_norm':
                    loss_op += mean_squared_norm_loss(self.target_var[i], output, weights=self.loss_weight[i])
                elif self.loss_name[i] == 'mean_squared_log':
                    loss_op += mean_squared_log_loss(self.target_var[i], output, weights=self.loss_weight[i])

                elif self.loss_name[i] == 'crossentropy':
                    loss_op += tf.losses.sparse_softmax_cross_entropy(self.target_var[i], output, weights=self.loss_weight[i])
                elif self.loss_name[i] == 'linear':
                    loss_op += tf.losses.absolute_difference(self.target_var[i], output, weights=self.loss_weight[i])
                    #self.metrics_ops.append(tf.losses.absolute_difference(self.target_var[i], output))
                elif self.loss_name[i] == 'switch':
                    loss_op += switch_loss(self.target_var[i], output, weights=self.loss_weight[i], c1=self.switch_c1[i], c2=self.switch_c2[i], c3=self.switch_c3[i], switch_value=self.switch_value[i])
                    #self.metrics_ops.append(switch_loss(self.target_var[i], output, weights=self.loss_weight[i], c1=self.switch_c1[i], c2=self.switch_c2[i], c3=self.switch_c3[i], switch_value=self.switch_value[i]))
                elif self.loss_name[i] == 'log':
                    loss_op += tf.losses.log_loss(self.target_var[i], output, weights=self.loss_weight[i])
                else:
                    raise NotImplementedError
            regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularizers = map(tf.nn.l2_loss, regularizers)
            loss_op = loss_op + self.weight_decay*tf.add_n(regularizers)
        return loss_op

    def get_metrics(self):
        assert hasattr(self, 'target_var')
        with tf.name_scope('metrics'):
            metrics_ops = []
            for metric_name in self.metrics_names:
                metric_op = 0
                for i,output in enumerate(self.net.outputs):
                    if metric_name[i] == 'inv_huber':
                        metric_op += -1*tf.losses.huber_loss(self.target_var[i], output, delta=self.huber_delta[i], weights=self.loss_weight[i])
                    elif metric_name[i] == 'inv_linear':
                        metric_op += -1*tf.losses.absolute_difference(self.target_var[i], output,weights=self.loss_weight[i])
                    elif metric_name[i] == 'inv_switch':
                        metric_op += -1*switch_loss(self.target_var[i], output, c1=self.switch_c1[i], c2=self.switch_c2[i], c3=self.switch_c3[i], switch_value=self.switch_value[i], weights=self.loss_weight[i])
                    elif metric_name[i] == 'inv_mean_squared':
                        metric_op += -1*tf.losses.mean_squared_error(self.target_var[i], output, weights=self.loss_weight[i])
                    elif metric_name[i] == 'inv_mean_squared_norm':
                        metric_op += -1*mean_squared_norm_loss(self.target_var[i], output, weights=self.loss_weight[i])
                    elif metric_name[i] == 'inv_mean_squared_log':
                        metric_op += -1*mean_squared_log_loss(self.target_var[i], output, weights=self.loss_weight[i])
                    elif metric_name[i] == 'inv_log':
                        metric_op += -1*tf.losses.log(self.target_var[i], output, weights=self.loss_weight[i])
                    elif metric_name[i] == 'acc':
                        predictions = tf.cast(tf.argmax(output, axis=-1), tf.int32)
                        metric_op += tf.reduce_mean(tf.cast(tf.equal(self.target_var[i],predictions), tf.float32))
                    elif metric_name[i] == 'none':
                        pass
                    else:
                        raise NotImplementedError
                metrics_ops.append(metric_op)
        return metrics_ops

    def get_train_step(self):
        assert hasattr(self, 'loss_op')
        assert hasattr(self, 'global_step_var')
        with tf.name_scope('optimizer'):
            if self.lr_policy == 'step':
                lr_var = tf.train.exponential_decay(self.learning_rate, self.global_step_var,
                                                    self.stepsize, self.gamma, staircase=True)
            elif self.lr_policy == 'custom':
                boundaries = [s[0] for s in self.schedule]
                values = [self.learning_rate]+[s[1] for s in self.schedule]
                lr_var = tf.train.piecewise_constant(self.global_step_var, boundaries, values)
            else:
                lr_var = self.learning_rate
            if self.solver == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr_var,
                                                   beta1=self.momentum, beta2=self.momentum2)
            elif self.solver == 'sgd':
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr_var,
                                                       momentum=self.momentum, use_nesterov=True)
            elif self.solver == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_var,
                                                      decay=self.rms_decay, momentum=self.momentum)
            else:
                raise NotImplementedError
            if self.pmaps_learning_rate_factor is not None:
                pmaps_optimizer = tf.train.MomentumOptimizer(learning_rate=lr_var*self.pmaps_learning_rate_factor, momentum=self.momentum, use_nesterov=True)
                pmaps_vars = []
                remaining_vars = []
                for var in tf.trainable_variables():
                    if 'pmaps' in var.name:
                        pmaps_vars.append(var)
                    else:
                        remaining_vars.append(var)
                grads = tf.gradients(self.loss_op, pmaps_vars+remaining_vars, colocate_gradients_with_ops=True)
                grads_pmaps = grads[:len(pmaps_vars)]
                grads_remaining = grads[len(pmaps_vars):]
                train_op1 = pmaps_optimizer.apply_gradients(zip(grads_pmaps, pmaps_vars))
                train_op2 = optimizer.apply_gradients(zip(grads_remaining, remaining_vars), global_step=self.global_step_var)
                train_op = tf.group(train_op1, train_op2)
            else:
                train_op = optimizer.minimize(self.loss_op, global_step=self.global_step_var, colocate_gradients_with_ops=True)

        return train_op, lr_var

    def get_prediction(self):
        with tf.name_scope('prediction'):
            if self.mode == 'regression':
                return tf.concat(self.net.outputs, axis=-1)
            elif self.mode == 'segmentation':
                return tf.argmax(self.net.outputs[0], axis=-1)
            else:
                raise NotImplementedError

    def add_summary(self):
        # add summary for metrics
        with tf.name_scope('summaries'):
            self.eval_image_var = tf.placeholder(tf.uint8, self.eval_image_shape)
            self.eval_coords_image_var = tf.placeholder(tf.uint8, self.eval_coords_image_shape)
            self.test_loss_var = tf.placeholder(tf.float32)
            self.test_metrics_vars = [tf.placeholder(tf.float32) for _ in range(len(self.metrics_names))]

        train_summaries = []
        for name, var in zip(self.metrics_names, self.metrics_ops):
            name = '_'.join(name)
            train_summaries.append(tf.summary.scalar('global/'+name, var))
        train_summaries.append(tf.summary.scalar('global/loss', self.loss_op))
        train_summaries.append(tf.summary.scalar('global/lr', self.lr_var))
        # train summary of all summaries defined up to now
        self.train_summary_op = tf.summary.merge(train_summaries)
        # test summary of special (hacky) metrics
        test_summaries = []
        test_summaries.append(tf.summary.scalar('global/test_loss', self.test_loss_var))
        for name, var in zip(self.metrics_names, self.test_metrics_vars):
            name = '_'.join(name)
            test_summaries.append(tf.summary.scalar('global/test_'+name, var))
        test_summaries.append(tf.summary.image('global/test_eval', self.eval_image_var))
        if 'coord' in self.labels:
            test_summaries.append(tf.summary.image('global/test_eval_coords', self.eval_coords_image_var))
        # image activation summaries
        if self.show_activation is not False:
            for layer in self.net.layers:
                if hasattr(layer, 'activation_summary'):
                    #print(layer, layer.inbound_nodes)
                    test_summaries.append(tf.summary.image('activation/'+layer.name, tf.transpose(layer.get_output_at(0)[0:1], perm=[3,1,2,0])))
            if self.mode == 'segmentation':
                test_summaries.append(tf.summary.image('activation/output', tf.cast(tf.expand_dims(self.prediction_op, 3)[0:1], tf.uint8)))

            #for t in tf.get_collection('activation_summary'):
            #    print(t, t.device)
            #    print(t.name)
            #    test_summaries.append(tf.summary.image('activation/'+t.name, tf.transpose(t[0:1], perm=[3,1,2,0])))
        self.test_summary_op = tf.summary.merge(test_summaries)
        self.kernel_summary_op = None
        if len(tf.get_collection('kernel_summary')) > 0:
            self.kernel_summary_op = tf.summary.merge(tf.get_collection('kernel_summary'))

    def save_net(self, checkpoint=True, best=False):
        save_path = os.path.join(self.snapshot_dir,'iter_{:06d}'.format(self.global_step_var.eval()))
        if best:
            save_path += '_best_model'
        save_path += '.weights'
        self.log.info('Saving weights to %s'%save_path)
        self.net.save_weights(save_path)
        if checkpoint:
            save_path = save_path.replace('.weights', '.checkpoint')
            self.log.info('Saving checkpoint to %s'%save_path)
            self.checkpoint_saver.save(self.sess, save_path)

    def remove_saved_net(self, iteration, best=False):
        file_name = os.path.join(self.snapshot_dir, 'iter_{:06d}'.format(iteration))
        if best:
            file_name += '_best_model'
        file_name += '.weights'
        self.log.info('removing weights file %s'%file_name)
        if os.path.exists(file_name):
            os.remove(file_name)

    def execute_stop_criterium(self, metric, loss):
        """Finds the best model depending on stop_criterium.
        returns whether training should stop"""
        if self.stop_criterium == 'max_global':
            if metric > self.best_metric:
                self.save_net(best=True, checkpoint=False)
                # remove previous best net
                self.remove_saved_net(self.best_iteration, best=True)
                self.best_metric = metric
                self.best_iteration = self.global_step_var.eval()
        if np.isnan(loss):
            self.log.info('Stopping training due to nan loss')
            return True
        return False

    def create_feed_dict(self, batch, phase=1):
        feed_dict = {}
        for i in range(len(self.net.inputs)):
            feed_dict[self.net.inputs[i]] = batch[0][i]
            if self.mode == 'regression':
                feed_dict[self.target_var[0]] = np.asarray([b[:1] for b in batch[1]])
                if 'direction' in self.labels:
                    feed_dict[self.target_var[1]] = np.asarray(batch[1])[:,1:]
                if 'coord' in self.labels:
                    feed_dict[self.target_var[1]] = np.asarray([b[1][0]+b[1][1] for b in batch[1]])
            else:
                # segmentation
                feed_dict[self.target_var[0]] = np.asarray(batch[1])
            feed_dict[K.learning_phase()] = phase
        return feed_dict

    def test_net(self,writer=None):
        total_metrics = [0 for _ in range(len(self.metrics_names))]
        total_loss = 0
        predictions = None
        labels = None
        for test_i in range(0, self.test_iter):
            batch = self.test_batch_iter.next()
            if predictions is None:
                if self.mode == 'regression':
                    predictions = np.zeros(([0,]+self.prediction_op.shape.as_list()[1:]))
                else:  # segmentation
                    predictions = np.zeros(((0,)+batch[1].shape[1:]))
            if labels is None:
                if self.mode == 'regression':
                    labels = np.zeros(([0,]+self.prediction_op.shape.as_list()[1:]))
                else:  # segmentation
                    labels = np.zeros(((0,)+batch[1].shape[1:]))
            res = self.sess.run(self.metrics_ops + [self.loss_op, self.prediction_op], feed_dict=self.create_feed_dict(batch, phase=0))
            for i in range(len(total_metrics)):
                total_metrics[i]+=res[i]
            total_loss += res[len(total_metrics)]
            predictions = np.concatenate([predictions, res[len(total_metrics)+1]], axis=0)
            if self.mode == 'regression':
                labels = np.concatenate([labels, [flatten(b) for b in batch[1]]], axis=0)
            else:
                labels = np.concatenate([labels, batch[1]], axis=0)
        for i in range(len(total_metrics)):
            total_metrics[i] = total_metrics[i]*1./self.test_iter
        total_loss = total_loss*1./self.test_iter
        # add loss and metrics to tensorboard
        if self.mode == 'regression':
            img = distance_plot_for_tensorboard(labels[:,0].flatten(), predictions[:,0].flatten(), self.eval_image_shape)
            if 'coord' in self.labels:
                img_coords = coords_xyz_plot_for_tensorboard(np.concatenate([labels[:,1:4], labels[:,4:7]], axis=0), np.concatenate([predictions[:,1:4], predictions[:,4:7]], axis=0), self.eval_coords_image_shape)
        elif self.mode == 'segmentation':
            img = confusion_matrix_for_tensorboard(labels[:,0].flatten(), predictions[:,0].flatten(), self.labels, self.eval_image_shape)
        feed_dict = {var:value for var, value in zip(self.test_metrics_vars, total_metrics)}
        feed_dict.update({self.test_loss_var: total_loss, self.eval_image_var: img})
        if 'coord' in self.labels:
            feed_dict.update({self.eval_coords_image_var: img_coords})
        feed_dict[K.learning_phase()] = 0
        if self.show_activation is not False:
            # feed activation images four times to deal with multi-gpu (up to 4)
            # if net has multiple inputs, need to feed multiple activation images
            if not isinstance(self.show_activation, tuple):
                images = (self.show_activation, )
            else:
                images = self.show_activation
            for i, img in enumerate(images):
                feed_dict.update({self.net.inputs[i]: np.concatenate((img, img, img, img))})
        summary = self.sess.run(self.test_summary_op, feed_dict=feed_dict)
        writer.add_summary(summary, self.global_step_var.eval())
        del img
        return total_metrics, total_loss

    def train(self, resume_from=None):
        # for traces:
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # ...
        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('trace_%d.json'%self.global_step_var.eval(), 'w') as f:
        #    f.write(chrome_trace)

        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        with self.sess.as_default():
            # load or initialize weights
            self.sess.run(self.init_op)
            if resume_from is not None:
                self.log.info('Resuming training from checkpoint %s'%resume_from)
                self.checkpoint_saver.restore(self.sess, resume_from)
            elif self.finetune:
                self.log.info('Finetuning from %s'%self.finetune)
                self.net.load_weights(self.finetune)

            self.log.info('Starting training from iteration %d', self.global_step_var.eval())
            while self.global_step_var.eval() < self.iterations:
                run_options = None
                run_metadata = None
                operations = [self.train_op, self.train_update_op, self.loss_op] + self.metrics_ops
                i = self.global_step_var.eval()+1
                if i%self.log_interval == 0 or i == self.iterations:
                    operations += [self.train_summary_op]
                start = time.time()
                batch = self.train_batch_iter.next()
                mid = time.time()
                res = self.sess.run(operations, feed_dict=self.create_feed_dict(batch, phase=1), options=run_options, run_metadata=run_metadata)
                end = time.time()
                self.log.info('Batch generation took {:.2}s, Forward-Backward pass took {:.2}s'.format(mid-start, end-mid))
                i=self.global_step_var.eval()
                if i%self.display == 0:
                    self.log.info('Iteration %d: loss %f'%(i,res[2]))
                    for metric, name in zip(res[3:], self.metrics_names):
                        name = '_'.join(name)
                        self.log.info('Iteration %d: %s %f'%(i,name,metric))
                if i%self.log_interval == 0 or i == self.iterations:
                    # add summary of current batch
                    writer.add_summary(res[-1], self.global_step_var.eval())
                if i%self.test_interval == 0 or i == self.iterations:
                    self.log.info('Testing net')
                    metrics, loss = self.test_net(writer)
                    self.log.info('Iteration %d: test_loss %f'%(i,loss))
                    for metric, name in zip(metrics, self.metrics_names):
                        name = '_'.join(name)
                        self.log.info('Iteration %d: test_%s %f'%(i,name,metric))
                    if self.execute_stop_criterium(metrics[0], loss):
                        break
                if (i % self.snapshot == 0 and i > 0) or i == self.iterations:
                    # add kernel summaries (dont want to do this too often - logsize!)
                    if self.kernel_summary_op is not None:
                        summary = self.sess.run(self.kernel_summary_op, feed_dict={K.learning_phase():0})
                        writer.add_summary(summary, self.global_step_var.eval())

                    self.save_net()
        self.sess.close()

