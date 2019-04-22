import keras as kr
import numpy as np
import tensorflow as tf
import os
from keras import backend as K


class File_Exists_Error(Exception):
    """Basic exception for preventing costly retraining"""
    def __init__(self, file_name, msg=None):
        if msg is None:
            # Set some default useful error message
            msg = "{0} already exists,".format(file_name)\
                  + "are you sure you want to overwrite it?"
        super(File_Exists_Error, self).__init__(msg)



def zca(x_1, x_2, epsilon=1e-5):
    """ This function applies ZCA Whitening to the image set

    Arguments
    ---------
    x_1 : numpy.ndarray or array like
      Array of MxNxC images to compute the ZCA Whitening
    x_2 :  numpy.ndarray or array like
      Array of MxNxC images to apply the ZCA transform
    num_batch :  numpy.ndarray or array like
      Number of batches to do the computation

   Returns
   -------
       An array of MxNxC zca whitened images
    """
    with tf.name_scope('ZCA'):
        x1 = tf.placeholder(
            tf.float64, shape=np.shape(x_1), name='placeholder_x1')
        x2 = tf.placeholder(
            tf.float64, shape=np.shape(x_2), name='placeholder_x2')
        flatx = tf.cast(
            tf.reshape(x1, (-1, np.prod(x_1.shape[-3:])), name="reshape_flat"),
            tf.float64, name="flatx")
        sigma = tf.tensordot(
            tf.transpose(flatx), flatx, 1, name="sigma") / tf.cast(
                tf.shape(flatx)[0], tf.float64)  # N-1 or N?
        s, u, v = tf.svd(sigma, name="svd")
        pc = tf.tensordot(tf.tensordot(u, tf.diag(
            1. / tf.sqrt(s+epsilon)), 1, name="inner_dot"),
                          tf.transpose(u), 1, name="pc")

        net1 = tf.tensordot(flatx, pc, 1, name="whiten1")
        net1 = tf.reshape(net1, np.shape(x_1), name="output1")

        flatx2 = tf.cast(tf.reshape(x2, (-1, np.prod(x_2.shape[-3:])),
                                    name="reshape_flat2"),
                         tf.float64, name="flatx2")
        net2 = tf.tensordot(flatx2, pc, 1, name="whiten2")
        net2 = tf.reshape(net2, np.shape(x_2), name="output2")

    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x_1, x_2 = sess.run([net1, net2], feed_dict={x1: x_1, x2: x_2})
    return x_1, x_2


def resize(X, num_batch=1, new_dim=(128, 128)):
    """This function resizes 32x32x3 images to 128x128x3 by adding padding

    Arguments
    ---------
    X : numpy.ndarray
        Array of W x H x 3 Images, where W and H are any real int
    num_batch : numpy.ndarray
      Number of batches to do the computation
    new_dim : tuple
      New dimensions of image array
    Returns
    -------
       A new_dim[0] x new_dim[1] x 3 set of images
    """
    _length = len(X)
    d = int(_length/num_batch)
    new_w, new_h = new_dim
    X_new = np.zeros((_length, new_w, new_h, 3)).astype(np.int8)
    ind = 0
    for i in range(num_batch):
        end = ind+d
        if i == num_batch-1:
            end = _length
        x_batch = X[ind:end]
        net = tf.image.resize_images(
            x_batch, size=([np.int(dim * 1.15) for dim in new_dim]))
        net = tf.random_crop(net, ((end-ind), new_w, new_h, 3))
        with tf.Session() as sess:
                X_new[ind:end] = sess.run(net)
        ind = ind + d
    return X_new


"""https://github.com/sameermanek/keras/commit/
7ceefaacd664f5065a06d76e80e7fccadf737db5
#diff-ca6be3eb60c04bc5fcd73355b7f99d0dL651"""

"""This class is copied from the above link.
 It is a very good idea, and lets tensorboard assume a very simple function
 of graphing the accuracy and loss at every step. If I could change it to """


class TensorBoard(kr.callbacks.Callback):
    """Tensorboard basic visualizations.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    TensorBoard is a visualization tool provided with TensorFlow.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        write_batch_performance: whether to write training metrics on batch 
            completion 
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 write_batch_performance=True,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.write_batch_performance = write_batch_performance
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.seen = 0

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)
                        tf.summary.histogram('{}_grad'.format(weight.name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            self.saver = tf.train.Saver(list(embeddings.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                     'keras_embedding.ckpt')

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, self.seen)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.seen)
        self.writer.flush()
        self.seen += self.batch_size

    def on_train_end(self, _):
        self.writer.close()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance is True:
            for name, value in logs.items():
                if name in ['step', 'size']:  # Had batch instead of step
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                # if name in  ['step', 'size']:
                summary_value.simple_value = value
                # else:
                #     summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.seen)
            self.writer.flush()

        self.seen += self.batch_size
