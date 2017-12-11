# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import glob
import numpy as np
import warnings

from keras.callbacks import Callback
from keras.models import load_model
from tensorflow.python.lib.io import file_io
import trainer.model as model


class ContinuousEval(Callback):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self,
                 eval_frequency,
                 eval_sequence,
                 # eval_generator,
                 learning_rate,
                 momentum,
                 job_dir,
                 steps):
        # self.eval_files = eval_files
        self.eval_frequency = eval_frequency
        self.eval_sequence = eval_sequence
        # self.eval_generator = eval_generator
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.job_dir = job_dir
        self.steps = steps

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0 and epoch % int(self.eval_frequency) == 0:

            # workaround bc h5py cannot write to GCS
            # save to local filesystem, then copy over to GCS
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith("gs://"):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                # select latest model checkpoint
                conv_model = load_model(checkpoints[-1])
                conv_model = model.compile_model(conv_model, self.learning_rate, self.momentum)
                loss, acc = conv_model.evaluate_generator(
                    # generator=self.eval_generator,
                    generator=self.eval_sequence,
                    steps=self.steps,
                    # max_queue_size=10,
                    # workers=1,
                    # use_multiprocessing=False
                )
                print('Evaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
                    epoch, loss, acc, conv_model.metrics_names))
                if self.job_dir.startswith("gs://"):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print('Evaluation epoch[{}] (no checkpoints found)'.format(epoch))


class ModelCheckpointDetached(Callback):
    """ Save detached from multi-GPU encapsulation model
    (very small) modification from https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L331

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpointDetached, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % mode, RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            detachmodel(self.model).save_weights(filepath, overwrite=True)
                        else:
                            detachmodel(self.model).save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    detachmodel(self.model).save_weights(filepath, overwrite=True)
                else:
                    detachmodel(self.model).save(filepath, overwrite=True)


def detachmodel(m):
    """ Detach model trained on GPUs from its encapsulation
    # Arguments
        :param m: obj, keras model
    # Returns
        :return: obj, keras model
    """
    for l in m.layers:
        if l.name == 'model_1':
            return l
    return m


def copy_file_to_gcs(job_dir, file_path):
    """ h5py workaround: copy local models over to GCS"""
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())
