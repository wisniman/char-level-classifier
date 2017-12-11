# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import math
import random
import time

import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.optimizers import SGD
from keras.utils import multi_gpu_model

from trainer.encoders import Encoder
from trainer.generators import DataSequence
from trainer.alphabets import English, German
from trainer.callbacks import copy_file_to_gcs, ModelCheckpointDetached, ContinuousEval
import trainer.model as model

"""
Parameters for dataset and ConvNet:
"""
# Dataset parameters:
LABEL_COLUMN = 0
DATA_COLUMNS = [1]
DATASET_TRAIN_SIZE = 187120
DATASET_TEST_SIZE = 20745
NUM_CLASSES = 5

# Encoder parameters:
# ALPHABET = English(lowercase=True).get_alphabet()
ALPHABET = German(lowercase=True).get_alphabet()
CLEAR_ACCENTS = True
REVERSE_ENCODING = True
MAXLEN = 1014

# Generator parameters:
BATCH_SIZE = 128
SHUFFLE = True

# Model parameters:
CONV_FILTERS_SMALL = 256
CONV_FILTERS_LARGE = 1024
CONV_KERNELS = [7, 7, 3, 3, 3, 3]
MAX_POOLING = [3, 3, None, None, None, 3]
DENSE_OUTPUT_UNITS_SMALL = 1024
DENSE_OUTPUT_UNITS_LARGE = 2048
DROPOUT_PROBS = [0.5, 0.5]
STDDEV_SMALL = 0.02
STDDEV_LARGE = 0.05

# Optimizer parameters:
LEARNING_RATE = 0.01  # optimizer learning rate
MOMENTUM = 0.9  # optimizer momentum

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = None  # 5000  # MUST BE MULTIPLE OF BATCH SIZE!!!
FILE_PATH = 'checkpoint.epoch_{epoch:02d}.hdf5'
CONV_MODEL = 'charlevel_conv.hdf5'


def dispatch(multi_gpu,
             train_files,
             eval_files,
             job_dir,
             train_steps,
             train_batch_size,
             num_epochs,
             learning_rate,
             stddev,
             eval_steps,
             eval_batch_size,
             eval_num_epochs,
             eval_frequency,
             checkpoint_epochs,
             gpus,
             workers,
             verbose):
    """
    Main training method:
    """

    # random seed
    random.seed(42)

    # load encoder:
    encoder = Encoder(alphabet=ALPHABET,
                      maxlen=MAXLEN,
                      num_classes=NUM_CLASSES,
                      clear_accents=CLEAR_ACCENTS)

    # prepare data generator sequences:
    train_sequence = DataSequence(input_file=train_files,
                                  label_column=LABEL_COLUMN,
                                  data_columns=DATA_COLUMNS,
                                  encoder=encoder,
                                  backwards=REVERSE_ENCODING,
                                  batch_size=train_batch_size,
                                  # workaround bc sequence.__len__ overwrites fit_generator arg
                                  steps_per_epoch=train_steps,
                                  shuffle=SHUFFLE)

    eval_sequence = DataSequence(input_file=eval_files,
                                 label_column=LABEL_COLUMN,
                                 data_columns=DATA_COLUMNS,
                                 encoder=encoder,
                                 backwards=REVERSE_ENCODING,
                                 batch_size=eval_batch_size,
                                 # workaround bc sequence.__len__ overwrites fit_generator arg
                                 steps_per_epoch=eval_steps,
                                 shuffle=SHUFFLE)

    # prepare log dictionaries
    job_dir += '/' + time.strftime("%Y%m%d-%H%M%S")
    try:
        os.makedirs(job_dir)
    except:
        print("ERROR: Directory 'job-dir' could not be created.")

    # workaround bc h5py cannot write to GCS
    # save to local filesystem, then copy over to GCS
    checkpoint_path = FILE_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

    # Learning rate scheduler callback --unused for the moment
    cb_learning_rate_scheduler = LearningRateScheduler(learning_rate_scheduler)

    # Detached model checkpoint callback to snapshot multi-gpu models
    detached_checkpoint = ModelCheckpointDetached(checkpoint_path,
                                                  monitor='acc',
                                                  verbose=1,
                                                  period=checkpoint_epochs,
                                                  mode='max')

    # Continuous eval callback, eval & copy checkpoints to gcs
    evaluation = ContinuousEval(eval_frequency=eval_frequency,
                                eval_sequence=eval_sequence,
                                # eval_generator=eval_generator,
                                learning_rate=learning_rate,
                                momentum=MOMENTUM,
                                job_dir=job_dir,
                                steps=eval_steps
                                )

    # Tensorboard logs callback
    tblog = TensorBoard(
        log_dir=os.path.join(job_dir, 'tb-logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [
        # cb_learning_rate_scheduler,
        detached_checkpoint,
        evaluation,
        tblog,
    ]

    # load model:
    with tf.device('/cpu:0'):
        conv_model = model.model_fn(maxlen=MAXLEN,
                                    vocab_size=encoder.vocab_size,
                                    conv_filters=CONV_FILTERS_SMALL,
                                    conv_kernels=CONV_KERNELS,
                                    # conv_padding=conv_padding,
                                    # conv_activation=conv_activation,
                                    max_pooling=MAX_POOLING,
                                    dense_output_units=DENSE_OUTPUT_UNITS_SMALL,
                                    # dense_activation=dense_activation,
                                    dropout_probs=DROPOUT_PROBS,
                                    output_cats=NUM_CLASSES,
                                    # output_activation=output_activation,
                                    # optimizer=optimizer,
                                    learning_rate=learning_rate,
                                    momentum=MOMENTUM,
                                    stddev=stddev,
                                    # loss=loss,
                                    # metrics=metrics
                                    )

    if multi_gpu:
        # Replicate the model on multiple GPUs:
        parallel_model = multi_gpu_model(conv_model, gpus=gpus)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer=SGD(lr=learning_rate, momentum=MOMENTUM),
                               metrics=['categorical_accuracy'])

    with tf.device('/cpu:0'):
        # compile local model
        conv_model.compile(loss='categorical_crossentropy',
                           optimizer=SGD(lr=learning_rate, momentum=MOMENTUM),
                           metrics=['categorical_accuracy'])
        conv_model.summary()

    if multi_gpu:
        parallel_model.fit_generator(callbacks=callbacks,
                                     # generator=train_generator,
                                     generator=train_sequence,
                                     steps_per_epoch=train_steps,
                                     epochs=num_epochs,
                                     workers=workers,
                                     # verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch
                                     verbose=verbose)
        conv_model.set_weights(parallel_model.get_weights())
    else:
        conv_model.fit_generator(callbacks=callbacks,
                                 generator=train_sequence,
                                 steps_per_epoch=train_steps,
                                 epochs=num_epochs,
                                 workers=workers,
                                 verbose=verbose)

    # workaround bc h5py cannot write to GCS
    # save to local filesystem, then copy over to GCS
    if job_dir.startswith("gs://"):
        conv_model.save(CONV_MODEL)
        copy_file_to_gcs(job_dir, CONV_MODEL)
    else:
        conv_model.save(os.path.join(job_dir, CONV_MODEL))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(conv_model, os.path.join(job_dir, 'export'))


def learning_rate_scheduler(epoch_index):
    """Schedule decay in learning rate --unused"""

    lr = 0.001
    # halfed every 3 epochs for 10 times:
    n = epoch_index // 3
    if n > 0 and n < 10:
        lr = lr / math.pow(2, n)
    else:
        lr = lr / math.pow(2, 10)
    return lr


def str2bool(v):
    """read argument input for multi-gpu and interpret as bool"""

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-gpu',
                        required=True,
                        type=str2bool,
                        help='Run Keras on multiple GPUs if available')
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS', nargs='+')
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--train-steps',
                        type=int,
                        default=math.ceil(DATASET_TRAIN_SIZE / BATCH_SIZE),
                        help='Maximum number of training steps to perform')
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=BATCH_SIZE,
                        help='Batch size for training steps')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=LEARNING_RATE,
                        help='Learning rate for SGD')
    parser.add_argument('--stddev',
                        type=float,
                        default=STDDEV_SMALL,
                        help='Standard deviation for weights initilization')
    parser.add_argument('--eval-steps',
                        type=int,
                        default=math.ceil(DATASET_TEST_SIZE / BATCH_SIZE),
                        help='Number of steps to run evalution for at each checkpoint')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=BATCH_SIZE,
                        help='Batch size for evaluation steps')
    parser.add_argument('--eval-num-epochs',
                        type=int,
                        default=1,
                        help='Number of epochs during evaluation')
    parser.add_argument('--eval-frequency',
                        default=10,
                        help='Perform one evaluation per n epochs')
    parser.add_argument('--checkpoint-epochs',
                        type=int,
                        default=5,
                        help='Checkpoint per n training epochs')
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='Number of GPUs')
    parser.add_argument('--workers',
                        type=int,
                        default=2,
                        help='Number of processes to spin up when fitting on generator')
    parser.add_argument('--verbose',
                        type=int,
                        default=2,
                        help='Verbosity: 0 = silent, 1 = progress bar, 2 = one line per epoch')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
