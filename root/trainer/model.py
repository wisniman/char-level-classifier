# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from keras import backend as K
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from keras.initializers import TruncatedNormal
from keras.optimizers import SGD

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


# csv columns in the input file
# LABEL_COLUMN = 0
# DATA_COLUMNS = [1, 2]


def model_fn(maxlen, vocab_size, conv_filters, conv_kernels,
             max_pooling, dense_output_units, dropout_probs,
             output_cats, learning_rate, momentum,
             stddev, mean=0.0, seed=42,
             conv_padding='valid', conv_activation='relu',
             dense_activation='relu', output_activation='softmax'):
    """Create a Keras Sequential model with layers."""
    inputs = Input(shape=(maxlen, vocab_size),
                   name='input',
                   dtype='float32')

    conv1 = Conv1D(filters=conv_filters,
                   kernel_size=conv_kernels[0],
                   padding=conv_padding,
                   kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                   activation=conv_activation,
                   input_shape=(maxlen, vocab_size))(inputs)

    pool1 = MaxPooling1D(pool_size=max_pooling[0])(conv1)

    conv2 = Conv1D(filters=conv_filters,  # 1024 or 256
                   kernel_size=conv_kernels[1],  # 7 or 3
                   padding=conv_padding,
                   kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                   activation=conv_activation)(pool1)

    pool2 = MaxPooling1D(pool_size=max_pooling[1])(conv2)

    conv3 = Conv1D(filters=conv_filters,  # 1024 or 256
                   kernel_size=conv_kernels[2],  # 7 or 3
                   padding=conv_padding,
                   kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                   activation=conv_activation)(pool2)

    conv4 = Conv1D(filters=conv_filters,  # 1024 or 256
                   kernel_size=conv_kernels[3],  # 7 or 3
                   padding=conv_padding,
                   kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                   activation=conv_activation)(conv3)

    conv5 = Conv1D(filters=conv_filters,  # 1024 or 256
                   kernel_size=conv_kernels[4],  # 7 or 3
                   padding=conv_padding,
                   kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                   activation=conv_activation)(conv4)

    conv6 = Conv1D(filters=conv_filters,  # 1024 or 256
                   kernel_size=conv_kernels[5],  # 7 or 3
                   padding=conv_padding,
                   kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                   activation=conv_activation)(conv5)

    pool3 = MaxPooling1D(pool_size=max_pooling[5])(conv6)

    flat = Flatten()(pool3)

    fc1 = Dense(units=dense_output_units,
                kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                activation=dense_activation)(flat)

    dropout1 = Dropout(dropout_probs[0])(fc1)

    fc2 = Dense(units=dense_output_units,
                kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                activation=dense_activation)(dropout1)

    dropout2 = Dropout(dropout_probs[1])(fc2)

    outputs = Dense(units=output_cats,
                    kernel_initializer=TruncatedNormal(mean=mean, stddev=stddev, seed=seed),
                    activation=output_activation,
                    name='output')(dropout2)

    model = Model(inputs=inputs, outputs=outputs)

    # Add a dense final layer with sigmoid function
    # compile_model(model, learning_rate, momentum)
    return model


def compile_model(model, learning_rate, momentum):
    """compile model with optimizer"""

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=learning_rate,
                                momentum=momentum,
                                #   decay=0.0,
                                #   nesterov=False
                                ),
                  metrics=['categorical_accuracy'])
    return model


def to_savedmodel(model, export_path):
    """Convert the Keras hdf5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()
