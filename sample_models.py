from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    # Assuming that recur_layers >=1
    rec_out = GRU(units = units, return_sequences = True, activation = 'relu', 
                  name = 'GRU'+str(0))(input_data)
    rec_out = BatchNormalization()(rec_out)
    for itr in range(recur_layers-1):
        rec_out = GRU(units = units, return_sequences = True, activation = 'relu', 
                      name = 'GRU'+str(itr+1))(rec_out)
        rec_out = BatchNormalization()(rec_out)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units = output_dim))(rec_out)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units = units, activation = 'relu', return_sequences = True), 
                              merge_mode = 'concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units = output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# we will also try layer_normalization
from keras_layer_normalization import LayerNormalization

def final_model(input_dim, cnn_units = 200, kernel_size = 5, padding = 'same', dilation_rate = 2, 
                num_rnn_layers = 2, rnn_units = 200, merge_mode= 'concat', dense_units = 100,  
                output_dim = 29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    cnn1 = Conv1D(filters = cnn_units, kernel_size = kernel_size, 
                 padding = padding, activation = 'relu', name = 'cnn1' )(input_data)
    batch1 = LayerNormalization(name= 'LNorm1')(cnn1)
    cnn2 = Conv1D(filters = cnn_units, kernel_size = kernel_size, name = 'cnn2',
                  padding = padding, activation = 'relu', dilation_rate = dilation_rate)(batch1)
    batch2 = LayerNormalization(name = 'LNorm2')(cnn2)
    rnn_out = Bidirectional(GRU(units = rnn_units, return_sequences = True, 
                                activation = 'relu' ), name = 'rnn0', 
                            merge_mode = merge_mode)(batch2)
    rnn_out = LayerNormalization(name = 'LNorm3')(rnn_out)
    for itr in range(num_rnn_layers-1):
        rnn_out = Bidirectional(GRU(units = rnn_units, return_sequences = True, 
                                    activation = 'relu'), name = 'rnn'+str(itr+1), 
                                merge_mode = merge_mode)(rnn_out)
        rnn_out = LayerNormalization(name = 'LNorm'+str(4+itr))(rnn_out)
    dense1 = TimeDistributed(Dense(units = dense_units), name = 'dense1')(rnn_out)
    batch_dense = LayerNormalization(name = 'LNorm_dense')(dense1) 
    dense2 = TimeDistributed(Dense(units = output_dim), name = 'dense2')(batch_dense)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name = 'softmax')(dense2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: final_model_output_length(x, kernel_size, padding, dilation_rate)
    print(model.summary())
    return model

def final_model_output_length(input_length, kernel_size, padding, dilation_rate):
    cnn1_length = cnn_output_length(input_length, kernel_size, padding, stride=1,
                       dilation=1)
    cnn2_length = cnn_output_length(cnn1_length, kernel_size, padding, stride=1,
                                    dilation=dilation_rate)
    return cnn2_length
    