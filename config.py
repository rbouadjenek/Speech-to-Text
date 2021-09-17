model = {
    #input and output dim
    'input_dim': 161,
    "out_dim": 28,

    #Convolutional layer's parameter
    'conv_channels': [256, 256],
    'conv_filters': [5, 5],
    'conv_strides': [2, 2],

    #Rnn layer's parameter
    'rnn_units': [64],
    'bidirectional': True,

    'future_context': 2,

    'BatchNormalization': True,

    'learning_rate': 0.001

}
training = {

    'tensorboard': False,
    'log_dir': './logs',

    'batch_size': 64,
    'epochs': 5,
    'validation_size': 0.2

}

preprocessing = {

    'data_trainning_dir': './train_data/',
    'data_val_dir': './dev_data/',
    'window_size': 20,
    'step_size': 10,
    "max_spec_length": None,
    "max_label_length": None,
  
    "step_per_epoch": int(28539//training["batch_size"])
}

