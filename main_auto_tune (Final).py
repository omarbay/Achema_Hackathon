import cv2
import os

# This option allows you to run keras/tf on
# the CPU. Useful if you have a weak GPU and
# a large number os CPU avaliable
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import kerastuner as kt

def build_dense(model, activation="relu", n_neurons=32, dropout=0.1, with_batch=False):
    """ This function builds the last fully connected layers of the CNN,
        returning a "dense-block". It is parametrized and it is possible
        to define.
        Arguments:
        activation --  the activation function ["relu"(default), "leaky_relu"]
        n_neurons  --  number of neurons on the FC layers  (default 32)
        dropout    --  dropout rate (defaut 0.1)
        with_batch --  adds batch normalization layer at the end of returned block
    """

    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_neurons))
    if activation == "relu":
        model.add(layers.ReLU())
    elif activation == "leaky_relu":
        model.add(layers.LeakyReLU())
    if with_batch:
        model.add(layers.BatchNormalization())

    return model


def build_conv(model, n_filters=4, kernel_size=3, pool_size=2, activation="relu", with_batch=False):
    """ This function builds the convolution blocks of the CNN,
        returning a "conv2d-block" (conv2d->maxpool2d->batch_norm).
        It is parametrized and it is possible to define:
        Arguments:
        n_filters   --  number of output filters in the convolution (default 4)
        kernel_size --  a single integer that define the spatial dimension
                        of an squared filter, kernel_size x kernel_size (default 3)
        pool_size   --  a single integer that defines the window size during
                        the maxpool2d block. No stride is defined and the padding
                        is "valid". (default 2)
        activation --  the activation function of the conv2d block ["relu"(default), "leaky_relu"]
        with_batch --  adds batch normalization layer at the end of returned block
    """

    model.add(layers.Conv2D(filters=n_filters, kernel_size=kernel_size))
    if activation == "relu":
        model.add(layers.ReLU())
    elif activation == "leaky_relu":
        model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    if with_batch:
        model.add(layers.BatchNormalization())
    return model

def model_builder(hp):
    """ This function builds the CNN model. This approach is useful during the
        hyperpameter study, since it is capable of producing CNN architectures
        only by passing parameters
        Arguments:
        hp -- a HyperModel instance of the Keras-Tuner API
              (see tensorflow.org/tutorials/keras/keras_tuner)
    """

    # Initializing the model through as a Sequential model
    model = keras.Sequential()

    # Scales the 8-bit unsigned integer images from 0-255
    # to 0.0-1.0 float values
    model.add(layers.experimental.preprocessing.Rescaling(1./255))

    # Defining the limits, step an default values that the hp
    # kerastuner instance may explore during the hyperpameter analysis

    # The hyperpameter analys may explore the use of batch_norm throught the CNN
    batch_option = hp.Choice('batch_norm',[True,False])

    # Since the conv2d block is parametrized, the hp instance may explore the number and
    # parameters of the convolution block

    # Number of conv2d blocks that the hp instance may explore
    N_CONV_LAYERS = hp.Int('n_conv_layers', min_value=1, max_value=2, default=2, step=1)

    # The parameters of each conv2d block is defined though the hp instace in
    # this portion of the function. Observe that the model options are stored on a
    # nested list format.
    # Example: If N_CONV_LAYERS=2 (two conv2d blocks in the CNN)
    #         conv_layers =
    #         [
    #           [64, 3, 2, 'relu'],      --- 1st conv2d layer prop
    #           [12, 5, 1, 'leakyrelu'], --- 2nd conv2d layer prop
    #         ]
    conv_layers = []
    for k in range(N_CONV_LAYERS):
        param_filter     = hp.Int('n_filter_%d' % k, min_value=4, max_value=64, default=8, step=16)
        param_kernel     = hp.Int('n_kernel_%d' % k, min_value=3, max_value=5, default=3, step=2)
        param_pool       = hp.Int('pool_%d' % k, min_value=1, max_value=2, default=1, step=1)
        param_activation = hp.Choice('conv_activ_%d' % k ,["relu","leaky_relu"])
        conv_layers.append((param_filter, param_kernel, param_pool, param_activation))

    # Here, the conv_layers nested list is being looped. During this loop,
    # the conv2d layers are being created and added to the model
    for k in range(N_CONV_LAYERS):
        conv_layer_param = conv_layers[k]
        model = build_conv(model,
                           n_filters   = conv_layer_param[0],
                           kernel_size = conv_layer_param[1],
                           pool_size   = conv_layer_param[2],
                           activation  = conv_layer_param[3],
                           with_batch  = batch_option
                           )

    # Now, the last layer is "flattened" for the following Fully-Connected (FC) layers
    model.add(layers.Flatten())

    # Number of FC  blocks that the hp instance may explore
    N_DENSE_LAYERS = hp.Int('n_dense_layers', min_value=1, max_value=4, default=2, step=1)

    # The FC-layers follow the same nested list methodology employed for the conv-2d block
    # creation
    # Example: If N_DENSE_LAYERS=2 (two FC-blocks in the CNN)
    #         conv_layers =
    #         [
    #           ['relu', 64, 0,1],      --- 1st FC layer prop
    #           ['relu', 64, 0,1],      --- 2nd FC layer prop
    #         ]
    dense_layers = []
    for k in range(N_DENSE_LAYERS):
        param_n_neurons  = hp.Int('n_neurons_%d' % k, min_value=32, max_value=256, default=64, step=32)
        param_dropout    = hp.Float('dropout_%d' % k, min_value=0.0, max_value=0.5, default=0.1, step=0.1)
        param_activation = hp.Choice('dense_activ_%d' % k , ["relu","leaky_relu"])
        dense_layers.append((param_activation, param_n_neurons, param_dropout))

    # Here, the dense_layers nested list is being looped. During this loop,
    # the FC layers are being created and added to the model
    for k in range(N_DENSE_LAYERS):
        dense_layer_param = dense_layers[k]
        model = build_dense(model,
                           activation  = dense_layer_param[0],
                           n_neurons   = dense_layer_param[1],
                           dropout     = dense_layer_param[2],
                           with_batch  = batch_option
                           )

    # The final activataion layer. For this binary classification, we
    # opted for a sigmoid layer [0.0 to 1.0], where values < 0.5 represents
    # class 1 and values >= 0.5 class 2
    model.add(layers.Dense(1, activation="sigmoid"))

    # Model compilation using the binary-crossentropy loss and the ADAM
    # SGD optimization based on the adaptive estimando of the 1st and 2nd
    # order moments
    model.compile(
            optimizer='adam',
            loss="binary_crossentropy",
            metrics=['accuracy'])

    return model

# Batch sizes used during CNN training
batch_size = 16

# Vertical and Horizontal trim from the original dataset
# While the training the CNN, we noticed that the camera
# positioning affected the acc. results. Hence, we decide
# to remove the background from the original images
vertical_trim = 50
horizonta_trim = 80

# Image dimensions used wilhe training the CNNs
# In order to reduced the number of parameter on the
# CNN, the image must be reduced. The idea is to find
# an optimal point between acc and inference time
img_height = 40
img_width = img_height // 2

# As we reduce the size of the images, we noted that
# some Fluten example with large bubbles were missclas-
# sifed. Analyzing the results, we observed that when
# the images were rescaled, the small gas-liquid inter-
# faces (pixel values between 0 and 20) lost their reso-
# lution. When the image went to a max-pooling layer, 
# this information completely was lost. Thus, in order to 
# keep this information, we inverted the image 
# (cv2.bitwise_not function). Then, this information would
# not be lost in the max-pooling layer 
img_inv = True

# If not in the train mode, keras-tuner mode will take place
# and the tool will try to find the best hyperparameters
# that minimizes the validation loss, i.e. increase the model
# accuraccy. In this mode, only a part of the dataset is used
# When training mode is on, the CNN will train the images with
# a larger dataset. The script will train the model only with the
# top 5 CNN architectures from the keras-tuner step
train_mode = True


# Image file format:
img_file_format = "png"


# Definition of the dataset folder
dataset_dir = ""

# Defining the Training folders and a string to differentiate
# between the training and the keras-tuner runs
if train_mode:
    train_string="train"
    training_dir = os.path.join(dataset_dir,"Training")
else:
    train_string="kt"
    training_dir = os.path.join(dataset_dir,"Training_reduced")
fluten_training_dir = os.path.join(training_dir,"Fluten")
normal_training_dir = os.path.join(training_dir,"Normalzustand")

# The dataset can be loaded multiple times during a train run, and
# the loading/processing step may take some time if the dataset is
# large. Hence, once the dataset is lodaded, the images are stored
# in a .npz file, which loads fast and contains processed images
try:
    dataset = np.load("dataset_%02d_%02d_%s_%s.npz" % (img_width, img_height, img_inv, train_string))
    creata_data = False
    print("Loading dataset")
except FileNotFoundError:
    creata_data = True
    print("Creating dataset....")


# Creating the data is a bit cumbersome, since we are removing
# lateral portion of the images an applying the cv2.bitwise 
# function on the original images, the keras.preprocessing tools
# do not handle the task. Therefore, it was necessary to create
# a cv2-based-tool for creating the train/val/test dataset
if creata_data:

    ############### Loading Fluten images ###############
    # Finding the jpg images on the folder
    fluten_training_files = os.listdir(fluten_training_dir)
    fluten_img_files = []
    for _file in fluten_training_files:
        if img_file_format in _file:
            fluten_img_files.append(_file)

    # Creating the image and label np arrays
    N_fluten_img_files = len(fluten_img_files)
    fluten_training_imgs   = np.zeros((N_fluten_img_files, img_height, img_width,1), dtype=float)
    fluten_training_labels = np.zeros((N_fluten_img_files), dtype=float)

    # Loading the images from the folder, applying the img-processing steps
    # You may uncoment the marked lines to follow the image process. steps

    # cv2.namedWindow("debug", cv2.WINDOW_NORMAL) ### uncomment for debug
    for i, img_file in enumerate(fluten_img_files):
        if (i % 50 == 0):
            print("Loaded %04d/%04d images from the fluten dataset" % (i, N_fluten_img_files))
        img = cv2.imread(os.path.join(fluten_training_dir, img_file),0)
        # cv2.imshow("debug", img); cv2.waitKey(0) ### uncomment for debug
        img_trim = img[vertical_trim:-vertical_trim,horizonta_trim:-horizonta_trim]
        # cv2.imshow("debug", img_trim); cv2.waitKey(0) ### uncomment for debug
        if img_inv:
            img_bit = cv2.bitwise_not(img_trim)
            # cv2.imshow("debug", img_bit); cv2.waitKey(0) ### uncomment for debug
            img_resized = cv2.resize(img_bit, (img_width, img_height))
        else:
            img_resized = cv2.resize(img_trim, (img_width, img_height))
            # cv2.imshow("debug", img_trim); cv2.waitKey(0) ### uncomment for debug
        # cv2.imshow("debug", img_resized); cv2.waitKey(0) ### uncomment for debug
        img_keras  = img_resized.astype(float)
        fluten_training_imgs[i,:,:,0] = img_keras
        fluten_training_labels[i] = 1.0

    ############### Loading Normal images ###############
    # Finding the jpg images on the folder
    normal_training_files = os.listdir(normal_training_dir)
    normal_img_files = []
    for _file in normal_training_files:
        if img_file_format in _file:
            normal_img_files.append(_file)

    # Creating the image and label np arrays
    N_normal_img_files = len(normal_img_files)
    normal_training_imgs   = np.zeros((N_normal_img_files, img_height, img_width,1), dtype=float)
    normal_training_labels = np.zeros((N_normal_img_files), dtype=float)

    # Loading the images from the folder, applying the img-processing steps
    # You may uncoment the marked lines to follow the image process. steps

    # cv2.namedWindow("debug", cv2.WINDOW_NORMAL) ### uncomment for debug
    for i, img_file in enumerate(normal_img_files):
        if (i % 50 == 0):
            print("Loaded %04d/%04d images from the Normal dataset" % (i, N_normal_img_files))
        img = cv2.imread(os.path.join(normal_training_dir, img_file),0)
        # cv2.imshow("debug", img); cv2.waitKey(0) ### uncomment for debug
        img_trim = img[vertical_trim:-vertical_trim,horizonta_trim:-horizonta_trim]
        # cv2.imshow("debug", img_trim); cv2.waitKey(0) ### uncomment for debug
        if img_inv:
            img_bit = cv2.bitwise_not(img_trim)
            # cv2.imshow("debug", img_bit); cv2.waitKey(0) ### uncomment for debug
            img_resized = cv2.resize(img_bit, (img_width, img_height))
        else:
            img_resized = cv2.resize(img_trim, (img_width, img_height))
            # cv2.imshow("debug", img_trim); cv2.waitKey(0) ### uncomment for debug
        # cv2.imshow("debug", img_resized); cv2.waitKey(0) ### uncomment for debug
        img_keras  = img_resized.astype(float)
        normal_training_imgs[i,:,:,0] = img_keras
        normal_training_labels[i] = 0.0

    # After loading the Fluten and Normal images and defining the labels
    # It is necessary to concatenate the images and labels into a single
    # np. array. That's what is happning in the next lines
    N_train_img_files = N_normal_img_files + N_fluten_img_files
    training_imgs   = np.concatenate((fluten_training_imgs,   normal_training_imgs),   axis=0)
    training_labels = np.concatenate((fluten_training_labels, normal_training_labels), axis=0)

    # After concatenating, it is neessary to shuffle the images. This block
    # concatenateing the images throught the np.random.shuffle function
    indexes = np.arange(N_train_img_files)
    indexes_shuffled = indexes.copy(); np.random.shuffle(indexes_shuffled)
    training_imgs[indexes_shuffled]   = training_imgs[indexes]
    training_labels[indexes_shuffled] = training_labels[indexes]

    # After loading/processing the images, defining the labels and shuffle
    # the entire dataset, the information is stored in a .npz file
    np.savez("dataset_%02d_%02d_%s_%s" % (img_width, img_height, img_inv, train_string),
            imgs=training_imgs,
            labels=training_labels)
else:

    # If the .npz file exists, the dataset can be loaded from
    # hard-drive following the .npz-file nomenclature
    training_imgs   = dataset["imgs"]
    training_labels = dataset["labels"]

    N_train_img_files = training_labels.shape[0]

# Divinding the entire dataset into a training and validation samples,
# The training_split var defines the ration between the two datasets
validation_split = 0.2
N_validation_imgs = int(validation_split * N_train_img_files)
N_training_imgs   = N_train_img_files - N_validation_imgs
val_dataset   = (training_imgs[:N_validation_imgs], training_labels[:N_validation_imgs])
train_dataset = (training_imgs[N_validation_imgs:], training_labels[N_validation_imgs:])


# The keras-tuner is being created. Here, we decided to used the HyperBand hyperparameter
# search due to its popular use on the ML community
tuner = kt.Hyperband(model_builder,         # parametric-function used to build the CNN
                     objective='val_loss',  # val_loss, objective is to minmize its value
                     max_epochs=10,         # max-epochs during the keras-tuner hp search
                     seed=42,               # seed for random values
                     directory='dir_%02d_%02d_%s' % (img_width,   # we are using a fixed
                                                    img_height,   # nomencl to store the
                                                    img_inv),     # kt information
                     project_name='kt-keen')

# Early-stopping to avoid unnecessary training epochs
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# The keras-tuner search is called and run through this function
tuner.search(
        x = train_dataset[0],
        y = train_dataset[1],
        validation_data=val_dataset,
        epochs=50,
        callbacks=[stop_early]
        )

# There is no need to add an "IF" statement for the keras-tuner part,
# The keras-tuner implementation knows when the hyperparameter seach is
# completed and does not repeat the entire search when recalled
if train_mode:

    # Here we are getting the top 5 CNN models generated by the
    # keras-tuner.
    best_hps=tuner.get_best_hyperparameters(num_trials=5)

    # Then we loop through these models and train with
    # a larger dataset
    for trial, best_hp in enumerate(best_hps):
        model = tuner.hypermodel.build(best_hp)

        # Save the model with the best accuraccy while training
        saving_the_best = tf.keras.callbacks.ModelCheckpoint(
                'best_model_%02d_%02d_%s_%02d.h5' % (img_width, img_height, img_inv, trial),
                verbose=1,
                save_best_only=True,
                monitor='val_loss',
                mode='auto')

        # Early-stopping to avoid unnecessary training epochs
        early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                verbose=1,
                patience=20)

        # Training the model
        model.fit(
                x = train_dataset[0],
                y = train_dataset[1],
                validation_data=val_dataset,
                epochs=500,
                callbacks=[saving_the_best, early_stopping]
                )
