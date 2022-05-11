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



# Image file format:
img_file_format = "png"

# Looks for all the .h5 model files in a folder and store in a list
h5_files = []
files_in_this_folder = os.listdir(os.getcwd())
for _file in files_in_this_folder:
    if _file.endswith(".h5") and "best_model" in _file:
        print(_file)
        h5_files.append(_file)

# Loops for all the .h5 model files found.
# The idea is load each .h5 file, extract its information from
# its filename and produced a log with the essential model information
h5_files.sort()
for h5_file in h5_files:
    keys = h5_file.split(".h5")[0].split("_")
    img_width  =  int(keys[2])
    img_height =  int(keys[3])
    img_inv    = bool(keys[4])
    trial      =  int(keys[5])


    # Vertical and Horizontal trim from the original dataset
    # While the training the CNN, we noticed that the camera
    # positioning affected the acc. results. Hence, we decide
    # to remove the background from the original images
    vertical_trim = 50
    horizonta_trim = 80

    # Definition of the dataset folder
    dataset_dir = ""

    # Defining the Training folders and a string to differentiate
    # between the training and the keras-tuner runs
    validation_dir = os.path.join(dataset_dir,"Validation")
    fluten_validation_dir = os.path.join(validation_dir,"Fluten")
    normal_validation_dir = os.path.join(validation_dir,"Normalzustand")

    # The dataset can be loaded multiple times during a train run, and
    # the loading/processing step may take some time if the dataset is
    # large. Hence, once the dataset is lodaded, the images are stored
    # in a .npz file, which loads fast and contains processed images
    try:
        dataset = np.load("val_dataset_%02d_%02d_%s.npz" % (img_width, img_height, img_inv))
        creata_data = False
        print("Loading dataset")
    except FileNotFoundError:
        creata_data = True
        print("Creating dataset....")


    if creata_data:

        ############### Loading Fluten images ###############
        # Finding the jpg images on the folder
        fluten_validation_files = os.listdir(fluten_validation_dir)
        fluten_img_files = []
        for _file in fluten_validation_files:
            if img_file_format in _file:
                fluten_img_files.append(_file)

        # Creating the image and label np arrays
        N_fluten_img_files = len(fluten_img_files)
        fluten_validation_imgs   = np.zeros((N_fluten_img_files, img_height, img_width,1), dtype=float)
        fluten_validation_labels = np.zeros((N_fluten_img_files), dtype=float)

        # Loading the images from the folder, applying the img-processing steps
        # You may uncoment the marked lines to follow the image process. steps

        # cv2.namedWindow("debug", cv2.WINDOW_NORMAL) ### uncomment for debug
        for i, img_file in enumerate(fluten_img_files):
            if (i % 50 == 0):
                print("Loaded %04d/%04d images from the fluten dataset" % (i, N_fluten_img_files))
            img = cv2.imread(os.path.join(fluten_validation_dir, img_file),0)
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
            fluten_validation_imgs[i,:,:,0] = img_keras
            fluten_validation_labels[i] = 1.0

        ############### Loading Normal images ###############
        # Finding the jpg images on the folder
        normal_validation_files = os.listdir(normal_validation_dir)
        normal_img_files = []
        for _file in normal_validation_files:
            if img_file_format in _file:
                normal_img_files.append(_file)

        # Creating the image and label np arrays
        N_normal_img_files = len(normal_img_files)
        normal_validation_imgs   = np.zeros((N_normal_img_files, img_height, img_width,1), dtype=float)
        normal_validation_labels = np.zeros((N_normal_img_files), dtype=float)

        # Loading the images from the folder, applying the img-processing steps
        # You may uncoment the marked lines to follow the image process. steps

        # cv2.namedWindow("debug", cv2.WINDOW_NORMAL) ### uncomment for debug
        for i, img_file in enumerate(normal_img_files):
            if (i % 50 == 0):
                print("Loaded %04d/%04d images from the Normal dataset" % (i, N_normal_img_files))
            img = cv2.imread(os.path.join(normal_validation_dir, img_file),0)
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
            normal_validation_imgs[i,:,:,0] = img_keras
            normal_validation_labels[i] = 0.0

        # After loading the Fluten and Normal images and defining the labels
        # It is necessary to concatenate the images and labels into a single
        # np. array. That's what is happning in the next lines
        N_train_img_files = N_normal_img_files + N_fluten_img_files
        validation_imgs   = np.concatenate((fluten_validation_imgs,   normal_validation_imgs),   axis=0)
        validation_labels = np.concatenate((fluten_validation_labels, normal_validation_labels), axis=0)

        # After loading/processing the images, defining the labels and shuffle
        # the entire dataset, the information is stored in a .npz file
        np.savez("val_dataset_%02d_%02d_%s" % (img_width, img_height, img_inv),
                imgs=validation_imgs,
                labels=validation_labels)
    else:

        # If the .npz file exists, the dataset can be loaded from
        # hard-drive following the .npz-file nomenclature
        validation_imgs   = dataset["imgs"]
        validation_labels = dataset["labels"]

    model_name = 'best_model_%02d_%02d_%s_%02d.h5' % (img_width, img_height, img_inv, trial)
    model = tf.keras.models.load_model(model_name, compile=True)
    model_train_params = model.count_params()

    # model.summary()
    score = model.evaluate(x=validation_imgs, y=validation_labels, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])

    # If you run this code, try to search for the word OUT in the stdout of script
    # I a Linux SO, you run "python test_model_acc.py | tee log.txt and the stdout 
    # goest to the file "log.txt". Then, from this file, you search(grep) the word
    # "OUT" and get the model details
    print("OUT: %02d, %03d, %03d, %s, %.4f, %.4e, %d" % (trial, img_height, img_width, img_inv, score[1], score[0], model_train_params))
