import cv2
from skimage.io import imread

import numpy as np

from keras.layers import Dense, Conv2D, Input, MaxPooling2D, concatenate, Conv2DTranspose
from keras.models import Model

def get_unet( fig_size = (128,128) ):
    """定義UNet模型架構。"""
    
    inputs = Input( (*fig_size,3) )
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='SAME')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='SAME')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='SAME')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='SAME')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='SAME')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='SAME')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='SAME')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='SAME')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='SAME')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='SAME')(conv5)

    up6 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='SAME')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='SAME')(conv6)

    up7 = concatenate([Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='SAME')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='SAME')(conv7)

    up8 = concatenate([Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='SAME')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='SAME')(conv8)

    up9 = concatenate([Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='SAME')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='SAME')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

class UNet_utils(object):
    """Utility functions to prepare the data for UNet."""
    
    @staticmethod
    def grey2rgb(img):
        """Convert greyscale images to rgb."""
        
        new_img = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img.append(list(img[i][j])*3)
        new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
        return new_img

    # generator that we will use to read the data from the directory
    @staticmethod
    def data_gen_small(data_dir, mask_dir, images, batch_size, dims):
            """
            data_dir: where the actual images are kept
            mask_dir: where the actual masks are kept
            images: the filenames of the images we want to generate batches from
            batch_size: self explanatory
            dims: the dimensions in which we want to rescale our images
            """
            while True:
                ix = np.random.choice(np.arange(len(images)), batch_size)
                imgs = []
                labels = []
                for i in ix:
                    # images
                    array_img = cv2.imread(data_dir + images[i])
                    array_img = cv2.resize(array_img,dims)/255.
                    imgs.append(array_img[:,:,::-1])

                    # masks
                    array_mask = imread(mask_dir + images[i].split(".")[0] + '_mask.gif')
                    array_mask = cv2.resize(array_mask,dims)/255.
                    labels.append(array_mask[:, :])
                imgs = np.array(imgs)
                labels = np.array(labels)
                yield imgs, labels.reshape(-1, dims[0], dims[1], 1)