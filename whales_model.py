from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D,MaxPooling2D
from keras.engine import Layer
import keras.backend as K
     
def my_ConvNet(input_shape):
    
    model = Sequential()
    #conv1
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                     strides=2,
                     padding='SAME',
                     input_shape=input_shape,
                     activation='relu'
                    )
             )
    #pooling1
    model.add( MaxPooling2D(pool_size=2,strides=2) 
             )
    #conv2
    model.add(Conv2D(filters=256, kernel_size=5,
                     strides=2,
                     padding='SAME',
                     activation='relu'
                    )
             )
    #pooling2
    model.add( MaxPooling2D(pool_size=2,strides=2) 
             )
    #conv3
    model.add(Conv2D(filters=256, kernel_size=3,
                     strides=2,
                     padding='SAME',
                     activation='relu'
                    )
             )
    #conv4
    model.add(Conv2D(filters=384, kernel_size=3,
                     strides=1,
                     padding='SAME',
                     activation='relu'
                    )
             )
    #conv5
    model.add(Conv2D(filters=384, kernel_size=3,
                     strides=1,
                     padding='SAME',
                     activation='relu'
                    )
             )
    #pooling3
    model.add( MaxPooling2D(pool_size=2,strides=2) 
             )

    return model

class SoftmaxMap(Layer):
    # Init function
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(SoftmaxMap, self).__init__(**kwargs)

    # There's no parameter, so we don't need this one
    def build(self,input_shape):
        pass

    # This is the layer we're interested in: 
    # very similar to the regular softmax but note the additional
    # that we accept x.shape == (batch_size, w, h, n_classes)
    # which is not the case in Keras by default.
    def call(self, x, mask=None):
        enum = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        denom = K.sum(enum, axis=self.axis, keepdims=True)
        return enum / denom

    # The output shape is the same as the input shape
    def compute_output_shape(self, input_shape):
        return (input_shape)
