from keras.models import Model
from keras.layers import Dense,Conv2D,Dropout,BatchNormalization,Activation,Input,Concatenate
from keras.layers import MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D
from keras.regularizers import l2
from keras.initializers import he_uniform

class MY_DenseNet(object):
    '''建立一個DenseNet模型。'''
    def __init__( self,in_shape,out_classes,
                  growth_rate=12,
                  block_layers=[16,16,16],
                  weight_decay=1.E-4,
                  first_conv_params={"fsize":(3,3),"stride":1,"maxpool":False},
                  dropout_rate=0.2
                ):

        self.in_shape=in_shape
        self.out_classes=out_classes
        
        self.growth_rate = growth_rate
        self.block_layers = block_layers
        
        self.channels_1st_conv = 2*self.growth_rate
        self.first_conv_params=first_conv_params
        
        self.wd=weight_decay
        self.dropout_rate=dropout_rate

    def conv_block(self,channels,x):
        '''建構convolution block。'''
        bn1=BatchNormalization(gamma_regularizer=l2(self.wd),
                              beta_regularizer=l2(self.wd))
        act1=Activation("relu")
        conv1=Conv2D(4*channels, (1, 1),
                     use_bias=False,
                     kernel_regularizer=l2(self.wd),
                     kernel_initializer=he_uniform())
        drop1=Dropout(self.dropout_rate)
        
        bn2=BatchNormalization(gamma_regularizer=l2(self.wd),
                              beta_regularizer=l2(self.wd))
        act2=Activation("relu")
        conv2=Conv2D(channels, (3, 3), padding="SAME",
                     use_bias=False,
                     kernel_regularizer=l2(self.wd),
                     kernel_initializer=he_uniform())
        drop2=Dropout(self.dropout_rate)
        
        x= drop2(
             conv2(
               act2( 
                 bn2(
                   drop1(
                     conv1(
                       act1(
                         bn1( x ) ) ) ) ) ) ) )
        return x
    
    def dense_block(self,num_layers,growth_rate,x):
        '''建構dense block。'''
        for j in range(num_layers):
            out=self.conv_block(growth_rate,x)
            x=Concatenate(axis=-1)([x,out])
        return x
    
    def trans_block(self,channels,x):
        '''建構transition block。'''
        bn=BatchNormalization(gamma_regularizer=l2(self.wd),
                              beta_regularizer=l2(self.wd))
        act=Activation("relu")
        conv=Conv2D(channels, (1, 1),
                    use_bias=False,
                    kernel_regularizer=l2(self.wd),
                    kernel_initializer=he_uniform())
        dropout=Dropout(self.dropout_rate)
        avg_pool=AveragePooling2D(pool_size=2,strides=2)
        
        x= avg_pool(
              dropout(
                   conv(
                      act(
                         bn( x ) ) ) ) )
        return x
        
    def build_model(self):
        '''以Model(input,output)的方式建立模型。'''
        image = Input(self.in_shape)         # input image
        # =============
        # initial block
        # =============
        conv_i=Conv2D(self.channels_1st_conv,self.first_conv_params["fsize"],
                      strides=self.first_conv_params["stride"],
                      padding="SAME",
                      use_bias=False,
                      kernel_regularizer=l2(self.wd),
                      kernel_initializer=he_uniform())
        bn_i=BatchNormalization(gamma_regularizer=l2(self.wd),
                                beta_regularizer=l2(self.wd))
        act_i=Activation("relu")
                 
        if self.first_conv_params["maxpool"] is False:        
            x= conv_i(
                      act_i(
                            bn_i( image ) ) )
        else:
            max_pool_i=MaxPooling2D(pool_size=3,strides=2)
            x= max_pool_i(
                          conv_i(
                                 act_i(
                                       bn_i( image ) ) ) )
        # =========================
        # dense +transition blocks
        # =========================
        channels = self.channels_1st_conv
        for i, num_layers in enumerate(self.block_layers):
            x=self.dense_block(num_layers,self.growth_rate,x)
            channels += num_layers * self.growth_rate
            if i != len(self.block_layers)-1:
                x=self.trans_block(channels//2,x)
        # =============
        # final block
        # =============
        bn_f=BatchNormalization(gamma_regularizer=l2(self.wd),
                                beta_regularizer=l2(self.wd))
        act1_f=Activation("relu")
        glb_avg_pool_f=GlobalAveragePooling2D()
        dense_f=Dense(self.out_classes,
                      kernel_regularizer=l2(self.wd),
                      bias_regularizer=l2(self.wd),
                      kernel_initializer=he_uniform())
        act2_f=Activation("softmax")

        x=act2_f(
               dense_f(
                    glb_avg_pool_f(
                         act1_f(
                              bn_f( x ) ) ) ) )
        # ==========================================================
        # connect the input & output tensor in order to form a model
        # ==========================================================
        model=Model(image,x)
        return model