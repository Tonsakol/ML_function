from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Input, Lambda
from keras.layers import Activation
from keras.layers import Conv2D, Dense, concatenate, Dropout, Flatten
from keras.layers import SeparableConv2D, BatchNormalization
from keras.layers import MaxPooling2D, GlobalAvgPool2D
from keras.layers import add
from keras.utils.vis_utils import plot_model
from keras.models import load_model

from keras.applications.vgg16 import VGG16
#import efficientnet.tfkeras as efn 

IMAGE_SHAPE = (299, 299, 3)
META_DIM = 10

def sep_bn(x, filters, kernel_size, activation='relu'):
	x = SeparableConv2D(filters=filters, 
							kernel_size = kernel_size, 
							strides=1, 
							padding = 'same', 
							use_bias = False,
							activation= activation,
							kernel_initializer='he_normal')(x)
	x = BatchNormalization()(x)
	return x

# function for creating an identity or projection residual module
def residual_block(layer_in, n_filters):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = sep_bn(layer_in, n_filters, (1,1), 'linear')
		#merge_input = SeparableConv2D(n_filters, (1,1), padding='same', activation='linear', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = sep_bn(layer_in, layer_in.shape[-1], (4,4), 'elu')
	#conv1 = SeparableConv2D(layer_in.shape[-1], (4,4), padding='same', activation='elu' , kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = sep_bn(conv1, layer_in.shape[-1], (4,4), 'elu')
	#conv2 = SeparableConv2D(layer_in.shape[-1], (4,4), padding='same', activation='elu', kernel_initializer='he_normal')(conv1)
	# conv2
	conv3 = sep_bn(conv2, n_filters, (4,4), 'linear')
	#conv3 = SeparableConv2D(n_filters, (4,4), padding='same', activation='linear', kernel_initializer='he_normal')(conv2)
	# add filters, assumes filters/channels last
	layer_out = add([conv3, merge_input])
	# activation function
	#layer_out = Activation('relu')(layer_out)
	return layer_out

def pool(x, filters):
	x = sep_bn(x, filters, (4,4), 'relu')
	#x = SeparableConv2D(filters, (4,4), padding='same', activation='relu', kernel_initializer='he_normal')(x)
	x = MaxPooling2D(pool_size=3, strides=2, padding = 'valid')(x)
	return x

def cnn_net():
  
    model = VGG16(include_top = False, 
                    weights = 'imagenet', 
                    input_shape = IMAGE_SHAPE) # pooling = 'avg'

    x = Flatten()(model.output)
    #x = model.output
    #x = GlobalAveragePooling2D()(x)
    
    #x = Dense(512, activation = 'relu')(x)
    #x = Dropout(0.2)(x, training = True)
    
    #x = Dense(256, activation = 'relu')(x)
    #x = Dropout(0.2)(x, training = True)
 
    #x = Dense(128, activation = 'relu')(x)
    #x = Dropout(0.2)(x, training = True)
    
    x = Dense(8, activation = 'relu')(x)
    #x = Dropout(0.1)(x, training = True)

    
    #x = Dense(8, activation = 'relu')(x)
        
    model = Model(model.input, x)
            
    return model

def mlp_net():
    
    model = Sequential()
    model.add(Dense(8, input_dim = META_DIM, activation = "relu"))
    model.add(Dense(4, input_dim = META_DIM, activation = "relu"))

    #model.add(Dense(8, activation = "relu"))
        
    return model

# Combined network

def concatenated_net(cnn, mlp):
    
    combinedInput = concatenate([cnn.output, mlp.output])
    
    #x = Dense(128, activation="relu")(combinedInput)
    #x = Dropout(0.2)(x, training = True)
    
    x = Dense(8, activation="relu")(combinedInput)
    x = Dropout(0.2)(x, training = True)

    x = Dense(1, activation="sigmoid")(x) # because our metric is AUC, i.e. 
                                                      # softmax with two neurons will not work
    
    model = Model(inputs = [cnn.input, mlp.input], outputs = x)
    return model

def load_model_weights(model_path):
    MLP_NET = mlp_net()
    CNN_NET = cnn_net()
    model = concatenated_net(CNN_NET, MLP_NET)
    model_save_path = model_path
    model.save(model_save_path)
    model.load_weights(model_save_path)
    return model


#plot_model(CNN_NET, to_file = 'model_architecture.png', show_shapes = True, show_layer_names = False)
#model.summary()
