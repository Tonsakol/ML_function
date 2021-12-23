from data_prep import *
from enhance_img import *
from model import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

META_PATH = 'D:\senior\capstone\ML_function\ISIC\metadata.csv'
IMAGE_LIST_PATH = 'D:\senior\capstone\ML_function\ISIC\imageList.csv'
image_width=299
image_height=299
image_size=(image_width,image_height)
image_channel=3 #RGB color
batch_size = 16

def own_train_generator_func(train_df, train_generator):
    count = 0
    while True:
        if count == len(train_df.index):
            train_generator.reset()
            #break
        count += 1
        data = train_generator.next()
        
        imgs = data[0]
        meta = data[1][:,:-1]
        targets = data[1][:,-1:]
        #print(targets)
        yield [imgs, meta], targets

def own_validation_generator_func(validation_df, validation_generator):
    count = 0
    while True:
        if count == len(validation_df.index):
            validation_generator.reset()
            #break
        count += 1
        data = validation_generator.next()
                
        imgs = data[0]
        meta = data[1][:,:-1]
        targets = data[1][:,-1:]
        
        yield [imgs, meta], targets

def focal_loss(alpha = 0.25, gamma = 2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true * y_pred) + ((1-y_true) * (1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true * alpha + ((1-alpha) * (1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor * modulating_factor * bce, axis = -1)
    
    return focal_crossentropy

def training_model(model, epochs, data, save_path):
    total_train = data.get_total_train()
    total_validate = data.get_total_validate()
    batch_size = data.get_batch_size()
    class_weight = data.get_class_weight()
    train_df = data.get_train_df()
    validation_df = data.get_validation_df()
    train_generator, validation_generator = data.get_train_val_data_gen()
    earlystop = EarlyStopping(monitor='val_auc', min_delta=0.0001,patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_auc', 
                                                patience=10, 
                                                verbose=1, 
                                                factor=0.2, 
                                                min_lr=0.000001)

    #checkpoint
    
    checkpoint = ModelCheckpoint(save_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
    callbacks = [earlystop, learning_rate_reduction, checkpoint]

    hist = model.fit(
            #train_generator,
            own_train_generator_func(train_df, train_generator),
            steps_per_epoch= total_train//batch_size,
            epochs= epochs,  # epochs: Integer, total number of iterations on the data.
            #validation_data= validation_generator,
            validation_data= own_validation_generator_func(validation_df, validation_generator),
            validation_steps= total_validate//batch_size,
            callbacks= callbacks,
            class_weight=class_weight
            )
    return hist

def plot_AUC_Loss(hist):
    auc = hist.history['auc']
    val_auc = hist.history['val_auc']

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(auc, label='Training')
    plt.plot(val_auc, label='Validation')
    plt.legend(loc='lower right')
    plt.ylabel('auc')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation AUC')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,0.2])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

if __name__ == "__main__":

    data = Dataset(META_PATH, IMAGE_LIST_PATH)
    data.clean_data()
    data.split_train_test_val()
    
   
    opt = Adam(learning_rate = 1e-05)

    MLP_NET = mlp_net()
    CNN_NET = cnn_net()
    model = concatenated_net(CNN_NET, MLP_NET)
    model.compile(loss = focal_loss(), metrics = [tf.keras.metrics.AUC(name = 'auc')], optimizer = opt)
    
    save_path="/content/drive/MyDrive/capstone/weight/eff2_weights.hdf5"
    hist = training_model(model, 10, data, save_path)
    
    plot_AUC_Loss(hist)

