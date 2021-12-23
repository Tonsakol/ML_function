import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class Dataset():
    def __init__(self, META_PATH, IMAGE_LIST_PATH):
        self.metadata = pd.read_csv(META_PATH)
        self.imageList = pd.read_csv(IMAGE_LIST_PATH)

        self.image_width = 299
        self.image_height = 299
        self.image_size = (self.image_width, self.image_height)
        self.image_channel = 3
        self.batch_size = 16

        self.age_max =None
        self.mel_df = None
        self.be_df = None
        self.df = None
        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.total_train = None
        self.total_validate = None

    def gen_meta_input(self, age_approx, anatom_site, sex):
        feature_col = ['age_approx_norm', '_anterior torso', '_head/neck', '_lateral torso',
       '_lower extremity', '_oral/genital', '_palms/soles', '_posterior torso',
       '_upper extremity', '_male']

        meta = pd.DataFrame(0,index=range(1), columns=feature_col)
        meta['age_aprox_norm'] = age_approx / self.age_max

        if anatom_site == 'anaterior torso':
            meta['_anterior torso'] = 1
        elif anatom_site == 'head/neck':
            meta['_head/neck'] = 1
        elif anatom_site == 'lateral torso':
            meta['_lateral torso'] = 1
        elif anatom_site == 'lower extremity':
            meta['_lower extremity'] = 1
        elif anatom_site == 'oral/genital':
            meta['_oral/genital'] = 1
        elif anatom_site == 'palms/soles':
            meta['_palms/soles'] = 1
        elif anatom_site == 'posterior torso':
            meta['_posterior torso'] = 1
        elif anatom_site == 'upper extremity':
            meta['upper extremity'] = 1
        
        if sex == 'male':
            meta['_male'] == 1
        else:
            meta['_male'] == 0
        
        return meta

    def clean_data(self):
        metadata = self.get_metadata()
        mel_df = metadata[metadata['diagnosis'] == 'melanoma']
        mel_df = mel_df[mel_df['sex'].notna()]
        mel_df = mel_df[mel_df['age_approx'].notna()]
        mel_df = mel_df[mel_df['anatom_site_general'].notna()]
        mel_df = mel_df[['isic_id','diagnosis','age_approx','anatom_site_general','sex']]
        mel_df['isic_id'] = mel_df['isic_id'] + '.jpg'
        mel_df['name'] = mel_df.apply(lambda x: self.rename_fn(x['diagnosis'], x['isic_id']), axis=1)

        be_df = metadata[metadata['diagnosis'] == 'nevus']
        be_df = be_df[be_df['sex'].notna()]
        be_df = be_df[be_df['age_approx'].notna()]
        be_df = be_df[be_df['anatom_site_general'].notna()]
        be_df = be_df[['isic_id','diagnosis','age_approx','anatom_site_general','sex']]
        be_df['isic_id'] = be_df['isic_id'] + '.jpg'
        be_df['name'] = be_df.apply(lambda x: self.rename_fn(x['diagnosis'], x['isic_id']), axis=1)
        
        df = pd.concat([mel_df, be_df])
        df['name'] = df.apply(lambda x: self.rename_fn(x['diagnosis'], x['isic_id']), axis=1)
        self.set_age_max(df['age_approx'].max())

        df['age_approx_norm'] = df['age_approx'] / df['age_approx'].max()
        df = pd.get_dummies(df, columns = ['anatom_site_general'], prefix = [''])
        df = pd.get_dummies(df, columns = ['sex'], prefix = [''],drop_first=True)
        df.loc[(df['diagnosis'] == 'melanoma'),'targets'] = 1
        df.loc[(df['diagnosis'] == 'nevus'),'targets'] = 0
        df['targets'] = df['targets'].astype(np.uint8)
        df = df.drop(columns=['isic_id', 'age_approx','diagnosis'])

        self.set_df(df)

    def split_train_test_val(self):
        df = self.get_df()
        tmp_df, test_df = train_test_split(df,test_size = 0.15, random_state = 42)
        train_df, validation_df = train_test_split(tmp_df,test_size = 0.20, random_state = 42, stratify = tmp_df['targets'])

        self.set_train_df(train_df.reset_index(drop=True))
        self.set_validation_df(validation_df.reset_index(drop=True))
        self.set_test_df(test_df.reset_index(drop=True))

        self.set_total_train(train_df.shape[0])
        self.set_total_validate(validation_df.shape[0])

    def get_train_val_data_gen(self):
        image_size = self.get_image_size()
        batch_size = self.get_batch_size()
        train_df = self.get_train_df()
        validation_df = self.get_validation_df()

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            height_shift_range=0.2,
            width_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip = True,
        )
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df, 
            directory='/content/drive/MyDrive/capstone/isic_resize', 
            x_col='name',
            y_col=train_df.columns[1:],
            target_size=image_size,
            class_mode='raw',
            batch_size=batch_size
        )

        validation_datagen = ImageDataGenerator(
            rescale=1./255
            )
        validation_generator = validation_datagen.flow_from_dataframe(
            validation_df, 
            '/content/drive/MyDrive/capstone/isic_resize', 
            x_col='name',
            y_col=validation_df.columns[1:],
            target_size=image_size,
            class_mode='raw',
            batch_size=batch_size
        )

        return train_generator, validation_generator

    def get_class_weight(self):
        df = self.get_df()
        neg, pos = np.bincount(df['targets'])
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        return class_weight

    def rename_fn(self, diag, name):
        return '/content/drive/MyDrive/capstone/Isic_data/' + diag + '/' + name

    def get_metadata(self):
        return self.metadata
    def set_metadata(self, metadata):
        self.metadata = metadata

    def get_age_max(self):
        return self.age_max
    def set_age_max(self, age_max):
        self.age_max = age_max

    def get_df(self):
        return self.df
    def set_df(self, df):
        self.df = df
    
    def get_train_df(self):
        return self.train_df
    def set_train_df(self, train_df):
        self.train_df = train_df
    
    def get_validation_df(self):
        return self.validation_df
    def set_validation_df(self, validation_df):
        self.validation_df = validation_df

    def get_test_df(self):
        return self.test_df
    def set_test_df(self, test_df):
        self.test_df = test_df

    def get_total_train(self):
        return self.total_train
    def set_total_train(self,total_train):
        self.total_train = total_train

    def get_total_validate(self):
        return self.total_validate  
    def set_total_validate(self, total_validate):
        self.total_validate = total_validate

    def get_batch_size(self):
        return self.batch_size
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def get_image_size(self):
        return self.image_size
    def set_image_size(self, image_size):
        self.image_size = image_size
