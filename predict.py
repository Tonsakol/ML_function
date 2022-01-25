from data_prep import *
from enhance_img import *
from model import *

class Predict():
    def __init__(self, MODEL_PATH):
        self.model_path = MODEL_PATH
        self.model = load_model_weights(self.model_path)

    def gen_enhance_img(self, IMAGE):
        denoise = noise_removal(IMAGE)
        en_img = enhance_img(IMAGE)
        return en_img

    def gen_metadata(self, age_approx, anatom_site, sex):
        age_max = 85.0
        feature_col = ['age_approx_norm', '_anterior torso', '_head/neck', '_lateral torso',
        '_lower extremity', '_oral/genital', '_palms/soles', '_posterior torso',
        '_upper extremity', '_male']

        meta = pd.DataFrame(0,index=range(1), columns=feature_col)
        meta['age_approx_norm'] = age_approx / age_max

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
            meta['_upper extremity'] = 1
        
        if sex == 'male':
            meta['_male'] == 1
        else:
            meta['_male'] == 0
        
        return meta

    def predict_img(self, IMAGE, metadata):
        IMAGE_SHAPE = (299, 299, 3)
        META_DIM = 10
        img = self.gen_enhance_img(IMAGE)
        img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

        meta = metadata
        meta = meta.to_numpy()
        meta = np.reshape(meta, (1, META_DIM))
        meta = meta.astype(np.float32)

        combined_input = [img, meta]
        return self.model.predict(combined_input)