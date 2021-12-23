from predict import *

MODEL_PATH = "D:\senior\capstone\ML_function\model2.hdf5"
IMG_PATH = 'D:\senior\capstone\ML_function\ISIC_0000013.jpg'

Predict = Predict(MODEL_PATH)
#fill metadata (age_approx, anatom_site, sex)
meta = Predict.gen_metadata(60, 'upper extremity', 'male')
score = Predict.predict_img(IMG_PATH, meta)

#return probability score
# 1 -> melanoma
# 0 -> nevus
print(score[0][0])

