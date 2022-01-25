from predict import *

MODEL_PATH = "D:\senior_2\capstone\ML_function\model2.hdf5"
IMG_PATH = 'D:\senior_2\capstone\ML_function\ISIC_0000013.jpg'

Predict = Predict(MODEL_PATH)
#fill metadata (age_approx, anatom_site, sex)
    # 8 anatomy site : anaterior torso, head/neck, lateral torso, lower extremity, oral/genital, palms/soles, posterior torso, upper extremity
meta = Predict.gen_metadata(60, 'upper extremity', 'male')

#read img
IMAGE = cv2.imread(IMG_PATH)
#Predict score
score = Predict.predict_img(IMAGE, meta)

#return probability score
# 1 -> melanoma
# 0 -> nevus
print(score[0][0])

