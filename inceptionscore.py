from numpy import expand_dims, log, mean, exp
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import os
import cv2

def calculate_inception_score(images, eps=1E-16):

    model = InceptionV3() # load inception v3 model
    yhat = []

    # preprocess raw images for inception v3 model
    for class_img in images:
            class_img = class_img.astype('float32')
            class_img = preprocess_input(class_img)
            yhat.append(model.predict(class_img))

    scores = []
    for i in range(10):

        print('######'+str(i)+'######')

        p_yx = yhat[i]
        p_y = expand_dims(p_yx.mean(axis=0), 0)# calculate p(y)
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))# calculate KL divergence using log probabilities
        sum_kl_d = kl_d.sum(axis=1)# sum over classes
        avg_kl_d = mean(sum_kl_d)# average over images
        is_score = exp(avg_kl_d)# undo the log
        scores.append(is_score)
            
    is_avg = mean(scores)

    return is_avg

if __name__ == '__main__':

    folder_path = './img'
    images = []

    for folder in os.listdir(folder_path):

        folder_dir = os.path.join(folder_path, folder)
        img = []
        
        for filename in os.listdir(folder_dir): # 遍歷每張圖片
            file_path = os.path.join(folder_dir, filename)
            
            image = cv2.imread(file_path)# read img
            resized_image = cv2.resize(image, (299, 299))# resize
            img.append(resized_image)# 合到img
        images.append(img) # 合到images

    images_array = np.array(images)
    inception_score = calculate_inception_score(images_array)
    
    print('inception_score:', inception_score)