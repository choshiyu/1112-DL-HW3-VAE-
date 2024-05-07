import numpy as np
from numpy import cov, trace, iscomplexobj, asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.transform import resize
import os
import cv2
from keras.datasets import cifar10

'''
calculate frechet inception distance
'''

def scale_images(images, new_shape):
    
	images_list = list()
	for image in images:
		new_image = resize(image, new_shape, 0)
		images_list.append(new_image)
  
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	
    # calculate activations
	act1, act2 = model.predict(images1), model.predict(images2)
 
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
 
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
 
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
 
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	
    # calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
 
	return fid

if __name__ == '__main__':

    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    # fake img
    images1 = randint(0, 255, 10*32*32*3)
    images1 = images1.reshape((10,32,32,3))
    images2 = randint(0, 255, 10*32*32*3)
    images2 = images2.reshape((10,32,32,3))

    folder_path = './img'; fake_images = []
    for folder in os.listdir(folder_path):

        folder_dir = os.path.join(folder_path, folder)
        img = []
        
        for filename in os.listdir(folder_dir):
            file_path = os.path.join(folder_dir, filename)
            image = cv2.imread(file_path)
            img.append(image)
        
        img = np.array(img)
        fake_images.append(img)
    fake_images = np.array(fake_images)
	
    # real
    real_img = []
    for target_class in range(10):
        (_, _), (x_test, y_test) = cifar10.load_data()
        x_test_filtered = x_test[y_test.flatten() == target_class]
        real_img.append(x_test_filtered)

    for i in range(10):

        images1 = fake_images[i]
        images2 = real_img[i]
        print('class'+str(i))
        
        # print('Prepared', images1.shape, images2.shape)

        images1, images2 = images1.astype('float32'), images2.astype('float32')

        # resize
        images1, images2 = scale_images(images1, (299,299,3)), scale_images(images2, (299,299,3))
        # print('Scaled', images1.shape, images2.shape)
        
        # pre-process images
        images1, images2 = preprocess_input(images1), preprocess_input(images2)

        # fid between images1 and images1
        fid = calculate_fid(model, images1, images1)
        print('FID (same): %.3f' % fid)
        
        # fid between images1 and images2
        fid = calculate_fid(model, images1, images2)
        print('FID (different): %.3f' % fid, '\n-------------------')