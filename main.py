import numpy as np
from tensorflow import keras
from keras.datasets import cifar10
from vae_model import define_encoder, define_decoder, VAE, plt_save

latent_dim = 128
n = 1000 # 每個類別生成n張
Epochs = 40
Batch_Size = 128

if __name__ == '__main__':

    for target_class in range(10):

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        x_train_filtered = x_train[y_train.flatten() == target_class]
        x_test_filtered = x_test[y_test.flatten() == target_class]

        cifar10_data = np.concatenate([x_train_filtered, x_test_filtered], axis=0)
        cifar10_data = cifar10_data.astype("float32") / 255.0
        cifar10_data = cifar10_data.reshape(cifar10_data.shape[0], 32, 32, 3)

        encoder = define_encoder(latent_dim)
        decoder = define_decoder(latent_dim)
        
        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(cifar10_data, epochs=Epochs, batch_size=Batch_Size)
        
        plt_save(vae, target_class, n)
