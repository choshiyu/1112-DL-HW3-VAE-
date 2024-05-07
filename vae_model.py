import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import os

class Sampling(layers.Layer):

    def call(self, inputs):
        
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0] # 取得z_mean的batch大小
        dim = tf.shape(z_mean)[1] # 取得z_mean的維度大小
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) # 生成一個符合標準正態分布的隨機張量
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon # 計算潛在向量z的公式

def define_encoder(latent_dim):

    encoder_inputs = keras.Input(shape=(32, 32, 3)) # 因為是cfar10所以32, 32, 3
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim + latent_dim)(x) # 因為output會有兩個z_mean跟z_log_var，所以要提供兩份維度來給兩個東東解碼
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var]) # 透過公式轉換為z

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    return encoder

def define_decoder(latent_dim):

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 32, activation="relu")(latent_inputs) # 把latent_inputs轉換成高維度，目的：捕捉更多特徵
    x = layers.Reshape((16, 16, 32))(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)# encoder是32-->64，所以要反過來
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x) # 反卷積層，第一個3是channel

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    return decoder

class VAE(keras.Model):
    
    def __init__(self, encoder, decoder, **kwargs):
        
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self): # 定義需要跟蹤的指標列表
        
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        
        with tf.GradientTape() as tape: # 使用梯度帶來計算梯度
            
            z_mean, z_log_var, z = self.encoder(data)# 得到潛在空間的均值、變異數和潛在向量
            reconstruction = self.decoder(z)# 通過解碼器重構輸入圖像
            
            # 計算重構loss，使用binary_crossentropy作為loss function
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) # 計算kl_loss
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) # 對損失進行平均
            total_loss = reconstruction_loss + kl_loss # total_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)# 計算梯度
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))# 使用優化器更新模型參數
        self.total_loss_tracker.update_state(total_loss)# 更新total_loss
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)# 更新reconstruction_loss
        self.kl_loss_tracker.update_state(kl_loss)# 更新kl_loss
        
        return { # 返回包含指標值的dict，用於顯示訓練過程中的loss
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
def plt_save(vae, target_class, n, latent_dim):
    
    folder_path = f'generate_class{target_class}_images'
    os.makedirs(folder_path, exist_ok=True)

    for i in range(n):
        z_sample = np.random.normal(size=(1, latent_dim))
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(32, 32, 3)

        # resized_digit = resize(digit, (128, 128, 3), anti_aliasing=True) # 變成可以看的圖
        
        img_name = f'digit_{i}.png'
        file_path = os.path.join(folder_path, img_name)
        plt.imsave(file_path, digit)