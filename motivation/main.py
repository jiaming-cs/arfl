import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

spaces = 10

IMAGE_SIZE = 28
NUM_CLASS = 10
seed = 221
tf.random.set_seed(seed)
np.random.seed(seed)


class Server:
    def __init__(self):
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        # Scale images to the [0, 1] range
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255
        self.X_train = X_train.reshape((-1, 28 * 28))
        self.X_test = X_test.reshape((-1, 28 * 28))
        self.Y_test = Y_test
        self.Y_train = Y_train


    def split_dataset(self, alpha, corrupte=False):
        min_size = 0
        K = 10
        N = self.Y_train.shape[0]
        num_clients = 2
        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(self.Y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                ## Balance
                # proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        num_samples = min(len(i) for i in idx_batch)
        np.random.shuffle(idx_batch[0])
        np.random.shuffle(idx_batch[1])
        idx_batch = [data[:num_samples] for data in idx_batch]
        self.X_train1, self.Y_train1 = self.X_train[idx_batch[0], :], self.Y_train[idx_batch[0]]
        self.X_train2, self.Y_train2 = self.X_train[idx_batch[1], :], self.Y_train[idx_batch[1]]

        if corrupte:
            sz = self.X_train2.shape[0]
            n_poisoned = int(sz * 0.9)
            poisoned_points = np.random.choice(sz, n_poisoned, replace=False)
            self.Y_train2[poisoned_points] = 0

        self.X_train = np.concatenate((self.X_train1, self.X_train2), axis=0)
        self.Y_train = np.concatenate((self.Y_train1, self.Y_train2), axis=0)
        return self.X_train1, self.Y_train1, self.X_train2, self.Y_train2

    def get_average_loss(self, theta, param1, param2, use_set='test'):
        model_avg = self.create_model()
        averaged_param = [(theta * var1 + (1 - theta) * var2) for var1, var2 in zip(param1, param2)]
        model_avg.set_weights(averaged_param)
        if use_set == 'test':
            metric = model_avg.evaluate(self.X_test, self.Y_test, verbose=0)
        else:
            metric = model_avg.evaluate(self.X_train, self.Y_train, verbose=0)
        print(metric)
        return metric[0]


    def train_models(self, alpha=0.5, corrupte=False, epochs=20):
        tf.random.set_seed(222)
        np.random.seed(222)
        model1 = self.create_model()
        model2 = self.create_model()

        model2.set_weights(model1.get_weights())
        X_train1, Y_train1, X_train2, Y_train2 = self.split_dataset(alpha=alpha, corrupte=corrupte)
        model1.fit(X_train1, Y_train1, epochs=epochs, verbose=0)
        model2.fit(X_train2, Y_train2, epochs=epochs, verbose=0)
        self.paras_model1, self.paras_model2 = model1.get_weights(), model2.get_weights()
        return model1.get_weights(), model2.get_weights()

    def get_curve(self,points=10):
        test_curve = pd.Series([self.get_average_loss(theta, self.paras_model1, self.paras_model2, 'test')
                                 for theta in np.linspace(-0.2, 1.2, points)], index=np.linspace(-0.2, 1.2, points))
        train_curve = pd.Series([self.get_average_loss(theta, self.paras_model1, self.paras_model2, 'train')
                                 for theta in np.linspace(-0.2, 1.2, points)], index=np.linspace(-0.2, 1.2, points))

        train_curve.plot()
        test_curve.plot()
        plt.show()
        return train_curve, test_curve

    def create_model(self):
        model = keras.Sequential(
            [
                layers.Dense(200, activation="relu", name="layer1", input_shape= (IMAGE_SIZE * IMAGE_SIZE,)),
                layers.Dense(200, activation="relu", name="layer2"),
                layers.Dense(10, name="output"),
            ]
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss=loss, metrics=['accuracy'])
        return model



if __name__ == "__main__":
    server = Server()
    server.train_models(epochs=20, alpha=0.2, corrupte=True)
    server.get_curve(points=10)