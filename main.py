from keras.datasets import imdb

"""Загрузка набора данных MNIST в Keras."""

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# Здесь train_images, train_labels - это тренировочный набор.
# После обучения модель будет проверяться тестовым набором.

print(train_data[0])