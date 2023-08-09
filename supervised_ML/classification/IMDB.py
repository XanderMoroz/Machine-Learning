""" ИЗВЛЕЧЕНИЕ ДАННЫХ """

from keras.datasets import imdb

# Извлечение данных (отзывов) и меток (хороший или плохой отзыв) в учебный и тестовый наборы
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


""" ПОДГОТОВКА ДАННЫХ """

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
  """
  Функция создающая 10000-мерную матрицу из списка закодированных слов
  """
  results = np.zeros((len(sequences), dimension))     # Создание матрицы с формой (len(sequences),
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.                         # Запись единицы в элемент с данным индексом
  return results

# Подготовка учебных и проверочных данных
x_train = vectorize_sequences(train_data)             # Векторизованные обучающие данные (данные переведенные в матрицы векторов с 10 000 )
x_test = vectorize_sequences(test_data)               # Векторизованные контрольные данные


# Подготовка учебных и проверочных меток к данным
y_train = np.asarray(train_labels).astype('float32')  # Векторизованные метки к учебному набору
y_test = np.asarray(test_labels).astype('float32')    # Векторизованные метки к контрольному набору

# Создание проверочного набора данных
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

# Создание проверочного набора меток к данным
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


""" КОНСТРУИРОВАНИЕ МОДЕЛИ """

from keras import models, layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x_train,
          y_train,
          epochs=4,
          batch_size=512)

results = model.evaluate(x_test, y_test)

print(results)