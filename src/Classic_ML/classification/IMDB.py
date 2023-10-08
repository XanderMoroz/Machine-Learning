from keras.datasets import imdb

from keras import models, layers

import numpy as np


# Constants

DIMENSION = 10000

EPOCHS = 4

BATCH_SIZE = 512


def load_imdb_data():

    """Load IMDB data."""

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=DIMENSION)

    return train_data, train_labels, test_data, test_labels


def vectorize_sequences(sequences, dimension=DIMENSION):

    """

    Create a 10,000-dimensional matrix from a list of encoded words.

    """

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1.  

    return results


def prepare_data(train_data, test_data, train_labels, test_labels):

    """Prepare training and testing data."""

    x_train = vectorize_sequences(train_data)  

    x_test = vectorize_sequences(test_data) 


    y_train = np.asarray(train_labels).astype('float32')  

    y_test = np.asarray(test_labels).astype('float32')    


    x_val = x_train[:DIMENSION]

    partial_x_train = x_train[DIMENSION:]


    y_val = y_train[:DIMENSION]

    partial_y_train = y_train[DIMENSION:]


    return x_train, x_test, y_train, y_test, x_val, partial_x_train, y_val, partial_y_train


def create_model():

    """Construct the model."""

    model = models.Sequential()

    model.add(layers.Dense(16, activation='relu', input_shape=(DIMENSION,)))

    model.add(layers.Dense(16, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):

    """Train and evaluate the model."""

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(f'Model accuracy on test data: {test_acc}')


if __name__ == "__main__":

    train_data, train_labels, test_data, test_labels = load_imdb_data()

    x_train, x_test, y_train, y_test, x_val, partial_x_train, y_val, partial_y_train = prepare_data(train_data, test_data, train_labels, test_labels)

    model = create_model()

    train_and_evaluate_model(model, x_train, y_train, x_test, y_test)