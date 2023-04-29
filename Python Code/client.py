import pandas as pd
import flwr as fl
import collections
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        history = self.model.fit(self.x_train, self.y_train, batch_size, epochs, shuffle=True)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        val_steps: int = config["val_steps"]
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, steps=val_steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    
    train_data = pd.read_csv('C:/.../train_data_1.csv')
    test_data = pd.read_csv('C:/.../test_data_1.csv')
    vocab_file = pickle.load(open('C:/.../vocab.pck', 'rb'))
    vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
    vocab.update(vocab_file['vocab'])
    
    tokenizer = Tokenizer(num_words=vocab_file['size'], lower=True, oov_token='<oov>') # For those words which are not found in word_index
    tokenizer.fit_on_texts(vocab)
    total_words = vocab_file['size']
    
    input_sequences = []
    for line in train_data['comments']:
        token_list = tokenizer.texts_to_sequences([line])[0]
    
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    test_sequences = []
    for test_line in test_data['comments']:
        test_token_list = tokenizer.texts_to_sequences([test_line])[0]
    
        for j in range(1, len(test_token_list)):
            test_n_gram_sequence = test_token_list[:j+1]
            test_sequences.append(test_n_gram_sequence)
    
    # pad sequences 
    max_sequence_len = 11
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    test_sequences = np.array(pad_sequences(test_sequences, maxlen=max_sequence_len, padding='pre'))

    xs, slabels = input_sequences[:,:-1],input_sequences[:,-1]
    ys = tf.keras.utils.to_categorical(slabels, num_classes=total_words)

    x, labels = test_sequences[:,:-1],test_sequences[:,-1]
    y = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    

   
    # Load and compile Keras model
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=16, input_length=10))
    model.add(RNN(tfa.rnn.LayerNormLSTMCell(64, recurrent_dropout = 0.6), return_sequences=False))
    #model.add(LSTM(64))
    model.add(Dense(10000, activation='softmax'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    client = CifarClient(model, xs, ys, x, y)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )



if __name__ == "__main__":
    main()
