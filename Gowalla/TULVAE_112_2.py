# -*- coding: UTF-8 -*-
"""
Refactored for TensorFlow 2.x
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from itertools import chain

# Parameters
batch_size = 64
iter_num = 20
n_input = 250  # embedding size
n_hidden = 300  # VAE embeddings
c_hidden = 512  # classifier embedding
beta = 0.5
z_size = 50
label_size = 112
learning_rate = 0.001

# Dataset placeholders (refactored for dynamic input)
keep_prob = 0.5

# Define weight and bias dictionary
weights_de = {
    "w_": tf.Variable(tf.random.normal([z_size, n_hidden], mean=0.0, stddev=0.01)),
    "out": tf.Variable(tf.random.normal([2 * c_hidden, label_size])),
}

biases_de = {
    "b_": tf.Variable(tf.random.normal([n_hidden], mean=0.0, stddev=0.01)),
    "out": tf.Variable(tf.random.normal([label_size])),
}


# Utility Functions
def get_onehot(index):
    x = [0] * label_size
    x[index] = 1
    return x


def extract_character_vocab(total_T):
    special_words = ["<PAD>", "<GO>", "<EOS>"]
    set_words = list(set(chain.from_iterable(total_T)))
    set_words = sorted(set_words)
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [
        sentence + [pad_int] * (max_sentence - len(sentence))
        for sentence in sentence_batch
    ]


def eos_sentence_batch(sentence_batch, eos_in):
    return [sentence + [eos_in] for sentence in sentence_batch]


# Embedding Lookup Table
def dic_em(table_X):
    dic_embeddings = list()
    for key in table_X:
        dic_embeddings.append(table_X[key])
    return tf.constant(dic_embeddings, dtype=tf.float32)


# VAE Encoder
class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=label_size, output_dim=embedding_dim
        )
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True
        )

    def call(self, x, training=False):
        x = self.embedding(x)
        output, h_state, c_state = self.lstm(x, training=training)
        return output, h_state, c_state


# VAE Decoder
class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=label_size, output_dim=embedding_dim
        )
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True
        )
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, x, initial_state, training=False):
        x = self.embedding(x)
        output, h_state, c_state = self.lstm(
            x, initial_state=initial_state, training=training
        )
        output = self.dense(output)
        return output, h_state, c_state


# Classifier
class Classifier(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=False, return_state=True
        )
        self.dense = tf.keras.layers.Dense(output_dim, activation="softmax")

    def call(self, x, training=False):
        output, _, _ = self.lstm(x, training=training)
        logits = self.dense(output)
        return logits


# Training Loop
def train_model():
    encoder = Encoder(embedding_dim=n_input, hidden_dim=n_hidden)
    decoder = Decoder(embedding_dim=n_input, hidden_dim=n_hidden, output_dim=label_size)
    classifier = Classifier(hidden_dim=c_hidden, output_dim=label_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(iter_num):
        print(f"Epoch {epoch + 1}/{iter_num}")
        total_loss = 0

        # Training loop (placeholder logic for batches)
        for step in range(10):  # Replace with actual dataset iteration
            with tf.GradientTape() as tape:
                # Example inputs (replace with real data)
                encoder_input = tf.random.uniform(
                    [batch_size, 10], maxval=label_size, dtype=tf.int32
                )
                decoder_input = tf.random.uniform(
                    [batch_size, 10], maxval=label_size, dtype=tf.int32
                )
                labels = tf.random.uniform(
                    [batch_size], maxval=label_size, dtype=tf.int32
                )

                # Forward pass
                _, h_state, c_state = encoder(encoder_input, training=True)
                decoder_output, _, _ = decoder(
                    decoder_input, initial_state=[h_state, c_state], training=True
                )
                logits = classifier(decoder_output, training=True)

                # Compute loss
                loss = loss_fn(labels, logits)
                total_loss += loss

            # Backward pass and optimization
            grads = tape.gradient(
                loss,
                encoder.trainable_variables
                + decoder.trainable_variables
                + classifier.trainable_variables,
            )
            optimizer.apply_gradients(
                zip(
                    grads,
                    encoder.trainable_variables
                    + decoder.trainable_variables
                    + classifier.trainable_variables,
                )
            )

        print(f"Loss: {total_loss.numpy() / 10:.4f}")


if __name__ == "__main__":
    train_model()

