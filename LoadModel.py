import os
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import numpy as np
import pickle
from google.colab import files
from PIL import Image
import random

trained_model_path = ["https://storage.googleapis.com/bucket-1-free/checkpoint", 
                      "https://storage.googleapis.com/bucket-1-free/tem/checkpoints/train/ckpt-10.data-00000-of-00001",
                      "https://storage.googleapis.com/bucket-1-free/tem/checkpoints/train/ckpt-10.index",
                      "https://storage.googleapis.com/bucket-1-free/tem/checkpoints/train/train_captions.data"]

random.seed(4)
top_k = 20000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
attention_features_shape = 64
checkpoint_path = "/checkpoints/train"

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def get_image_path(defualt_url=
                   'https://raw.githubusercontent.com/eniktab/MoE_nlp/main/1.jpg'):
    """ upload an image on the hard disk or get a file from url"""

    image_url = files.upload()
    # image_url= input("Set your image path: you can upload it at this website " \
    #                "\n get the link from --->>> https://imgur.com/upload \n " \
    #                "then copy past the link in this box --->>>  ")
    image_path = str(*image_url.keys())
    if len(image_path) < 1:
        print("Using the defualt image. Thank you Philip for your contribution!")
        image_path = defualt_url
        image_path = tf.keras.utils.get_file(defualt_url.split(r"/")[0],
                                             origin=defualt_url)
    return image_path


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def evaluate(image, encoder, decoder, max_length, tokenizer):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def main(image_to_caption):
    if not os.path.exists('/checkpoints/train/'):
        os.makedirs('/checkpoints/train/')

    for i in trained_model_path:
        tf.keras.utils.get_file(
            i.split("/")[-1], i, cache_subdir=os.path.abspath(checkpoint_path))

    with open("/checkpoints/train/train_captions", 'rb') as pickle_file:
        train_captions = pickle.load(pickle_file)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    max_length = calc_max_length(train_seqs)
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    image_path = image_to_caption
    result, attention_plot = evaluate(image_path, encoder, decoder, max_length, tokenizer)
    plot_attention(image_path, result, attention_plot)
    result = " ".join(result).replace(' <end>', ".")
    print('\n\n\n Predicted Caption: \n {} \n\n\n'.format(result))


if __name__ == "__main__":
    image_to_caption = str(sys.argv[1])
    main(image_to_caption)
