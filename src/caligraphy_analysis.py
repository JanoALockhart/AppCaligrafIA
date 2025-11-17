from collections import Counter
import cv2
import keras
from matplotlib import pyplot as plt
import tensorflow as tf

from main import decoding_function


def preprocess_image(img):
    start = 84
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
    #img = img[:, start:]
    img = tf.convert_to_tensor(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=-1)

    img = 1.0 - img
    img = tf.image.resize_with_pad(img, 32, 512)
    img = 1.0 - img

    img = (img - 0.5) / 0.5

    return img


def decode_logits(logits):
    print(logits.shape)

    input_len = tf.ones(logits.shape[0]) * logits.shape[1]
    top_paths, probabilities = keras.ops.ctc_decode(logits, sequence_lengths=input_len, strategy="greedy")
    y_pred_ctc_decoded = top_paths[0][0]
    pred_string = tf.strings.reduce_join(decoding_function(y_pred_ctc_decoded)).numpy().decode("utf-8")
    return pred_string


def model_processing(model_path, lines_images):
    model = keras.models.load_model(filepath=model_path, compile=False)

    recomendations = {}

    for letter_line in lines_images.keys():
        img_pre = preprocess_image(lines_images[letter_line])
        img = tf.expand_dims(img_pre, axis=0)
        logits = model.predict(img)
        predicted_string = decode_logits(logits)

        print("PRED", predicted_string)

        count = Counter(predicted_string)
        only_chars = predicted_string.replace(" ", "")
        accuracy = count[letter_line]/len(only_chars)
        recomendations[letter_line] = accuracy

        plt.imshow(img_pre)
        plt.title(f"{letter_line}: {predicted_string} - Acc: {accuracy}")
        plt.show()

    sorted_recomendations = sorted(recomendations.items(), key=lambda item: item[1])
    return sorted_recomendations