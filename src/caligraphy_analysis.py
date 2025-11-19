import base64
from collections import Counter
from io import BytesIO
import cv2
import keras
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import src.line_splitter as line_splitter
import src.settings as settings

def process_image_form(file, selected_model):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    line_images = line_splitter._process_image(img, LETTERS)
    model_path = f"{settings.PRODUCTION_MODEL_FOLDER}{selected_model["filename"]}"

    decoding_function = keras.layers.StringLookup(vocabulary=list(selected_model["vocabulary"]), oov_token="", invert=True)
    recomendations, predictions = model_processing(model_path, line_images, decoding_function)
    
    row_images_b64 = {letter: _convert_to_base64(image) for letter, image in line_images.items()}

    return recomendations, row_images_b64, predictions


def analyze_caligraphy(args):
    VOCABULARY_LIST = list(" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") #TODO: to the manifest
    #VOCABULARY_LIST = [' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '¤', '°', '²', 'È', 'É', 'à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', '€']
    print(VOCABULARY_LIST)
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    decoding_function = keras.layers.StringLookup(vocabulary=VOCABULARY_LIST, oov_token="", invert=True)
    img_path = f"{settings.IMAGES_FOLDER}{args.file}"
    model_path = f"{settings.PRODUCTION_MODEL_FOLDER}{args.model}"

    img = _open_image(img_path)
    line_images = line_splitter._process_image(img, LETTERS)
    recomendations = model_processing(model_path, line_images, decoding_function)
    return recomendations, line_images

def _convert_to_base64(image):
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def preprocess_row_image(img):
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


def decode_logits(logits, decoding_function):
    input_len = tf.ones(logits.shape[0]) * logits.shape[1]
    top_paths, probabilities = keras.ops.ctc_decode(logits, sequence_lengths=input_len, strategy="greedy")
    y_pred_ctc_decoded = top_paths[0][0]
    pred_string = tf.strings.reduce_join(decoding_function(y_pred_ctc_decoded)).numpy().decode("utf-8")
    return pred_string


def model_processing(model_path, lines_images, decoding_function):
    model = keras.models.load_model(filepath=model_path, compile=False)

    recomendations = {}
    predictions = {}

    for letter_line in lines_images.keys():
        img_pre = preprocess_row_image(lines_images[letter_line])
        img = tf.expand_dims(img_pre, axis=0)
        logits = model.predict(img)
        predicted_string = decode_logits(logits, decoding_function)

        print("PRED", predicted_string)

        count = Counter(predicted_string)
        only_chars = predicted_string.replace(" ", "")
        accuracy = count[letter_line]/len(only_chars)
        recomendations[letter_line] = accuracy
        predictions[letter_line] = predicted_string

        
    sorted_recomendations = dict(sorted(recomendations.items(), key=lambda item: item[1]))
    return sorted_recomendations, predictions


def _open_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


