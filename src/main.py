import argparse
import cv2
import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from line_splitter import LineSplitter
import settings
from collections import Counter

VOCABULARY_LIST = [' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '¤', '°', '²', 'È', 'É', 'à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', '€']
INPUT_IMG_SHAPE = (32, 256, 1)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
decoding_function = keras.layers.StringLookup(vocabulary=VOCABULARY_LIST, oov_token="", invert=True)

def main():
    parser = argparse.ArgumentParser(description="Application to take pictures of handwritten letters (A..Z) from students and recommend which ones to practice more")
    parser.add_argument("--model", required=True, type=str, help=f"The file name of the .keras model saved in {settings.PRODUCTION_MODEL_FOLDER}")
    parser.add_argument("--file", required=True, type=str, help=f"The file name of the image of the student letters saved in {settings.PRODUCTION_MODEL_FOLDER}")
    args = parser.parse_args()

    img_path = f"{settings.PRODUCTION_MODEL_FOLDER}{args.file}"
    
    splitter = LineSplitter(img_path, LETTERS)
    line_images = splitter.get_lines()

    # create model
    model_path = f"{settings.PRODUCTION_MODEL_FOLDER}{args.model}"
    print(VOCABULARY_LIST)
    recomendations = model_processing(model_path, line_images)
    print("RECOMENDATIONS: ", recomendations)

def model_processing(model_path, lines_images):
    model = keras.models.load_model(filepath=model_path, compile=False)
    
    recomendations = {}
    for line in lines_images:#[0:1]:
        img_pre = preprocess_image(line.img)
        img = tf.expand_dims(img_pre, axis=0)
        logits = model.predict(img)
        predicted_string = decode_logits(logits)

        count = Counter(predicted_string)        
        only_chars = predicted_string.replace(" ", "")
        accuracy = count[line.char]/len(only_chars)
        recomendations[line.char] = accuracy

        plt.imshow(img_pre)
        plt.title(f"{line.char}: {predicted_string} - Acc: {accuracy}")
        plt.show()

    sorted_recomendations = sorted(recomendations.items(), key=lambda item: item[1])
    return sorted_recomendations

def decode_logits(logits):
    print(logits.shape)

    input_len = tf.ones(logits.shape[0]) * logits.shape[1]
    top_paths, probabilities = keras.ops.ctc_decode(logits, sequence_lengths=input_len, strategy="greedy")
    y_pred_ctc_decoded = top_paths[0][0]
    pred_string = tf.strings.reduce_join(decoding_function(y_pred_ctc_decoded)).numpy().decode("utf-8")
    return pred_string


def preprocess_image(img):
    start = 130
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
    img = img[:, start:512+start]
    img = tf.convert_to_tensor(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=-1)

    img = 1.0 - img
    img = tf.image.resize_with_pad(img, 32, 256)
    img = 1.0 - img

    img = (img - 0.5) / 0.5

    return img


    



if __name__ == "__main__":
    main()