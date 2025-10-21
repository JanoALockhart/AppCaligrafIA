import argparse
import keras
import tensorflow as tf
from line_splitter import LineSplitter
import settings

VOCABULARY_LIST = [' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '¤', '°', '²', 'È', 'É', 'à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', '€']
INPUT_IMG_SHAPE = (32, 256, 1)

def main():
    parser = argparse.ArgumentParser(description="Application to take pictures of handwritten letters (A..Z) from students and recommend which ones to practice more")
    parser.add_argument("--model", required=True, type=str, help=f"The file name of the .keras model saved in {settings.PRODUCTION_MODEL_FOLDER}")
    parser.add_argument("--file", required=True, type=str, help=f"The file name of the image of the student letters saved in {settings.PRODUCTION_MODEL_FOLDER}")
    args = parser.parse_args()

    img_path = f"{settings.PRODUCTION_MODEL_FOLDER}{args.file}"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    splitter = LineSplitter(img_path, letters)
    line_images = splitter.get_lines()

    # create model
    model_path = f"{settings.PRODUCTION_MODEL_FOLDER}{args.model}"
    print(VOCABULARY_LIST)
    #model_processing(model_path)
    
    
    # load weights
    # evaluate

    # compare results
    # recommend letters

    pass

def model_processing(model_path, lines_images):
    model = keras.models.load_model(filepath=model_path, compile=False)
    
    for lines in lines_images:
        img = preprocess_image(lines.img)
        logits = model.predict(img)
        predicted_string = decode_logits(logits)
        print(predicted_string)

def decode_logits(logits):
    #TODO
    pass

def preprocess_image(img):
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