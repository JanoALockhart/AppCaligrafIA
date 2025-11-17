import argparse
import keras
import line_splitter
from caligraphy_analysis import model_processing
import settings

#VOCABULARY_LIST = [' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '¤', '°', '²', 'È', 'É', 'à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', '€']
VOCABULARY_LIST = list(" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
decoding_function = keras.layers.StringLookup(vocabulary=VOCABULARY_LIST, oov_token="", invert=True)

def main():
    parser = argparse.ArgumentParser(description="Application to take pictures of handwritten letters (A..Z) from students and recommend which ones to practice more")
    parser.add_argument("--model", required=True, type=str, help=f"The file name of the .keras model saved in {settings.PRODUCTION_MODEL_FOLDER}")
    parser.add_argument("--file", required=True, type=str, help=f"The file name of the image of the student letters saved in {settings.IMAGES_FOLDER}")
    args = parser.parse_args()

    img_path = f"{settings.IMAGES_FOLDER}{args.file}"
    
    line_images = line_splitter._process_image(img_path, LETTERS)

    # create model
    model_path = f"{settings.PRODUCTION_MODEL_FOLDER}{args.model}"
    print(VOCABULARY_LIST)
    recomendations = model_processing(model_path, line_images)
    print("RECOMENDATIONS: ", recomendations)

if __name__ == "__main__":
    main()