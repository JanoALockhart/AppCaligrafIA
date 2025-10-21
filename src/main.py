import argparse
from line_splitter import LineSplitter
import settings

def main():
    parser = argparse.ArgumentParser(description="Application to take pictures of handwritten letters (A..Z) from students and recommend which ones to practice more")
    parser.add_argument("--model", required=True, type=str, help=f"The file name of the .keras model saved in {settings.PRODUCTION_MODEL_FOLDER}")
    parser.add_argument("--file", required=True, type=str, help=f"The file name of the image of the student letters saved in {settings.PRODUCTION_MODEL_FOLDER}")
    args = parser.parse_args()
    # load image

    img_path = f"{settings.PRODUCTION_MODEL_FOLDER}{args.file}"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    splitter = LineSplitter(img_path, letters)

    # count letters 

    # create model
    #model_processing(args)
    
    
    # load weights
    # evaluate

    # compare results
    # recommend letters

    pass

def model_processing(args):
    import keras
    model_path = f"{settings.PRODUCTION_MODEL_FOLDER}{args.model}"
    print(model_path)
    model = keras.models.load_model(filepath=model_path, compile=False)

if __name__ == "__main__":
    main()