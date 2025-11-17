import argparse
from caligraphy_analysis import analyze_caligraphy
import settings

def main():
    parser = argparse.ArgumentParser(description="Application to take pictures of handwritten letters (A..Z) from students and recommend which ones to practice more")
    parser.add_argument("--model", required=True, type=str, help=f"The file name of the .keras model saved in {settings.PRODUCTION_MODEL_FOLDER}")
    parser.add_argument("--file", required=True, type=str, help=f"The file name of the image of the student letters saved in {settings.IMAGES_FOLDER}")
    args = parser.parse_args()

    recomendations = analyze_caligraphy(args)
    print("RECOMENDATIONS: ", recomendations)


if __name__ == "__main__":
    main()