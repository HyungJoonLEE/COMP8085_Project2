from scripts import process
import nltk
import os

# TODO: You need to add json file to ./data directory
input_file_path = './data/yelp_academic_dataset_review.json'
output_file_path = './data/yelp_review.csv'


def main():
    if not os.path.exists(output_file_path):
        nltk.download('averaged_perceptron_tagger')
        process.clean_json(input_file_path, output_file_path)
    else:
        print("Project start from here")


if __name__ == "__main__":
    main()