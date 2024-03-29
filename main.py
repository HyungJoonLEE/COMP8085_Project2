from scripts import process
import os
import pandas as pd


# TODO: You need to add json file to ./data directory
input_file_path = './data/yelp_academic_dataset_review.json'
output_file_path = './data/clean_review.json'
csv_file_path = './data/yelp_review.csv'


def main():
    if not os.path.exists(csv_file_path):
        # process.make_cleaned_json(input_file_path, output_file_path)
        process.convert_to_csv(output_file_path)
    else:
        df = pd.read_csv(csv_file_path, low_memory=False)
        df['stars'] = df['stars'].astype(int)
        df['useful'] = df['useful'].astype(int)
        df['funny'] = df['funny'].astype(int)
        df['cool'] = df['cool'].astype(int)
        print(df.dtypes)
        print(df.head(10))
    return 0


if __name__ == "__main__":
    main()