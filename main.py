import argparse
import json
import sys

import nltk
import numpy as np
from scripts import process
import sys
import os
import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os
from models import NB


# TODO: You need to add json file to ./data directory
input_file_path = './data/yelp_academic_dataset_review.json'
output_file_path = './data/yelp_review.csv'
output_file_path1 = './data/yelp_review_1.csv'

def get_args(args):
    parser = argparse.ArgumentParser(description="COMP8085 Project 1")
    task_group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('test_set',
                        type=str,
                        help='Path to held-out test set, in CSV format')
    parser.add_argument('classification_method',
                        type=str,
                        help='Selected classification method name')

    task_group.add_argument('--text',
                            dest='task',
                            action='store_const',
                            const='Label',
                            help='Predict whether the packet is normal or not')


    return parser


function_hashmap = {
    'NB': NB.naive_bay,

}

def run_function_by_key(key, training_data, test_data, validate_data, target):
    if key in function_hashmap:
        function_to_run = function_hashmap[key]
        function_to_run(training_data, test_data, validate_data, target)
    else:
        print(f"No function found for key: {key}")
def main():
    if not os.path.exists(output_file_path):
        nltk.download('averaged_perceptron_tagger')
        process.clean_json(input_file_path, output_file_path)
    else:
        #args
        args = get_args(sys.argv[1:]).parse_args()
        # Read csv using pandas in Latin mode


        # Process data - change values of ports and null + factorize


        if not os.path.exists(output_file_path1):
            df = pd.read_csv(output_file_path,
                             encoding='ISO-8859-1',
                             low_memory=False, nrows=80000)

            # Filtering rows: keep rows where all specified columns have values between 1 and 5 (inclusive)
            filtered_df = df[(df[['useful', 'funny', 'cool', 'stars']] >= 1).all(axis=1) &
                             (df[['useful', 'funny', 'cool', 'stars']] <= 5).all(axis=1)]

            # Save the filtered data to a new CSV file
            filtered_df.to_csv(output_file_path1, index=False)
            #df = pd.read_csv(output_file_path1, encoding='ISO-8859-1', low_memory=False,nrows=80000)

        df = pd.read_csv(output_file_path1, encoding='ISO-8859-1', low_memory=False, nrows=80000)
        train_df, validate_test_df = train_test_split(df,
                                                      train_size=0.7,
                                                      shuffle=True,
                                                      random_state=32)

        validate_df, test_df = train_test_split(validate_test_df,
                                                train_size=0.5,
                                                shuffle=True,
                                                random_state=34)
        # train_df = train_df.to_csv('train.csv', index=False)
        # validate_df = validate_df.to_csv('validate.csv', index=False)
        # test_df = test_df.to_csv('test_df.csv', index=False)
        # df = pd.read_csv('./train.csv', encoding='r', low_memory=False)
        # df = pd.read_csv('./validate.csv', encoding='r', low_memory=False)
        # df = pd.read_csv('./test_df.csv', encoding='r1', low_memory=False)


        run_function_by_key(args.classification_method, train_df, test_df, validate_df, args.task)

if __name__ == "__main__":
    main()