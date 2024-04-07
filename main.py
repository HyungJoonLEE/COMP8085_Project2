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
from models import NB1
import pandas as pd
from sklearn.model_selection import train_test_split


# TODO: You need to add json file to ./data directory
input_file_path = './data/yelp_academic_dataset_review.json'
output_file_path = './data/yelp_review_untouched.csv'


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
    'NB': NB1.naive_bay,

}


def run_function_by_key(key, training_data, test_data, validate_data, target):
    if key in function_hashmap:
        function_to_run = function_hashmap[key]
        function_to_run(training_data, test_data, validate_data, target)
    else:
        print(f"No function found for key: {key}")


def main():
    if not os.path.exists(output_file_path):
        # nltk.download('averaged_perceptron_tagger')
        process.clean_json(input_file_path, output_file_path)
    else:
        df = pd.read_csv(output_file_path, low_memory=False)
        df_subset = df.sample(n=100000, random_state=44)
        df_subset.to_csv('./data/yelp_review_100000.csv', index=False)

if __name__ == "__main__":
    main()
