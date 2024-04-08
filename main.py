import sys
import argparse
import os
from scripts import process
from models import NB1
from models import model_bert
import pandas as pd


# TODO: You need to add json file to ./data directory
input_file_path = './data/yelp_academic_dataset_review.json'
output_file_path = './data/yelp_review_untouched.csv'
training_data = './data/train_data.csv'
test_data = './data/test_data.csv'
validate_data = './data/validate_data.csv'


args_create_hashmap = {
    'NB': NB1.naive_bay,
    'BERT': model_bert.create_train_model,
}

args_use_hashmap = {
    'BERT': model_bert.use_trained_model,
}


def main():
    if not os.path.exists(output_file_path):
        # nltk.download('averaged_perceptron_tagger')

        # This is for cleaning unused attribute in data field
        process.clean_json(input_file_path, output_file_path)

        # Data split function for train, test, and validate
        process.split_data(output_file_path)
    else:
        """
        *** model name is on hashmap ***
        Usage: main.py {file name}.csv --create {model name} --{attribute}
        Example: 
            1) create (TRAINING model)
            main.py yelp_review_untouched.csv --create NB --stars
            main.py yelp_review_untouched.csv --create BERT --funny

            2) use (TRAINED model)
            main.py test_data.csv --use BERT --cool
        """
        # TODO: When we are ready we will comment out lines below
        args = get_args(sys.argv[1:]).parse_args()
        # Accessing the test_set path
        print(f"Test set path: {args.test_set}")

        # Initialize a variable to hold the selected review attribute
        review_attribute = None
        if args.stars:
            review_attribute = "stars"
        elif args.funny:
            review_attribute = "funny"
        elif args.useful:
            review_attribute = "useful"
        elif args.cool:
            review_attribute = "cool"

        # Handling --create or --use with the model name
        if args.create:
            print(f"Action: Create new model using '{args.create}'")
            create_function_by_key(args.create, training_data, test_data,
                                validate_data, review_attribute)
        elif args.use:
            print(f"Action: Use existing model named '{args.use}'")
            # TODO: ADD Trained Model for each member
            # model_bert.live_star_prediction(f"{args.use}", "This is so good")
            use_function_by_key(f"{args.test_set}", review_attribute)




        # df = pd.read_csv(output_file_path)
        # process.plot_labels(df, 'cool')


def create_function_by_key(key, training_data, test_data, validate_data,
                          target):
    if key in args_create_hashmap:
        function_to_run = args_create_hashmap[key]
        function_to_run(training_data, test_data, validate_data, target)
    else:
        print(f"No function found for key: {key}")


def use_function_by_key(key, test_data, target):
    if key in args_use_hashmap:
        function_to_run = args_use_hashmap[key]
        function_to_run(training_data, test_data, validate_data, target)
    else:
        print(f"No function found for key: {key}")


def get_args(args):
    parser = argparse.ArgumentParser(description="COMP8085 Project 2")

    parser.add_argument('test_set',
                        type=str,
                        help='Path to the test set in CSV format.')

    # Mutually exclusive group for creating or using a model
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--create', metavar='MODEL_NAME',
                             help='Create a new model with the specified name.')
    model_group.add_argument('--use', metavar='MODEL_NAME',
                             help='Use an existing model with the specified name.')

    # Mutually exclusive group for review attributes
    review_group = parser.add_mutually_exclusive_group(required=True)
    review_group.add_argument('--stars', action='store_true',
                              help='Select "stars" as the review attribute.')
    review_group.add_argument('--funny', action='store_true',
                              help='Select "funny" as the review attribute.')
    review_group.add_argument('--useful', action='store_true',
                              help='Select "useful" as the review attribute.')
    review_group.add_argument('--cool', action='store_true',
                              help='Select "cool" as the review attribute.')

    return parser


if __name__ == "__main__":
    main()
