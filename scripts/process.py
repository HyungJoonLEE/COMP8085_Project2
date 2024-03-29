import json
import pandas as pd


def make_cleaned_json(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r') as input_file:
            reviews = []
            for line in input_file:
                review = json.loads(line)
                if having_attr(review):
                    # Remove new line feed
                    if 'text' in review:
                        review['text'] = review['text'].replace('\n', '')

                    # Remove key-value pair
                    review.pop('review_id', None)       # Remove 'review_id'
                    review.pop('user_id', None)         # Remove 'user_id'
                    review.pop('business_id', None)     # Remove 'business_id'
                    review.pop('date', None)            # Remove 'data'
                    reviews.append(review)
                else:
                    break

        # Wrap the list into a new dictionary under the key "reviews"
        data_to_save = {"reviews": reviews}

        with open(output_file_path, 'w') as output_file:
            json.dump(data_to_save, output_file, indent=4)

        print("File converted and key-value pairs removed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def having_attr(json_review):
    important_attr = ['stars', 'useful', 'funny', 'cool', 'text']
    for attr in important_attr:
        if attr not in json_review:
            print(f"{attr} does not exist in the json line\n{json_review}")
            return False
        return True


def convert_to_csv(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Convert the list of reviews to a DataFrame
    df = pd.DataFrame(data['reviews'])

    # Fill NaN values with 0 or another appropriate placeholder before type conversion
    # df.fillna(0, inplace=True)
    # Make rate fields to integer
    # df['stars'] = pd.DataFrame(data['reviews'])['stars'].astype(int)
    # df['useful'] = pd.DataFrame(data['reviews'])['useful'].astype(int)
    # df['funny'] = pd.DataFrame(data['reviews'])['funny'].astype(int)
    # df['cool'] = pd.DataFrame(data['reviews'])['cool'].astype(int)

    output_csv_path = './data/yelp_review.csv'
    df.to_csv(output_csv_path, index=False)

    print("CSV file has been created successfully.")
