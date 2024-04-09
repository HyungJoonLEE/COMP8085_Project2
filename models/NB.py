import os
import pickle

from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import  LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import nltk

nltk.download('stopwords')

alpha = 0.8
total = 10000


def preprocess_text_data(df):
    # Check and replace non-string entries with "NOT_A_STRING"
    df['text'] = df['text'].apply(lambda x: str(x) if not isinstance(x, str) else x)

    # Filter out empty strings after conversion
    filtered_df = df.loc[df['text'].apply(lambda x: x.strip() != '')]

    # Ensure 'stars' can be integers between 1 and 5; includes checking for float representations of integers
    filtered_df = filtered_df.loc[
        filtered_df['stars'].apply(lambda x: isinstance(x, (int, float)) and x in range(1, 6))]
    filtered_df['text'] = filtered_df['text'].astype(str)
    filtered_df['stars'] = filtered_df['stars'].astype(int)

    # Count the total number of valid rows
    total_valid_rows = len(filtered_df)
    print(total_valid_rows)
    return filtered_df
def prediction( LabelDictionary, labelCount, labelWordsCount, uniqueNumber, labelTotal, inputText):
    list_probably_per_label = []
    for index in range(0,5):
        list_probably_per_label.append(labelCount[index]/labelTotal)

    if not isinstance(inputText, str):
        return 3 # So rare I'm going to ignore it.

    for text in inputText.split():
        for index in range(0,5):
            #skip OOV
            if text in LabelDictionary[index] :
                list_probably_per_label[index] = (LabelDictionary[index][text]/(alpha*uniqueNumber+labelWordsCount[index]))*list_probably_per_label[index]
            else:
                break
    max = list_probably_per_label[0]
    maxIndex = 0
    for index in range(1,5):
        if max < list_probably_per_label[index]:
            max = list_probably_per_label[index]
            maxIndex = index
    return maxIndex+1

def text_linReg(trainning_file, test_file, validation_file, target):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    training_data = pd.read_csv(trainning_file, nrows=50000)
    test_data = pd.read_csv(test_file, nrows=50000)
    validation_data = pd.read_csv(validation_file, nrows=50000)

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    documents = []
    IsOpen = True
    if not os.path.exists('data/gensim2.mdl'):
        yvals = []
        count = 0
        for row in training_data.itertuples():
            try:
                splited_words = row.text.split(" ")
            except:
                print(row)
                continue

            usefulIndex = int(row.useful)
            coolIndex = int(row.cool)
            funnyIndex = int(row.funny)
            starsIndex = int(row.stars)

            yvals.append([usefulIndex, coolIndex, funnyIndex, starsIndex])
            documents.append(row.text)

            count += 1
            if count % 50000 == 0:
                print(count)

        print('Fitting Vectorizer')
        # Tagging documents for Doc2Vec model
        data = [" ".join(text) for text in documents]
        print('Tagging data...')
        tagged_data = [TaggedDocument(words=d.split(), tags=[str(i)]) for i, d in enumerate(data[0:50000])]

        # Train Doc2Vec model
        max_epochs = 10
        vec_size = 40
        alpha = 0.025

        print('Building Model...')
        model = Doc2Vec(vector_size=vec_size, min_count=2, dm=1, epochs=max_epochs, workers=8)

        print('Building Vocab...')
        model.build_vocab(tagged_data)

        print('Training...')
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

        print('Saving...')
        model.save('data/gensim2.mdl')
    else:
        print('Loading doc2vec model...')
        model = Doc2Vec.load('data/gensim2.mdl')

    if not os.path.exists('data/x_train.np'):
        print('Starting Text Embed...')
        tmpList = []
        for i in range(len(documents)):
            tmpList.append(model.infer_vector(simple_preprocess(documents[i])))
        x_train = np.array(tmpList)
        y_train = np.array(yvals)
        print(x_train.shape)
        print(y_train.shape)

        print('Saving Train Data To Disk...')
        np.save('data/x_train.npy', x_train)
        np.save('data/y_train.npy', y_train)
    else:
        print('Loading Train Data From Disk...')
        x_train = np.load('data/x_train.npy')
        y_train = np.load('data/y_train.npy').astype(int)

        print(x_train.shape)
        print(y_train.shape)

    if IsOpen:
        count = 0
        ytestvals = []
        documents = []
        for row in test_data.itertuples():
            try:
                row.text.split(" ")
            except:
                print(row)
                continue
            documents.append(row.text)
            usefulIndex = int(row.useful)
            coolIndex = int(row.cool)
            funnyIndex = int(row.funny)
            starIndex = int(row.stars)

            ytestvals.append([usefulIndex, coolIndex, funnyIndex, starIndex])

            count += 1
            if (count % 50000 == 0):
                print(count)

        print('converting to embedding...')
        tempLst = []
        for i in range(len(documents)):
            tempLst.append(model.infer_vector(simple_preprocess(documents[i])))
            if (i % 10000 == 0):
                print(i)
        x_test = np.array(tempLst)


        y_test = np.array(ytestvals)
        np.save('data/x_test.npy', x_test)
        np.save('data/y_test.npy', y_test)

        print(x_test.shape)
        print(y_test.shape)

    else:
        print('Loading Test Data From Disk...')
        x_test = np.load('data/x_test.npy')
        y_test = np.load('data/y_test.npy').astype(int)
        print(x_test.shape)
        print(y_test.shape)
def stars(trainning_file, test_file, validation_file):
    training_data = pd.read_csv(trainning_file, nrows=total)
    test_data = pd.read_csv(test_file, nrows=total)
    validation_data = pd.read_csv(validation_file, nrows=total)
    warnings.filterwarnings("ignore")

    words = {}
    stars = [{}, {}, {}, {}, {}]
    starslist = [0, 0, 0, 0, 0]
    stars_wordCounts = [0, 0, 0, 0, 0]
    count = 0
    print('Starting Training...')
    for row in training_data.itertuples(index=False):
        try:
            splited_words = row.text.split(" ")
        except:
            print(row)
            continue

        starsIndex = int(float(row.stars)) - 1
        starslist[starsIndex] += 1

        for word in splited_words:
            words[word] = words.get(word, 0) + 1
            stars[starsIndex][word] = stars[starsIndex].get(word, 0) + 1
            stars_wordCounts[starsIndex] += 1

        count += 1
        if count % 10000 == 0:
            print(count)
    uniqueWords = len(words)
    starsTotal = sum(starslist)

    for word in words:
        for row in range(0, 5):
            stars[row][word] = stars[row].get(word, 0) + alpha

    # prediction function
    predictionStars = []

    print('starting Predicitons:')
    count = 0
    for row in test_data.itertuples():
        predictionStars.append(prediction(stars, starslist, stars_wordCounts, uniqueWords, starsTotal, row.text))
        count += 1
        if (count % 10000 == 0):
            print(count)

    starsInd = np.where((test_data['stars'] >= 1) & (test_data['stars'] <= 5))[0]
    filteredStars = test_data[(test_data['stars'] >= 1) & (test_data['stars'] <= 5)]
    print(len(test_data['stars']))

    predictionStars = [predictionStars[i] for i in starsInd]

    test_data['stars'] = pd.to_numeric(test_data['stars'], errors='coerce')
    labels = list(range(1, 6))
    labels_names = [f"{label} stars" for label in range(1, 6)]
    stars_report = classification_report(filteredStars['stars'], predictionStars)
    scm = confusion_matrix(filteredStars['stars'], predictionStars)

    print("=============== Stars with Text Classification based on PPT ==================")
    print(stars_report)
    print(scm)
    print()

    training_data = pd.read_csv(trainning_file, nrows=total)
    test_data = pd.read_csv(test_file, nrows=total)
    validation_data = pd.read_csv(validation_file, nrows=total)
    warnings.filterwarnings("ignore")

    training_data = preprocess_text_data(training_data)
    test_data = preprocess_text_data(test_data)
    X_train = training_data['text']  # Assuming 'text' is the correct column after preprocessing
    Y_train = training_data['stars']  # Assuming 'stars' is your label
    X_test = test_data['text']
    Y_test = test_data['stars']

    # Feature extraction

    count_vectorizer = CountVectorizer(ngram_range=(1, 1))
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)

    # Model training
    mnb = MultinomialNB(alpha=0.80)
    mnb.fit(X_train_counts, Y_train)
    feature_log_prob = mnb.feature_log_prob_
    pred = mnb.predict(X_test_counts)

    report = classification_report(Y_test, pred)
    scm = confusion_matrix(Y_test, pred)
    print("=============== Stars with MultinomialNB ==================")
    print(report)
    print(scm)


    with open('MultinomialNB_stars_model.pkl', 'wb') as model_file:
        pickle.dump(mnb, model_file)

    with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(count_vectorizer, vectorizer_file)
    n_top_features = 10
    feature_names = count_vectorizer.get_feature_names_out()
    feature_log_prob = mnb.feature_log_prob_

    log_probabilities = mnb.feature_log_prob_[0]
    feature_importance = zip(feature_names, log_probabilities)
    sorted_features_by_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features_by_importance[:10]:  # Top 10 features
        print(f"{feature}: {importance}")
    # Plotting
    # Generate the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_features_by_importance, y=feature_names)
    plt.xlabel('Log Probability')
    plt.ylabel('Feature')
    plt.title('Top Features by Log Probability in MultinomialNB Model')
    plt.show()


    plt.figure(figsize=(8, 6))
    sns.heatmap(scm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    # Saving the model and vectorizer


def funny_useful_cool(trainning_file, test_file, validation_file, target):
    features = ['cool', 'funny', 'useful']  # All potential columns
    for target in features:
        training_data = pd.read_csv(trainning_file, nrows=total)
        test_data = pd.read_csv(test_file, nrows=total)
        validation_data = pd.read_csv(validation_file, nrows=total)

        training_data = preprocess_text_data(training_data)
        test_data = preprocess_text_data(test_data)


        # Feature extraction
        tfidf_vectorizer = TfidfVectorizer()
        X_train = tfidf_vectorizer.fit_transform(training_data['text'])
        X_test = tfidf_vectorizer.transform(test_data['text'])

        # Preparing the labels, assuming labels are in a column named as the value of 'target'
        y_train = training_data[target].clip(lower=1, upper=5).values
        Y_test = test_data[target].clip(lower=1, upper=5).values

        # Model training
        logisticReg = LogisticRegression(max_iter=1000)
        logisticReg.fit(X_train, y_train)

        # Prediction
        y_pred = logisticReg.predict(X_test)

        # Evaluation
        report = classification_report(Y_test, y_pred)
        scm = confusion_matrix(Y_test, y_pred)  # Use Y_test for consistency

        print(f"=============== {target} with LogisticRegression ==================")
        print(report)
        print(scm)

        # Saving the model and vectorizer
        with open(f'logistic_regression_{target}.pkl', 'wb') as file:
            pickle.dump(logisticReg, file)

        with open(f'TfidfVectorizer_{target}.pkl', 'wb') as TfidfVectorizer_file:
            pickle.dump(tfidf_vectorizer, TfidfVectorizer_file)

        feature_names = tfidf_vectorizer.get_feature_names_out()
        coefficients = logisticReg.coef_[0]  # For binary classification or one vs rest

        # Getting the absolute values of coefficients for ranking
        sorted_indices = np.argsort(np.abs(coefficients))[::-1]

        # Selecting the top n features/words
        n_top_features = 10
        top_feature_names = feature_names[sorted_indices[:n_top_features]]
        top_coefficients = coefficients[sorted_indices[:n_top_features]]

        # Printing out the top words/features and their coefficients
        print("Top words/features and their coefficients:")
        for feature, coef in zip(top_feature_names, top_coefficients):
            print(f"{feature}: {coef}")

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_coefficients, y=top_feature_names)
        plt.xlabel('Coefficient Value')
        plt.ylabel('Top Features (Words)')
        plt.title('Top Features by Coefficient Value in Logistic Regression Model')
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.heatmap(scm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

def use_stars_model(file_name, target):
    with open('MultinomialNB_stars_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)


    with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    print("recreating the stars summary.........")
    total = None
    test_data = pd.read_csv('./data/test_data.csv', nrows=total)
    test_data = preprocess_text_data(test_data)

    new_data_transformed = loaded_vectorizer.transform(test_data['text'])

    new_predictions = loaded_model.predict(new_data_transformed)
    Y_test = test_data['stars'].clip(lower=1, upper=5)

    report = classification_report(Y_test, new_predictions)
    scm = confusion_matrix(Y_test, new_predictions)

    print("=============== Stars with MultinomialNB ==================")
    print(report)
    print(scm)


def use_other_regression_models(file_name, target):
    features = ['cool', 'funny', 'useful']  # All potential columns
    for target in features:
        print(f"staring to analyze {target}")
        with open(f'logistic_regression_{target}.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        with open(f'TfidfVectorizer_{target}.pkl', 'rb') as file:
            loaded_vectorizer = pickle.load(file)

        # Assuming you have new test data loaded into `new_test_data`
        new_test_data = pd.read_csv('data/test_data.csv')
        new_test_data = preprocess_text_data(new_test_data)

        # Transform the new test data
        X_new_test = loaded_vectorizer.transform(new_test_data['text'])

        # Make predictions
        new_predictions = loaded_model.predict(X_new_test)

        # If you have the true labels for this new test data
        Y_new_test = new_test_data[target].clip(lower=1, upper=5).values  # Adjust as necessary

        # Evaluate the predictions
        new_report = classification_report(Y_new_test, new_predictions)
        new_scm = confusion_matrix(Y_new_test, new_predictions)

        print(f"=============== {target} with LogisticRegression ==================")
        print(new_report)
        print(new_scm)

def create_train_model(trainning_file, test_file, validation_file, target):
    if target == stars:
        stars(trainning_file, test_file, validation_file)
    else:
        funny_useful_cool(trainning_file, test_file, validation_file, target)
    #text_linReg(trainning_file, test_file, validation_file, target)
def use_train_model(file_name, target):
    if 'stars' in file_name:
      use_stars_model(file_name, target)
    else:
      use_other_regression_models(file_name, target)