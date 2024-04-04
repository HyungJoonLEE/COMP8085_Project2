from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
alpha = 0.1
import pandas as pd
def prediction( LabelDictionary, labelCount, labelWordsCount, uniqueNumber, labelTotal, inputText):
    list_probably_per_label = []
    for index in range(0,5):
        list_probably_per_label.append(labelCount[index]/labelTotal)

    if not isinstance(inputText, str):
        return 3# So rare I'm going to ignore it.

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

def fullPred(   stars, starslist, stars_wordCounts, starsTotal,
                useful, usefullist, useful_wordCounts, usefulTotal,
                funny, funnylist, funny_wordCounts, funnyTotal,
                cool, coollist, cool_wordCounts, coolTotal, uniqueWords, txt):
    ypred = []
    ypred.append(prediction(stars, starslist, stars_wordCounts, uniqueWords, starsTotal, txt))
    ypred.append(prediction(useful, usefullist, useful_wordCounts, uniqueWords, usefulTotal, txt))
    ypred.append(prediction(funny, funnylist, funny_wordCounts, uniqueWords, funnyTotal, txt))
    ypred.append(prediction(cool, coollist, cool_wordCounts, uniqueWords, coolTotal, txt))
    return ypred


def naive_bay(training_data, test_data, validation_data, target):
    words = {}
    stars = [{},{},{},{},{}]
    useful = [{},{},{},{},{}]
    cool = [{},{},{},{},{}]
    funny = [{},{},{},{},{}]
    starslist = [0,0,0,0,0]
    usefullist = [0,0,0,0,0]
    coollist = [0,0,0,0,0]
    funnylist = [0,0,0,0,0]
    stars_wordCounts = [0, 0, 0, 0, 0]
    useful_wordCounts = [0, 0, 0, 0, 0]
    cool_wordCounts = [0, 0, 0, 0, 0]
    funny_wordCounts = [0, 0, 0, 0, 0]
    count = 0
    print('Starting Training...')
    for row in training_data.itertuples():
        try:
            splited_words = row.text.split(" ")
        except:
            print(row)
            continue

        starsIndex = int(float(row.stars)) - 1
        starslist[starsIndex] += 1

        usefulIndex = int(row.useful) - 1
        if usefulIndex >= 0 and usefulIndex < 5:
            usefullist[usefulIndex] += 1

        coolIndex = int(row.cool) - 1
        if coolIndex >= 0 and coolIndex < 5:
            coollist[coolIndex] += 1

        funnyIndex = int(row.funny) - 1
        if funnyIndex >= 0 and funnyIndex < 5:
            funnylist[funnyIndex] += 1


        for word in splited_words:
            words[word] = words.get(word, 0) + 1
            stars[starsIndex][word] = stars[starsIndex].get(word,0)+1
            stars_wordCounts[starsIndex] += 1
            if usefulIndex >=0 and usefulIndex<5:
                useful[usefulIndex][word] = useful[usefulIndex].get(word, 0) + 1
                useful_wordCounts[usefulIndex] += 1
            if coolIndex >= 0 and coolIndex < 5:
                cool[coolIndex][word] = cool[coolIndex].get(word, 0) + 1
                cool_wordCounts[coolIndex]  += 1
            if funnyIndex >= 0 and funnyIndex < 5:
                funny[funnyIndex][word] = funny[funnyIndex].get(word, 0) + 1
                funny_wordCounts[funnyIndex] += 1
        count += 1
        if count % 10000 == 0:
            print(count)
    uniqueWords = len(words)
    #demon for prob
    starsTotal = sum(starslist)
    usefulTotal = sum(usefullist)
    funnyTotal = sum(funnylist)
    coolTotal = sum(coollist)
    for word in words:
        for row in range(0,5):
            stars[row][word] = stars[row].get(word,0)+alpha
            useful[row][word] = useful[row].get(word,0)+alpha
            cool[row][word] = cool[row].get(word,0)+alpha
            funny[row][word] = funny[row].get(word,0)+alpha
    #prediction function
    predictionStars = []
    predictionUserful = []
    predictionFunny = []
    predictionCool = []
    predictionFull = []
    print('starting Predicitons:')
    count = 0
    for row in test_data.itertuples():
        predictionStars.append(prediction(stars,starslist, stars_wordCounts, uniqueWords, starsTotal, row.text))
        predictionUserful.append(prediction(useful, usefullist, useful_wordCounts, uniqueWords, usefulTotal, row.text))
        predictionFunny.append(prediction(funny, funnylist, funny_wordCounts, uniqueWords, funnyTotal, row.text))
        predictionCool.append(prediction(cool, coollist, cool_wordCounts, uniqueWords, coolTotal, row.text))
        predictionFull.append(fullPred(stars ,starslist, stars_wordCounts,  starsTotal,
                                       useful, usefullist, useful_wordCounts, usefulTotal,
                                       funny, funnylist, funny_wordCounts, funnyTotal,
                                       cool, coollist, cool_wordCounts,  coolTotal,
                                       uniqueWords, row.text))
        count+=1
        if(count%10000 == 0):
            print(count)

    import numpy as np
    from sklearn.metrics import confusion_matrix

    starsInd = np.where((test_data['stars'] >= 1) & (test_data['stars'] <= 5))[0]
    funnyInd = np.where((test_data['funny'] >= 1) & (test_data['funny'] <= 5))[0]
    UserfulInd = np.where((test_data['useful'] >= 1) & (test_data['useful'] <= 5))[0]
    coolInd = np.where((test_data['cool'] >= 1) & (test_data['cool'] <= 5))[0]
    filteredStars = test_data[(test_data['stars'] >= 1) & (test_data['stars'] <= 5)]
    # predictionStars = predictionStars[starsInd]
    print(len(test_data['stars']))
    print(len(starsInd))
    print(len(funnyInd))
    print(len(UserfulInd))
    print(len(coolInd))

    predictionStars = [predictionStars[i] for i in starsInd]

    filteredUserful = test_data[(test_data['useful'] >= 1) & (test_data['useful'] <= 5)]
    predictionUserful = [predictionUserful[i] for i in UserfulInd]

    filteredcool = test_data[(test_data['cool'] >= 1) & (test_data['cool'] <= 5)]
    predictionCool = [predictionCool[i] for i in coolInd]

    filteredfunny = test_data[(test_data['funny'] >= 1) & (test_data['funny'] <= 5)]
    predictionFunny = [predictionFunny[i] for i in funnyInd]

    test_data['stars'] = pd.to_numeric(test_data['stars'], errors='coerce')
    test_data['useful'] = pd.to_numeric(test_data['useful'], errors='coerce').astype(int)
    test_data['funny'] = pd.to_numeric(test_data['funny'], errors='coerce').astype(int)
    test_data['cool'] = pd.to_numeric(test_data['cool'], errors='coerce').astype(int)

    # Filtering logic
    # mask = (test_data.drop('text', axis=1) >= 1) & (test_data.drop('text', axis=1) <= 5)
    # filtered_rows = mask.all(axis=1)
    # filteredFull = test_data[filtered_rows]
    # filteredFull_report = classification_report(filteredFull, predictionFull)
    # full = (confusion_matrix(filteredFull, predictionFull))

    stars_report = classification_report(filteredStars['stars'], predictionStars)
    scm = confusion_matrix(filteredStars['stars'], predictionStars)

    useful_report = classification_report(filteredUserful['useful'], predictionUserful)
    ucm = (confusion_matrix(filteredUserful['useful'], predictionUserful))

    cool_report = classification_report(filteredcool['cool'], predictionCool)
    ccm = (confusion_matrix(filteredcool['cool'], predictionCool))

    funny_report =classification_report(filteredfunny['funny'], predictionFunny)
    fcm = (confusion_matrix(filteredfunny['funny'], predictionFunny))


    print(stars_report)
    print(scm)
    print()
    print(useful_report)
    print(ucm)
    print()
    print(cool_report)
    print(ccm)
    print()
    print(funny_report)
    print(fcm)

    # X_train = training_data.drop(['text'], axis=1)
    # Y_train = training_data['text']

    # count_vectorizer = CountVectorizer(ngram_range=(1, 1))
    # X_train_counts = count_vectorizer.fit_transform(X_train.astype('U').values)

    # mnb = MultinomialNB(alpha=2.0)
    #
    #
    # # Fit the classifier using X_train and Y_train
    # mnb.fit(X_train, Y_train)
    #
    # # Predict on the training set
    # pred = mnb.predict(X_train)
    #
    # # Generate classification report
    # report = classification_report(Y_train, pred)
    # print(report)
