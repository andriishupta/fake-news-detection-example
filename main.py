import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def detect(headline, test_size=0.2, random_state=50):
    dataset = pd.read_csv("news_dataset.csv")
    dataset.head()
    print(dataset.shape)

    # print(dataset["title"].isnull())
    # print(dataset["text"].isnull())
    # print(dataset["label"].isnull())

    list(dataset.columns)

    # dataset.drop(["Unnamed: 0"], axis=1, inplace=True)

    x = np.array(dataset["title"])
    y = np.array(dataset["label"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state=random_state)
    # Naïve Baye
    model = MultinomialNB()
    model.fit(xtrain, ytrain)
    print(model.score(xtest, ytest))

    data = cv.transform([headline]).toarray()
    print(model.predict(data))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    real_headline = """
        Ukraine: Austrian leader, Putin meet…other new developments
    """
    fake_headline = """
        A lion was found flying in South America
    """
    print("Is real headline real?")
    detect(real_headline)

    print("Is random headline fake?")
    detect(fake_headline)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
