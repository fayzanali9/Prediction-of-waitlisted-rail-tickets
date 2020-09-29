import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
def load_data(file):
    df = pd.read_csv(file)

    # split the data into x and y
    x = df.iloc[:,[1,2,3,4,5,6]].values
    y = df.iloc[:,7].values

    return x, y


def clean_and_split_data(x, y):
    # clean the data
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    # convert the gender into numeric values
    # 1: male, 0: female
    x[:, 0] = encoder.fit_transform(x[:, 0])

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    return x_train, x_test, y_train, y_test


def build_gradientboost_model(x_train, y_train):
    # build the model
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier()
    classifier = classifier.fit(x_train, y_train)
    return classifier

def build_adaboost_model(x_train, y_train):
    # build the model
    from sklearn.ensemble import AdaBoostClassifier
    classifier = AdaBoostClassifier()
    classifier = classifier.fit(x_train, y_train)
    return classifier

def build_randomforest_model(x_train, y_train):
    # build the model
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    classifier = classifier.fit(x_train, y_train)
    return classifier

def build_xgboost_model(x_train, y_train):
    # build the model
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier = classifier.fit(x_train, y_train)
    return classifier


def build_decisiontree_model(x_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(x_train, y_train)
    return  classifier


def build_naive_bayes_model(x_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier = classifier.fit(x_train, y_train)
    return  classifier



def build_svc_model(x_train, y_train):
    from sklearn.svm import SVC
    classifier = SVC()
    classifier = classifier.fit(x_train, y_train)
    return  classifier

# #def save_image(x):
#     x=
#     plt.savefig(f'/home/fayzan/PycharmProjects/myapp/templates{i}')


def cross_validation(algorithm, classifier, x_test, y_test):
    # evaluate the model
    predictions = classifier.predict(x_test)



    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print(f"accuracy of {algorithm} is {accuracy * 100}%")

    colors = ['red' if value ==1 else 'yellow' for value in predictions]
    plt.ylabel('travelClass')
    plt.xlabel('BookingStatus')
    x = plt.scatter(x_test[:,1],y_test[:,],c=colors)
        #save_image(x)
    plt.savefig(f'/home/fayzan/PycharmProjects/myapp/templates')


    plt.show()
    return accuracy

# predict the output


x, y = load_data('final_df.csv')
x_train, x_test, y_train, y_test = clean_and_split_data(x, y)


classifier_nb = build_naive_bayes_model(x_train, y_train)
classifier_svc = build_svc_model(x_train, y_train)
classifier_gb = build_gradientboost_model(x_train, y_train)
classifier_rf = build_randomforest_model(x_train, y_train)
classifier_dt = build_decisiontree_model(x_train, y_train)
classifier_ada = build_adaboost_model(x_train, y_train)
classifier_xg = build_xgboost_model(x_train,y_train)

accuracy_nb = cross_validation('Naive Bayes', classifier_nb, x_test, y_test)
accuracy_svc = cross_validation('SVM', classifier_svc, x_test, y_test)
accuracy_ada = cross_validation('Ada Boost', classifier_ada, x_test, y_test)
accuracy_dt = cross_validation('Decision Tree', classifier_dt, x_test, y_test)
accuracy_rf = cross_validation('Random forest', classifier_rf, x_test, y_test)
accuracy_gb = cross_validation('Gradient Boost', classifier_gb, x_test, y_test)
accuracy_xg = cross_validation('XGBoost',classifier_xg,x_test,y_test)

