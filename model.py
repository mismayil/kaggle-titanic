import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read train data
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# Explatory data analysis
train_data.head()
train_data.shape
train_data.ndim
train_data.columns
train_data.info()
train_data.describe()

# Data visualization

# By plotting a count plot of passenger class with respect to Survived, we can see that class is highly correlated with the survival
# (most of first class passengers survived)
sns.countplot(x='Pclass', hue='Survived', data=train_data)

# Sex vs Survived shows larger portion of females survived compared to males
sns.countplot(x='Sex', hue='Survived', data=train_data)
sns.barplot(x='Age', y='Sex', hue='Survived', data=train_data)
sns.countplot(x='Embarked', hue='Survived', data=train_data)
sns.countplot(x='SibSp', hue='Survived', data=train_data)
sns.countplot(x='Parch', hue='Survived', data=train_data)

# Data cleaning

def get_nan(data, feature):
    return data[data[feature].isnull()]

def print_nan(data):
    for feature in data.columns:
        print('{0}: {1}'.format(feature, len(get_nan(data, feature))))

print('NaN values before cleaning')
# As we can see from the results, we have some missing data for Age, Cabin and Embarked.
print_nan(train_data)

titlemap = {
    'Ms': 'Ms',
    'Rev': 'Officer',
    'Mme': 'Mrs',
    'Capt': 'Officer',
    'Don': 'Royalty',
    'Col': 'Officer',
    'Mr': 'Mr',
    'Sir': 'Royalty',
    'Lady': 'Royalty',
    'the Countess': 'Royalty',
    'Jonkheer': 'Royalty',
    'Master': 'Royalty',
    'Mlle': 'Ms',
    'Major': 'Officer',
    'Mrs': 'Mrs',
    'Dr': 'Officer',
    'Miss': 'Ms',
    'Dona': 'Royalty'
}

def get_title(name):
    names = name.split(',')
    if len(names) > 1: return names[1].split('.')[0].strip()
    return name

def get_titles():
    titles = set()
    for name in train_data.Name:
        titles.add(get_title(name))

    return titles

def process_name(data):
    data['Title'] = data['Name'].map(lambda name: titlemap[get_title(name)])
    title_dummies = pd.get_dummies(data['Title'], prefix='Title')
    data = pd.concat([data, title_dummies], axis=1)
    data.drop(['Name', 'Title'], axis=1, inplace=True)
    return data

def process_sex(data):
    return data.replace({'male': 0, 'female': 1})

def hash_ticket(tickets):
    ticketstr = ''.join(tickets[:-1 if tickets[-1].isnumeric() else len(tickets)])
    num = 0
    for ch in ticketstr:
        num += ord(ch)

    return str(num + int(tickets[-1]) if tickets[-1].isnumeric() else 0)

def process_ticket(data):
    data['Ticket'] = data['Ticket'].map(lambda ticket: ticket if ticket.isnumeric() else hash_ticket(ticket.split(' ')))
    return data

# Cabin has NaN for most of the rows, we will populate it with U and then do one-hot encoding
def process_cabin(data):
    data['Cabin'].fillna('U', inplace=True)
    data['Cabin'] = data['Cabin'].map(lambda cabin: cabin[0])
    cabin_dummies = pd.get_dummies(data['Cabin'], prefix='Cabin')
    data = pd.concat([data, cabin_dummies], axis=1)
    data.drop('Cabin', axis=1, inplace=True)
    return data

# For Embarked, we have 2 missing values, lets just replace them with most common embarking point which is S
def process_embarked(data):
    counts = data.Embarked.value_counts()
    data['Embarked'].fillna(counts[counts == counts.max()].index.values[0], inplace=True)
    embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = pd.concat([data, embarked_dummies], axis=1)
    data.drop('Embarked', axis=1, inplace=True)
    return data

# Age is an important feature as we can tell from the graphs. So it would be better to keep it.
# We can replace it with the mean/median of the age, but that would be problematic as ages might differ depending on people's classes
# Therefore, best way would be to assign the age based on Pclass and Sex of the person

def get_age(row, data):
    condition = (data['Pclass'] == row['Pclass']) & (data['Sex'] == row['Sex'])

    if np.isnan(row['Age']):
        return data[condition]['Age'].values[0]
    else:
        return row['Age']

def process_age(data):
    grouped_data = data.groupby(['Sex', 'Pclass'])
    grouped_data_median = grouped_data.median().reset_index()

    data['Age'] = data.apply(lambda row: get_age(row, grouped_data_median), axis=1)

    return data

# Let's look at Fare distribution with respect to Survived
# Since fare is a continuous variable, we can group it in ranges
# First, we find its median and create ranges around that. (median is 14.)
def get_range(fare):
    if fare < 5: return '0-5'
    if fare < 10: return '5-10'
    if fare < 15: return '10-15'
    if fare < 100: return '15-100'
    if fare < 200: return '100-200'
    if fare < 300: return '200-300'
    if fare < 400: return '300-400'
    return '400'

def process_fare(data):
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    return data
#     train_data['FareRange'] = train_data['Fare'].map(get_range)
#     We see people who paid had better chance of surviving
#     sns.countplot(x='FareRange', hue='Survived', data=train_data)
#     fare_dummies = pd.get_dummies(train_data['FareRange'], prefix='FareRange')
#     train_data = pd.concat([train_data, fare_dummies], axis=1)
#     train_data.drop('FareRange', axis=1, inplace=True)

def process_family(data):
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    data['Single'] = data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    data['SmallFamily'] = data['FamilySize'].map(lambda s: 1 if s < 3 else 0)
    data['LargeFamily'] = data['FamilySize'].map(lambda s: 1 if s > 2 else 0)
    return data


# Let's build our first model based on features that we think are important to the model
# we will use Gaussian Naive Bayes model for this
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def process_data(data):
    data = process_name(data)
    data = process_sex(data)
    data = process_age(data)
    data = process_ticket(data)
    data = process_cabin(data)
    data = process_embarked(data)
    data = process_fare(data)
    data = process_family(data)
    return data

def prepare_data(data):
    data = process_data(data)
    feature_data = data.drop('Survived', axis=1)
    label_data = data.Survived
    return train_test_split(feature_data, label_data, test_size=0.3, random_state=42)

def run_model(model, features_train, features_test, labels_train, labels_test):
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    accuracy = accuracy_score(labels_test, predictions)
    report = classification_report(labels_test, predictions)
    confusion = confusion_matrix(labels_test, predictions)
    return (accuracy, report, confusion)

features_train, features_test, labels_train, labels_test = prepare_data(train_data)

# We get 64% accuracy on this model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
results = run_model(model, features_train, features_test, labels_train, labels_test)
for score in results: print(score)

# Now let's use Random Forest Classifier and we get 83% accuracy
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=180, min_samples_split=4)
results = run_model(model, features_train, features_test, labels_train, labels_test)
for score in results: print(score)

print(model.feature_importances_)
importance_map = list(zip(model.feature_importances_, features_train.columns))
# This shows that most important features are Ticket, Sex, Title_Mr, Fare, Age, Pclass, FamilySize, Title_Mrs etc.
importance_map.sort(key=lambda imp: imp[0])
print(importance_map)

# Prepare for final submission
test_data = process_data(test_data)

# print_nan(test_data)
# fill in Cabin_T as it is missing from test_data
test_data['Cabin_T'] = 0
print(set(features_train.columns)-set(test_data.columns))

predictions = model.predict(test_data)

print(len(predictions))

test_data = pd.read_csv('./data/test.csv')
submission = pd.DataFrame()
submission['PassengerId'] = test_data['PassengerId']
submission['Survived'] = predictions
submission[['PassengerId','Survived']].to_csv('titanic_submission.csv', index=False)
