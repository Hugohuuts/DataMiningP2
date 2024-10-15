import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def main():     
    #Hugo Voorheijen, 1706276
    #Hiddo Laane, 6128556
    #Pepijn Briene, 6788106
    print("Start")  
    train_df, test_df = import_data()
    #print(train_df.head())

     
    print("Multinomial_Naive_Bayes Unigrams: ")    
    #pas corssvalidation toe om de juiste hyperparameters te vinden (unigrams)
    min_df, alpha = CV_naive_bayes(train_df, 1)
    Multinomial_Naive_Bayes(train_df,test_df, 1, 0.005, 0.5)
    
    
    print("Multinomial_Naive_Bayes Bigrams: ")
    #pas corssvalidation toe om de juiste hyperparameters te vinden (bigrams)
    min_df, alpha = CV_naive_bayes(train_df, 2)
    Multinomial_Naive_Bayes(train_df,test_df, 2, min_df, alpha)

    print("Logistic_Regression_Lasso Unigrams: ")
    min_df, C = CV_logistic_regression_lasso(train_df, 1)    
    Logistic_Regression_Lasso(train_df,test_df, 1, min_df, C)
    
    print("Logistic_Regression_Lasso Bigrams: ")
    min_df, C = CV_logistic_regression_lasso(train_df, 2)    
    Logistic_Regression_Lasso(train_df,test_df, 2, min_df, C)


    # Classification_Tree(train_df,test_df, 2)
    # Random_Forest(train_df,test_df, 2)
    
    
    print("End")

def CV_naive_bayes(train_data, n_gram):
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, n_gram))),
        ('classifier', MultinomialNB())
    ])

    param_grid = {
        'vectorizer__min_df': [0.001, 0.005, 0.01, 0.05, 0.1],  # Test verschillende waarden voor min_df
        'classifier__alpha': [0.05, 0.1, 0.5,1, 1.5, 2],  # Test verschillende waarden voor alpha
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)  # Gebruik 5-fold cross-validation
    grid_search.fit(train_data['text'], train_data['label'])

    print("Beste hyperparameters:", grid_search.best_params_)
    print("accuracy hyperparameters op train:", grid_search.best_score_)

    return grid_search.best_params_['vectorizer__min_df'], grid_search.best_params_['classifier__alpha']
    
def Multinomial_Naive_Bayes(train_data, test_data, n_gram, min_df, alpha):
    # Vectoriseer de tekstgegevens
    vectorizer = CountVectorizer(ngram_range=(1, n_gram), min_df = min_df)
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])    
    
    # Labels voor training en testen
    y_train = train_data['label']
    y_test = test_data['label']
    
    # Initialiseer het Multinomial Naive Bayes model
    model = MultinomialNB(alpha=alpha)   

    # Train het model
    model.fit(X_train, y_train)
    
    # Voorspel de labels voor de testset
    y_pred = model.predict(X_test)
    
    # evalueer model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Voeg deze regel toe
    recall = recall_score(y_test, y_pred, average='weighted')        # Voeg deze regel toe
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)                                      # Voeg deze regel toe
    print("Recall:", recall)                                            # Voeg deze regel toe
    print("F1 Score:", f1)  
    
def CV_logistic_regression_lasso(train_data, n_gram):
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, n_gram))),
        ('classifier', LogisticRegression(penalty='l1', solver='liblinear'))
    ])

    param_grid = {
        'vectorizer__min_df': [0.001, 0.005, 0.01, 0.05, 0.1],  # Test verschillende waarden voor min_df
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],      
    }

    

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)  # Gebruik 5-fold cross-validation
    grid_search.fit(train_data['text'], train_data['label'])

    print("Beste hyperparameters:", grid_search.best_params_)
    print("accuracy hyperparameters op train:", grid_search.best_score_)

    return grid_search.best_params_['vectorizer__min_df'], grid_search.best_params_['classifier__C']
    
def Logistic_Regression_Lasso(train_data, test_data, n_gram, min_df, C):
    # Vectoriseer de tekstgegevens
    vectorizer = CountVectorizer(ngram_range=(1,n_gram), min_df= min_df)
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    
    # Labels voor training en testen
    y_train = train_data['label']
    y_test = test_data['label']
    
    # Initialiseer het Logistic Regression model met Lasso penalty
    model = LogisticRegression(penalty='l1', solver='saga', C= C)
    
    # Train het model
    model.fit(X_train, y_train)
    
    # Voorspel de labels voor de testset
    y_pred = model.predict(X_test)
    
    # evalueer model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Voeg deze regel toe
    recall = recall_score(y_test, y_pred, average='weighted')        # Voeg deze regel toe
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)                                      # Voeg deze regel toe
    print("Recall:", recall)                                            # Voeg deze regel toe
    print("F1 Score:", f1) 

def Classification_Tree(train_data, test_data, n_gram):
    # Vectoriseer de tekstgegevens
    vectorizer = CountVectorizer(ngram_range=(n_gram,n_gram))
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    
    # Labels voor training en testen
    y_train = train_data['label']
    y_test = test_data['label']
    
    # Initialiseer het Decision Tree model
    model = DecisionTreeClassifier()
    
    # Train het model
    model.fit(X_train, y_train)
    
    # Voorspel de labels voor de testset
    y_pred = model.predict(X_test)
    
    # Bereken de nauwkeurigheid
    accuracy = accuracy_score(y_test, y_pred)
    print("Decision Tree Accuracy:", accuracy)

def Random_Forest(train_data, test_data, n_gram):
    # Vectoriseer de tekstgegevens
    vectorizer = CountVectorizer(ngram_range=(n_gram,n_gram))
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    
    # Labels voor training en testen
    y_train = train_data['label']
    y_test = test_data['label']
    
    # Initialiseer het Random Forest model
    model = RandomForestClassifier()
    
    # Train het model
    model.fit(X_train, y_train)
    
    # Voorspel de labels voor de testset
    y_pred = model.predict(X_test)
    
    # Bereken de nauwkeurigheid
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", accuracy)
        
def import_data():
    train_data = []
    test_data = []    
    # Hardcoded pad naar de hoofdmap
    base_path = r"raw_files/negative_polarity"   

    # Loop door de klassen (deceptive en truthful)
    for class_folder in os.listdir(base_path):
        class_path = os.path.join(base_path, class_folder)
        
        if os.path.isdir(class_path):
            subfolders = os.listdir(class_path)
            # pak de eerste 4 folder als train set
            for subfolder in subfolders[:-1]:  
                subfolder_path = os.path.join(class_path, subfolder)
                for file_name in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:

                        #lowercase alle woorden
                        content = file.read().lower()     
                        # Verwijder engelse stopwoorden
                        content = ' '.join([word for word in content.split() if word not in ENGLISH_STOP_WORDS])
                        
                        train_data.append((content, class_folder))
            
            # Pak de laatste folder als testset
            test_subfolder = subfolders[-1]
            test_subfolder_path = os.path.join(class_path, test_subfolder)
            for file_name in os.listdir(test_subfolder_path):
                file_path = os.path.join(test_subfolder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    #lowercase alle woorden
                    content = file.read().lower() 
                    
                    # Verwijder engelsestopwoorden
                    content = ' '.join([word for word in content.split() if word not in ENGLISH_STOP_WORDS])

                    test_data.append((content, class_folder))
    
    # Zet de gegevens om in een DataFrame
    train_df = pd.DataFrame(train_data, columns=['text', 'label'])
    test_df = pd.DataFrame(test_data, columns=['text', 'label'])
    
    return train_df, test_df

if __name__ == "__main__":
    main()

