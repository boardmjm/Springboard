import pandas as pd
import numpy as numpy

# Dummy/OneHot Encode categorical features
dfo = pd.DataFrame(df_chal[['city', 'phone', 'ultimate_black_user']])
df_chal = pd.concat([df_chal.drop(dfo, axis=1), pd.get_dummies(dfo)], axis=1)
df_chal





# Classifier Evaluation Formula with Classification Report and Confusion Matrix
def evaluate(y_test, y_pred):
    '''Print Classification Report and Confusion Matrix'''
    
    print(classification_report(y_test, y_pred))
    
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(7,5))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()





# Drop Column Feature Importance
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

from sklearn.base import clone 

def drop_col_feat_imp(model, X_train, y_train, random_state = 34):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    
    importances_df = imp_df(X_train.columns, importances)
    return importances_df

def plot_feat_imp(df, X):
    '''create horizontal bar plot of feature importances'''
    
    df['feat_imp_rate'] = 100.0 * (df.feature_importance / df.feature_importance.max())
    sorted_idx_rf = np.argsort(df.feat_imp_rate)
    pos_rf = np.arange(sorted_idx_rf.shape[0]) + .5
    plt.figure(figsize=(5,7))
    plt.barh(pos_rf, df.feat_imp_rate[sorted_idx_rf], align='center')
    plt.yticks(pos_rf, X.columns[sorted_idx_rf])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()



