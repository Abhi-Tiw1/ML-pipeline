# Machine Learning Pipeline

Here is a general machine pipeline used in most of my works created using *scikit-learn* and some other libraries. This general pipeline has various components which have been divided into functions. In order to their use the components are:
- Handling unbalanced data : This has been used using data sampling strategies from *Imbalanced-learn*. Various over and undersampling options are avalible.
- Feature Selection Strategies: Three options are avalible for feature selection:
  - ANOVA based feature selection (ANOVA)
  - Recurcive feature elimination (RFE)
  - minimum redundency maximum relevance (mRMR): This is implemented using the *pymrmr* library
- Classifier selection: A function implements various classifiers, SVM (linear and rbf kernals), KNN, Random Forest, Logistic Regression
- Performance Metric: Various performance metrics have been used: accuracy, f1-score, precision, recall, cohen's kappa score and matthews correlation coefficient. The last two metrics are useful for unbalanced datasets.
- Cross validation strategy: This strategy repeats 5-fold cross validation X-times with different random seeds. For each train and test set, first feature selection is performed followed by classification from the set of classifiers given
- Feature importance: This can be assessed by looking at most frequenct features in the selected feature set across various cross validation folds. If a feature appreaded as one of the top features in all train-test combinations, it would be a robust feature.
- Saving performance metrics: The performance metrics are saved for each fold along with random voting performance for each fold and can be used later for significance testing.

## Libraries Used
Apart from scikit-learn the following libraries have been used
- [pymrmr](https://pypi.org/project/pymrmr/) 

    $ pip install numpy Cython
    $ pip install pymrmr
    
- [Imbalanced-learn](https://imbalanced-learn.org/stable/index.html)

    $ pip install -U imbalanced-learn
