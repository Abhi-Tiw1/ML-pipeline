# Machine Learning Pipeline

Here is a general machine pipeline used in most of my works created using scikit-learn and some other libraries. This general pipeline has various components which have been divided into functions. In order to their use the components are:
- Handling unbalanced data : This has been used using data sampling strategies from Imbalanced-learn. Various over and undersampling options are avalible.
- Feature Selection Strategies: Three options are avalible for feature selection:
  - ANOVA based feature selection (ANOVA)
  - Recurcive feature elimination (RFE)
  - minimum redundency maximum relevance (mRMR): This is implemented using the pymrmr library
- Classifier selection: A function implements various classifiers, SVM (linear and rbf kernals), KNN, Random Forest, Logistic Regression
- Cross validation strategy: This strategy repeats 5-fold cross validation X-times. For each train and test set, first feature selection is performed followed by classification from the set of classifiers given
- Feature importance can be assessment by looking at most frequenct features in the selected feature sets.
