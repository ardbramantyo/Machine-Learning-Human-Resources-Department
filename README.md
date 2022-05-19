### Machine Learning Model to Predict Employee Attrition

#### Overview
Data Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

**Tools:**
1. Pandas
2. Numpy
3. Seaborn
4. Matplotlib
5. Scikit-Learn
6. Tensorflow
7. Keras

|   Categorical   |         Numerical        |
|-----------------|--------------------------|
| BusinessTravel  | Age                      |
| Department      | DailyRate                |
| EducationField  | DistanceFromHome         |
| Gender          | Education                |
| JobRole         | EnvironmentSatisfaction  |
| MaritalStatus   | HourlyRate               |
|                 | JobInvolvement           |
|                 | JobLevel                 |
|                 | JobSatisfaction          |
|                 | MonthlyIncome            |
|                 | MonthlyRate              |
|                 | NumCompaniesWorked       |
|                 | OverTime                 |
|                 | PercentSalaryHike        |
|                 | PerformanceRating        |
|                 | RelationshipSatisfaction |
|                 | StockOptionLevel         |
|                 | TotalWorkingYears        |
|                 | TrainingTimesLastYear    |
|                 | WorkLifeBalance          |
|                 | YearsAtCompany           |
|                 | YearsInCurrentRole       |
|                 | YearsSinceLastPromotion  |
|                 | YearsWithCurrManager     |

``` ruby
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
```

#### Model Accuracy

![image](https://user-images.githubusercontent.com/37673834/169185958-bc168712-12f0-46a0-bc03-00b51799d58f.png)

### Reference:
1. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
2. https://seaborn.pydata.org/generated/seaborn.heatmap.html
3. https://seaborn.pydata.org/generated/seaborn.countplot.html
4. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
5. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
6. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
7. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
8. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
9. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
10. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
11. https://towardsdatascience.com/a-practical-guide-to-implementing-a-random-forest-classifier-in-python-979988d8a263
12. https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
13. https://www.tensorflow.org/guide/keras/train_and_evaluate
14. https://matplotlib.org/3.5.0/plot_types/index.html
15. https://www.tensorflow.org/api_docs/python/tf/keras/Model
