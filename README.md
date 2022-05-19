### Machine Learning Model to Predict Employee Attrition

#### Overview
Data Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

**Tools:** Pandas, Numpy, Seaborn, Matplotlib, Scikit-Learn, Tensorflow, Keras

#### Data Cleaning


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

Converting using scikit-learn from Categorical field into number
``` ruby
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
```

#### Deep Learning Model Evaluation

![image](https://user-images.githubusercontent.com/37673834/169192599-d06e652a-7a06-4309-b9e5-d2f6e7eab3ff.png)

|            | precision  |  recall | f1-score  | support |
|------------|------------|---------|-----------|---------|
|         0  |      0.88  |     0.92|     0.90  |     307 |
|          1 |      0.50  |    0.39 |     0.44  |      61 |
|    accuracy|            |         |     0.83  |     368 |
|   macro avg|      0.69  |    0.66 |     0.67  |     368 |
|weighted avg|      0.82  |    0.83 |     0.83  |     368 |


### Reference:
1. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
2. https://matplotlib.org/3.5.0/plot_types/index.html
3. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
4. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
5. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
6. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
7. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
8. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
9. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
10. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
11. https://seaborn.pydata.org/generated/seaborn.heatmap.html
12. https://seaborn.pydata.org/generated/seaborn.countplot.html
13. https://seaborn.pydata.org/generated/seaborn.kdeplot.html
14. https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
15. https://www.tensorflow.org/guide/keras/train_and_evaluate
16. https://www.tensorflow.org/api_docs/python/tf/keras/Model
17. https://towardsdatascience.com/a-practical-guide-to-implementing-a-random-forest-classifier-in-python-979988d8a263
