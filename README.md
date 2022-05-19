### Machine Learning Model to Predict Employee Attrition

#### Overview
Data Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

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

### Reference
1. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
2. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
3. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
