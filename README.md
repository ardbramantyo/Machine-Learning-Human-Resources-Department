### Comparative Machine Learning Method to Predict Employee Attrition
Data Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
### Overview
The project is aimed to develop Machine Learning models and make comparative prediction from "IBM HR Analytics Employee Attrition & Performance" fictional data (1470 rows of data) that could better predict employees attrition.

**Tools:** Pandas, Numpy, Seaborn, Matplotlib, Scikit-Learn, Tensorflow, Keras

### Data Cleaning
To avoid AI misunderstanding when interpreting data, 2 variables (X) are made based on their data type and converting categorical variable (X_cat) into numerical using scikit-learn and concatenate both them back.

Variables:
1. Categorical(X_cat): Anything from fields exclude Attrition that has object data type
2. Numerical(X_numerical):  Anything from fields that has numerical data type.

Code:
``` ruby
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
```
### Model Accuracy Test Method:
1. Logistic Regression
2. Random Forest
3. Deep Learning Model 

### Accuracy Measurement Method:
 - Training: 1102 (75%)
 - Test: 368 (25%)

### 1. Logistics Regression Model
![image](https://user-images.githubusercontent.com/37673834/169258872-4b8e00a5-164f-4a29-9985-805864f0d7c5.png)


 - Logistic regression is best used to predict binary outputs with two possible values labeled "0" or "1".
 - Logistic model output can be one of two classes: stayed/left, pass/fail, win/lose, etc.
 - Logistic regression algorithm works by implementing a linear equation first with independent predictors to predict a value.
 

### 2. Random Forest Classifier Model
![image](https://user-images.githubusercontent.com/37673834/169261437-ebbbbf42-e0c7-4dfa-8528-47c52edf93e9.png)

 - Decision Trees are supervised Machine Learning technique where the data is split according to a certain condition/parameter. 
 - Random Forest Classifier is a type of ensemble algorithm. 
 - It creates a set of decision trees from randomly selected subset of training set. 
 - It then combines votes from different decision trees to decide the final class of the test object.

### 3. Deep Learning Model
![image](https://user-images.githubusercontent.com/37673834/169259386-f9650727-6042-40de-9f92-74e5b1d58a46.png)

**Parameter for training:**
1. Input = 50 (from table fields)
2. Hidden Layer = 3 layers (dense, 500 neurons each, relu activation function)
3. Output = 1 (sigmoid activation function)
4. Epochs = 100 
5. Batch size = 50

### Deep Learning Performance
![image](https://user-images.githubusercontent.com/37673834/169192599-d06e652a-7a06-4309-b9e5-d2f6e7eab3ff.png)


### Confusion Matrix Comparison

![image](https://user-images.githubusercontent.com/37673834/169203314-84f5b0e0-2406-4e74-af64-d9022e0f4da9.png)
Confusion Matrix: Logistic Regression(left), Random Forest(mid), and Deep Learning(right)

|   Method           |Accuracy (%)|
|--------------------|------------|
| Logistic Regression|         89 |
| Random Forest      |         85 |
| Deep Learning      |         83 |

### Conclusion
Based on analysis with 3 different Machine Learning Methods, Logistic Regression has best Accuracy value (89%) and suitable to be applied to predict employee attriction.
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
17. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
18. https://towardsdatascience.com/a-practical-guide-to-implementing-a-random-forest-classifier-in-python-979988d8a263
