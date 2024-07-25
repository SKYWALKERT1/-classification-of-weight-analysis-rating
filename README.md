* 📖 Introduction *
This project analyzes a dataset related to obesity and various influencing factors. The aim is to understand the data and build a predictive model to classify the frequency of physical activity (FAF) of individuals based on other features.

📊 Dataset
The dataset used is ObesityDataSet_raw_and_data_sinthetic.csv, containing 2111 entries and 17 columns including Gender, Age, Height, Weight, and other lifestyle and health-related attributes.

🛠 Data Preprocessing
Data preprocessing steps include:

Loading the Dataset: The dataset is loaded using Pandas.
python
Kodu kopyala
import pandas as pd
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
One-Hot Encoding: Categorical variables are converted into numerical format using one-hot encoding.
python
Kodu kopyala
df_encoded = pd.get_dummies(df, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'])
Checking for Missing Values: Ensuring there are no missing values in the dataset.
python
Kodu kopyala
df.isnull().sum()
🔍 Exploratory Data Analysis
Exploratory Data Analysis (EDA) includes:

Summary Statistics: Providing an overview of the dataset.
python
Kodu kopyala
df.describe()
Correlation Matrix: Identifying relationships between numerical features.
python
Kodu kopyala
corr_matrix = df.corr(numeric_only=True)
Visualization: Visualizing the correlation matrix using a heatmap.
python
Kodu kopyala
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()
🔍 Feature Selection
Feature selection process includes:

Correlation with Target Variable: Identifying features with significant correlation with FAF.
python
Kodu kopyala
cor_target = abs(corr_matrix["FAF"])
relevant_features = cor_target[cor_target > 0.1]
Dropping Irrelevant Features: Dropping features with low correlation.
python
Kodu kopyala
to_drop = cor_target[cor_target < 0.1]
row_names_list = list(to_drop.index)
row_names_list.append('FAF')
X = df_encoded.drop(row_names_list, axis=1).values
y = df_encoded['FAF'].values
🏗 Model Building
Building a Decision Tree Classifier:

Splitting the Data: Splitting the dataset into training and testing sets.
python
Kodu kopyala
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
Pipeline Creation: Creating a pipeline with scaling and decision tree classifier.
python
Kodu kopyala
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
steps = [('scaler', StandardScaler()), ('dec_tree', DecisionTreeClassifier())]
pipeline = Pipeline(steps)
Hyperparameter Tuning: Using GridSearchCV to tune hyperparameters.
python
Kodu kopyala
from sklearn.model_selection import GridSearchCV
params = {"dec_tree__criterion": ['gini', 'entropy'], "dec_tree__max_depth": np.arange(3, 15)}
for cv in range(3, 8):
    cv_grid = GridSearchCV(pipeline, param_grid=params, cv=cv)
    cv_grid.fit(X_train, y_train)
    print(f"{cv}-fold score: {cv_grid.score(X_test, y_test):.2f}")
    print("Best parameters: ", cv_grid.best_params_)
Model Training: Training the model with the best parameters.
python
Kodu kopyala
best_tree = DecisionTreeClassifier(criterion='gini', max_depth=4)
best_tree.fit(X_train, y_train)
📈 Results
Evaluating the model performance:

python
Kodu kopyala
from sklearn.metrics import classification_report
y_pred = best_tree.predict(X_test)
print(classification_report(y_test, y_pred))
✅ Conclusion
The Decision Tree Classifier was built to classify the frequency of physical activity (FAF). The best parameters for the decision tree were found using cross-validation.

📦 Dependencies
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Install the dependencies using:

bash
Kodu kopyala
pip install pandas numpy matplotlib seaborn scikit-learn
Feel free to adjust the content as needed!
