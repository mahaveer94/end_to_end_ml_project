import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load datasets
student_mat = pd.read_csv('student-mat.csv', sep=';')
student_por = pd.read_csv('student-por.csv', sep=';')

# Merge datasets on specified columns
merge_columns = ["school", "sex", "age", "address", "famsize", "Pstatus",
                 "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]

merged_data = pd.merge(student_mat, student_por, on=merge_columns, suffixes=('_mat', '_por'))

merged_data.head()

merged_data.isnull().sum()


print(merged_data.describe())

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
#corr_matrix = merged_data.corr()
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#plt.show()

# Distribution of final grades
sns.histplot(merged_data['G3_mat'], kde=True)
plt.title('Distribution of Final Grades (Mathematics)')
plt.show()

sns.histplot(merged_data['G3_por'], kde=True)
plt.title('Distribution of Final Grades (Portuguese)')
plt.show()

# Pairplot to visualize relationships
sns.pairplot(merged_data[['G1_mat', 'G2_mat', 'G3_mat', 'studytime_mat', 'absences_mat']])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
categorical_features = ['famsize', 'internet']

le = LabelEncoder()
for feature in categorical_features:
    merged_data[feature] = le.fit_transform(merged_data[feature])

# Select features and target
features = ['studytime_mat', 'G1_mat', 'G2_mat', 'absences_mat', 'age', 'famsize', 'Medu', 'Fedu', 'internet']
target_mat = merged_data['G3_mat']
target_por = merged_data['G3_por']

X = merged_data[features]

# Train-test split for Mathematics
X_train_mat, X_test_mat, y_train_mat, y_test_mat = train_test_split(X, target_mat, test_size=0.2, random_state=42)

# Train-test split for Portuguese
X_train_por, X_test_por, y_train_por, y_test_por = train_test_split(X, target_por, test_size=0.2, random_state=42)

# Linear Regression Model for Mathematics
model_mat = LinearRegression()
model_mat.fit(X_train_mat, y_train_mat)
predictions_mat = model_mat.predict(X_test_mat)

mse_mat = mean_squared_error(y_test_mat, predictions_mat)
r2_mat = r2_score(y_test_mat, predictions_mat)

print(f'Mathematics - Mean Squared Error: {mse_mat}')
print(f'Mathematics - R² Score: {r2_mat}')

# Linear Regression Model for Portuguese
model_por = LinearRegression()
model_por.fit(X_train_por, y_train_por)
predictions_por = model_por.predict(X_test_por)

mse_por = mean_squared_error(y_test_por, predictions_por)
r2_por = r2_score(y_test_por, predictions_por)

print(f'Portuguese - Mean Squared Error: {mse_por}')
print(f'Portuguese - R² Score: {r2_por}')

import matplotlib.pyplot as plt

plt.scatter(y_test_mat, predictions_mat)
plt.xlabel('True Grades (Mathematics)')
plt.ylabel('Predicted Grades (Mathematics)')
plt.title('True vs Predicted Grades (Mathematics)')
plt.show()

plt.scatter(y_test_por, predictions_por)
plt.xlabel('True Grades (Portuguese)')
plt.ylabel('Predicted Grades (Portuguese)')
plt.title('True vs Predicted Grades (Portuguese)')
plt.show()
