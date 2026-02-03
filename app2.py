import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import pickle

df=pd.read_csv("telco_data.xls")

df=df.drop("customerID",axis=1)
X=df.drop("Churn",axis=1)
y=df.Churn
num_col=X.select_dtypes(include="number")
obj_col=X.select_dtypes(exclude="number")
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=42)
num_preprocessing=Pipeline(
    steps=[
        ('imputer_for_numcols',SimpleImputer(strategy='mean')),
        ('standardscaler',StandardScaler())
    ]
)
cat_preprocessing=Pipeline(
    steps=[
        ("imputer_for_obj_cols",SimpleImputer(strategy='constant',fill_value='Unknown')),
        ('OrdinalEncoder',OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]
)
preprocessing=ColumnTransformer(transformers=[
    ("num_preprocessing", num_preprocessing, num_col.columns),
    ("cat_preprocessing",cat_preprocessing, obj_col.columns)
])
pipeline= Pipeline(
    steps=[
        ("preprocessing",preprocessing),
        ("model_",KMeans(n_clusters=2)),
        ("model", LogisticRegression()),
    ]
)
pipeline.fit(xtrain,ytrain)

with open("model2.pkl","wb") as f:
    pickle.dump(pipeline,f)

