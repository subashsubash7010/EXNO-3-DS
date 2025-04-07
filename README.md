## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/764c2026-cfcb-48a3-9a37-3be19fed6642)

## ORDINAL ENCODER
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/7556abef-6296-4c0c-8316-c3d3124cd8b0)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/ad7ce34d-dbc4-4c17-8194-ff1b6029b123)

## LABEL ENCODER
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
```
![image](https://github.com/user-attachments/assets/5ccb6d3a-dd7f-4f8e-9711-2e8881e6ee0c)

```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
`![image](https://github.com/user-attachments/assets/55f477a0-8120-4a69-83b4-8a6402e1ad58)

## ONEHOT ENCODER
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/c0114dc6-c45a-4761-a0ba-bbdfdbbf9f05)

```
df2=pd.concat([df,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/1ed7a7c7-ebff-4aff-aab5-802b7007921c)

```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/b37c6643-385a-4e04-9b30-f14fd6dafd0a)

## BinaryEncoder
```
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data (1).csv")
df
```
![image](https://github.com/user-attachments/assets/cae8f3bc-413c-413d-b14b-9bc87f2b5418)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/3213d425-c0a9-4852-b23b-6fb2815eec5f)

## TARGET ENCODER
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/637d44e6-21da-4d81-93e9-a01533e2927c)

## FEATURE ENGINEERING
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/a57c1cf1-590b-44d8-b58c-d12904ea888f)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/9def3c3a-d4ed-46c5-83f1-399301b35006)

```
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/f61043ac-3fb1-44b6-b682-1302c7ab4ce2)
```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/c9c09563-c5c2-4eca-9681-04d08aaaf817)
```
df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/08f6b9da-99b0-4bbe-a49d-cb6b4d12817f)

```
df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/799e50eb-f4a5-4d10-a4b6-46572e30b85e)

## POWER TRANSFORMATION
```
df["Highly Positive Skew"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/b6b0f4f2-e2f5-47d8-a9c6-876ab4366c01)

```
df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/f3c51986-dc9b-4575-ba98-9456f82cacce)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/cc55a050-ffc7-4ed1-9761-d739adb3a207)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/d44a5f2e-efce-4a3f-922b-8ab97e8791aa)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/d8b8ccba-ded9-4b22-9786-383cc9efed47)


# RESULT:

      The given data and perform Feature Encoding and Transformation process and save the data to a file is created

       
