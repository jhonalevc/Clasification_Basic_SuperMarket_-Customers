#Project No.1  - Segmentation

#Importing Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Inspecting the data 

def sp():
  print('\n')

data = pd.read_csv('/content/drive/MyDrive/Data_Science/Mall_Consumer_Segmentation.csv')
print(data.head())
sp()
print(data.info())
sp()
print(data.describe())

# After inspecting the data, we found that the dataset is composed by 5 columns and 200 rows.
# Column age is stored as 'object' type, the values in this column need to be converted to categories for the machine learning model to work 

data['Gender'] =  data['Gender'].astype('category')
print(data['Gender'].dtypes)

#Exploration linear Relationships:

sns.set()

# Explore Age  vs Annual Income 

pearson_r_age_vs_income = np.corrcoef(
    x=data['Age'],
    y=data['Annual Income (k$)']
    )

print(pearson_r_age_vs_income)
sp()
print('There is no linear correlation between Age and annual income as shown below')
sp()
sns.scatterplot(
    x='Age',
    y='Annual Income (k$)',
      data=data )
plt.title('Age vs Annual Income')
plt.show()
sp()

pearson_r_age_vs_spending = np.corrcoef(
    x = data['Age'],
    y = data['Spending Score (1-100)'])
sp()
print(pearson_r_age_vs_spending)
sp()
print('There is no correlation between the Age and the spending Score')
sp()
sns.scatterplot(
    x=data['Age'],
    y=data['Spending Score (1-100)'],
    data= data)
plt.title('Age vs Spending Score')
plt.show()
sp()

pearson_r_Income_vs_spending_score= np.corrcoef(
    data['Annual Income (k$)'],
    data['Spending Score (1-100)']
    )
print(pearson_r_Income_vs_spending_score)
sp()
print('Theres no correlation between Spending socre and anual income')
print('But you can see that there are 5 indentifiable possible clusters ')
sp()
sns.scatterplot(x=data['Annual Income (k$)'],
                y= data['Spending Score (1-100)'],
                data=data)
plt.title('Anual Income Vs Spending Score')
plt.show()

#How are some variables disteibuted ¿ Can we make Basic predictions ?

#Create an empirical distribution Funciton  - 

    #ECDF

def ecdf(data_):
  '''
  Compute ECDF for a one-dimensional array of measurments
  '''
  n = len(data_)
  x = np.sort(data_)
  y = np.arange(1 , n+1) / n
  return x, y

  # Plot Empirical distribution functions vs Normal distribution of age

x_age, y_age = ecdf(data['Age'])
plt.plot(
    x_age,
    y_age,
    marker='.',
    linestyle='none'
    )
plt.xlabel('Ages of consumers')
plt.ylabel('ECDF')

      # Plotting Normal distribution

mean_age = np.mean(data['Age'])
std_age = np.std(data['Age'])
normal_age = np.random.normal(
    loc = mean_age,
    scale = std_age,
    size = 2000
)

normal_x, normal_y = ecdf(normal_age)
plt.plot(
    normal_x,
    normal_y,
    marker = '.',
    linestyle= 'none',
    color = 'r',
    alpha = 0.02
)
plt.title('ECDF AGE VS NORMAL')
plt.show()
sp()
print(' As seen in the image above, the age data does not follow the normal distribution so we can not use it to make accurate predictions ')
sp()

    # Plot empirical distribution function vs normal  - Income

x_income, y_income = ecdf(data['Annual Income (k$)'])
plt.plot(
    x_income,
    y_income,
    linestyle='none',
    marker= '.'
)
mean_income = np.mean(data['Annual Income (k$)'])
std_income = np.std(data['Annual Income (k$)'])
normal_income = np.random.normal(
    loc=mean_income,
    scale = std_income,
    size=2000)
x_normal_income, y_normal_income = ecdf(normal_income)

plt.plot(
    x_normal_income,
    y_normal_income,
    marker = '.',
    linestyle = 'none',
    color = 'r',
    alpha = 0.02)
plt.title('Income ECDF')
plt.xlabel('Income in thousands of dollars')
plt.ylabel('ECDF')
plt.show()
sp()
print(' As seen in the image above, the Income data does not follow the normal distribution so we can not use it to make accurate predictions ')
sp()

    # Plot empirical distribution Spending Score Vs normal 

x_spending, y_spending = ecdf(data['Spending Score (1-100)'])
plt.plot(
    x_spending,
    y_spending,
    linestyle='none',
    marker='.'
)

spending_mean = np.mean(data['Spending Score (1-100)'])
spending_std = np.std(data['Spending Score (1-100)'])
normal_spending = np.random.normal(
    loc = spending_mean,
    scale = spending_std,
    size = 2000
)

x_spending_normal, y_spending_normal = ecdf(normal_spending)
plt.plot(
    x_spending_normal,
    y_spending_normal,
    marker = '.',
    linestyle = 'none',
    alpha = 0.02
)
plt.title('ECDF Spending Score')
plt.xlabel('Spending Score (1 - 100 )')
plt.ylabel('ECDF')
plt.show()
sp()
print(' As seen in the image above, the Score data does not follow the normal distribution so we can not use it to make accurate predictions ')

#  Different visualitazion

plt.hist(x = data['Age'], bins=6)
plt.xlabel('Ages')
plt.ylabel(' Count Ages')
plt.title('')

plt.hist(normal_age, histtype='step', bins = 120)
plt.show()

# Label encoding Age option 1 .cat.codes 

sp()
data_ = pd.read_csv('/content/drive/MyDrive/Data_Science/Mall_Consumer_Segmentation.csv')
data_['Gender'] = data_['Gender'].astype('category')
data_['Gender_cat'] = data_['Gender'].cat.codes
print(data_.head())
sp()
print(data_.dtypes)

# Starting The model  - K means from scikit learn will be used for this exercise.
# Let's start by deciding how many clusters we're going to choose - We will use the elbow rule

from sklearn.cluster import KMeans

ks_option_1 = range(1,8)
inertias_1 = []

for k in ks_option_1:
  model_ = KMeans(n_clusters = k)
  model_.fit(data_[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_cat']])
  inertias_1.append(model_.inertia_)

sp()
plt.bar(ks_option_1,height=inertias_1)
plt.show()
sp()
plt.plot(ks_option_1, inertias_1,'-o')
plt.xlabel('number of clusters')
plt.ylabel('inertia')
plt.xticks(ks_option_1)
plt.show()
sp()
print('A seen in this chart, the elbow rule does not allow uss to determine how many clusters to choos' )
print( 'This can happen because the Variables are not normalized and are not scaled and may also be for the encoding of the age variable')
sp()
print(inertias_1)

# Label Encoding using get_dummies

with_dummies = pd.get_dummies(data,columns=['Gender'])
with_dummies = with_dummies.drop('CustomerID',axis= 1)
print(with_dummies.head())
sp()

ks = np.arange(1,8)
inertias = []

with_dummies.to_excel('With_dummies.xlsx')

for k in ks:
  model = KMeans(n_clusters = k)
  model.fit(with_dummies)
  inertias.append(model.inertia_)


sp()
plt.bar(ks,height=inertias)
plt.show()
sp()
plt.plot(ks, inertias,'-o')
plt.xlabel('number of clusters')
plt.ylabel('inertia')
plt.xticks(ks)

plt.show()

print(inertias)

# Let's not take into account the Gender 

no_gender = data.drop(['Gender','CustomerID'],axis=1)
print(no_gender.head(3))
sp()
ks_nogender = np.arange(1,8)
inertias_no_gender = []

for k in ks_nogender:
  model_nogender = KMeans(n_clusters = k)
  model_nogender.fit(no_gender)
  inertias_no_gender.append(model_nogender.inertia_)


plt.bar(ks_nogender,
        height=inertias_no_gender)
plt.show()
sp()
plt.plot(ks_nogender,inertias_no_gender,'-o')
plt.show()
sp()
print(inertias_no_gender)

# Looks like the Gender data doesn't provide a lot pf insigght in order to cluster the information 
      # We Will try dimention Reduction and scaling - Principal component Analysis. - PCA

            # STEP 1 : DE-CORRELATION

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


scaler = StandardScaler()
pca = PCA()
pipeline_1 = make_pipeline(scaler,pca)
pipeline_1.fit(with_dummies)


#plotting

features = range(pca.n_components_)
variance = pca.explained_variance_
plt.bar(x = features,
        height = variance)
plt.xticks(features)
plt.xlabel('PCA Features')
plt.ylabel('Variance')
plt.show()
sp()
print('As shown in the graph above, the intrinsec dimension can be considered as 3')


#Reducing the dimensions

pca_red = PCA(n_components = 3)
pipeline_red = make_pipeline(scaler,
                              pca_red)
pipeline_red.fit(with_dummies)
transformed = pca_red.transform(with_dummies)
sp()
print()
print(transformed.shape)
print(type(transformed))
sp()
print(transformed[0:10,:])

#Final Model

# Final determination on how many clusters 

clusters = np.arange(1,10)
final_inertias = []
for cluster in clusters:
  model_clusters = KMeans(n_clusters = cluster)
  model_clusters.fit(transformed)
  final_inertias.append(model_clusters.inertia_)

plt.plot(clusters,
         final_inertias, '-o')
plt.xticks(clusters)
plt.xlabel('# Clusters')
plt.ylabel('Inertia')
plt.plot()
plt.show()
sp()






final = KMeans(n_clusters= 5)
cluster_no = final.fit_predict(final_data.drop('Gender',axis=1))
data['Cluster'] = cluster_no
print(data.head(3))
sp()
grouped = data.groupby('Cluster').agg([np.mean,min,max])
grouped.to_excel('grouped.xlsx')
data['Cluster'] = data['Cluster'].astype('category')
sp()

sns.scatterplot(x='Age',
                y='Spending Score (1-100)',
                hue = 'Cluster',
                data = data)
plt.title('Age vs Spending Score')
plt.show()
sp()

sns.scatterplot(x='Annual Income (k$)',
                y='Spending Score (1-100)',
                hue = 'Cluster',
                data = data)
plt.title('Income vs Spending Score')
plt.show()
sp()
sns.scatterplot(x='Age',
                y='Annual Income (k$)',
                hue = 'Cluster',
                data = data)
plt.title('Age vs Annual Income (k$)')
plt.show()
