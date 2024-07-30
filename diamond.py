import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

%matplotlib inline

sns.set(rc={'figure.figsize': [15, 15]}, font_scale=1.3, style='dark')

df = pd.read_csv('diamonds.csv', index_col=0)
df

df.info()

cat_attributes= ['cut','color','clarity']
for att in cat_attributes:
    print(f'{att}:\n--------')
    res = df[att].value_counts()
    print(res)
    print('***********************************')


sns.set(rc={'figure.figsize': [15, 15]}, font_scale=1.3, style='dark')

plt.subplot(1,3,1)
myExplode = [0.3, 0, 0, 0, 0, 0, 0]
sns.set(rc={'figure.figsize': [5, 5]}, font_scale=1.3, style='dark')
plt.pie(list(df['color'].value_counts().values), labels=list(df['color'].value_counts().index) , explode= myExplode, autopct='%1.1f%%',
         startangle=90)
plt.title('color')
 

plt.subplot(1,3,2)
myExplode = [0.3, 0, 0, 0, 0]
sns.set(rc={'figure.figsize': [5, 5]}, font_scale=1.3, style='dark')
plt.pie(list(df['cut'].value_counts().values), labels=list(df['cut'].value_counts().index) , explode= myExplode, autopct='%1.1f%%', startangle=90)
plt.title('cut')

plt.subplot(1,3,3)
myExplode = [0.3, 0, 0, 0, 0,0,0,0]
sns.set(rc={'figure.figsize': [10, 5]}, font_scale=1.3, style='dark')
plt.pie(list(df['clarity'].value_counts().values), labels=list(df['clarity'].value_counts().index) , explode= myExplode, autopct='%1.1f%%',startangle=90)
plt.title('clarity')


sns.set(rc={'figure.figsize': [15, 10]}, font_scale=1.3, style='dark')

plt.subplot(1,3,1)
#sns.barplot(x=df['color'].value_counts())
sns.barplot(data=df, y='color', x='price')
plt.title('color')
 

plt.subplot(1,3,2)
sns.barplot(data=df, y='cut', x='price')
plt.title('cut')

plt.subplot(1,3,3)
sns.barplot(data=df, y='clarity', x='price')
plt.title('clarity')


df.describe()

sns.set(rc={'figure.figsize': [15, 15]}, font_scale=1.3, style='dark')
df.hist(bins=30)
plt.show()


sns.set(rc={'figure.figsize': [15, 15]}, font_scale=1.3, style='dark')
num_attr= ['carat','depth','table','price','x','y','z']
for i, att in zip(range(1,30),num_attr):
    plt.subplot(3,3,i)   
    sns.distplot(df[att],hist=False)


sns.color_palette("flare", as_cmap=True)
sns.pairplot(df)


sns.color_palette("flare", as_cmap=True)
#sns.pairplot(df[['x','y','z','depth']], palette='flare', markers='x',hue=df['cut'])
sns.pairplot(data=df, vars=['x','y','z','depth'], palette='flare', hue='cut', markers='x')


sns.pairplot(data=df, vars=['x','y','z','depth'], palette='flare', hue='color')#, markers='x')


sns.pairplot(data=df, vars=['x','y','z','depth'], palette='flare', hue='clarity', markers='x')


sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.3, style='dark')
sns.histplot(data=df,x='price', palette='viridis', kde=True ,hue='color', multiple="stack")


sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.3, style='dark')
sns.histplot(data=df,x='price', palette='viridis', kde=True ,hue='cut', multiple="stack")


sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.3, style='dark')
sns.histplot(data=df,x='price', palette='viridis', kde=True ,hue='clarity', multiple="stack")

list(df['clarity'].value_counts().values)

list(df['color'].value_counts().index)

sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.3, style='dark')
sns.heatmap(data=df.corr(), cmap='rocket', annot=True)

sns.set(rc={'figure.figsize': [15, 10]}, font_scale=1.3, style='white')
sns.boxplot(data=df, x='price', y='color', palette='Spectral')

sns.set(rc={'figure.figsize': [15, 10]}, font_scale=1.3, style='white')
sns.violinplot(data=df, x='price', y='color', palette='Spectral')

sns.boxplot(data=df, x='price', y='cut', palette='Spectral')

sns.violinplot(data=df, x='price', y='cut', palette='Spectral')

sns.boxplot(data=df, x='price', y='clarity', palette='Spectral')

sns.violinplot(data=df, x='price', y='clarity', palette='Spectral')

df['volume']=df['x']*df['y']*df['z'] # measured in mm cube
df['table_mm']= df['table']/100
df['diameter']=df['table']*df['y']/100  #df['y']*100/df['table'] 
df['depth_mm']=df['depth']/100
df['pavilion depth']=df['depth']-df['z'] # PD= height - depth in mm
df['pavilion angle']= np.tan(df['diameter']/df['pavilion depth']) #PA = tan(diameter / depth in mm)  # measured in degree

df.head()

df = df[['carat','cut','color','clarity','depth','table','x','y','z','volume','table_mm','diameter','depth_mm','pavilion depth','pavilion angle','price']]
sns.set(rc={'figure.figsize': [15, 15]}, font_scale=1.3, style='dark')
sns.heatmap(data=df.corr(), cmap='rocket', annot=True)

df.drop(['depth','table','table_mm', 'depth_mm', 'pavilion angle'], axis=1, inplace=True)

df.info()
