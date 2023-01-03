#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import skew
from scipy.stats import kurtosis
import pingouin as pg
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV


# In[2]:


#Importing Data From Excel (csv) file
taskdata = pd.read_csv('datajm.csv')


# In[3]:


#Viewing the Dataset
taskdata.head()


# In[4]:


#Descriptive Statistics
taskdata.describe()


# In[5]:


#Spliting Dataset into Vaiables
x = taskdata.iloc[:,1]
y = taskdata.iloc[:,2]
z = taskdata.iloc[:,3]
p = taskdata.iloc[:,4]
q = taskdata.iloc[:,5]
r = taskdata.iloc[:,6]
j = taskdata.iloc[:,7]
k = taskdata.iloc[:,8]
l = taskdata.iloc[:,9]


# In[6]:


#Computing Skewness
for col in taskdata:
    print(col)
    print(skew(taskdata[col]))
    
    plt.figure()
    sns.distplot(taskdata[col])
    plt.show()


# In[7]:


#Computing Kurtosis
for col in taskdata:
    print(col)
    print(kurtosis(taskdata[col]))


# In[8]:


#Paired Sample t-test Analysis to Compare Irelant and Spain with Regard to % Change in Production
#H0: M0=M1
#H1: M0 not equal to M1
#Alpha=0.05
stats.ttest_rel(x,y)


# In[9]:


#Paired Sample t-test Analysis to Compare Ireland and Spain with Regard to % Change in Nutrient Balance
#H0: M0=HM1
#H1: M0 Note Equal to M1
#Alpha=0.05
stats.ttest_rel(p,q)


# In[10]:


#Paired Sample t-test Analysis to Compare Ireland and Spain with Regard to % Change in Labor Input
#H0: M0=HM1
#H1: M0 Note Equal to M1
#Alpha=0.05
stats.ttest_rel(j,k)


# In[11]:


#Independent Sample ttest Anlysis to Compare Ireland and Spain with Regard to % Change in Production
#H0: M0=M1
#H1: M0 not equal to M1
#Alpha=0.05
stats.ttest_ind(x,y)


# In[12]:


#Indpendent Sample ttest Analaysis to Compare Ireland and Spain with Regard to % Change in Nutrient Balance
#H0: M0=M1
#H1: M0 not equal to M1
#Alpha=0.05
stats.ttest_ind(p,q)


# In[13]:


#Independent Sample ttest Anlysis to Compare Ireland and Spain with Regard to % Labor Input
#H0: M0=M1
#H1: M0 not equal to M1
#Alpha=0.05
stats.ttest_ind(j,k)


# In[14]:


#One sample ttest Analysis to Test Satatistical Significance in the Difference between Spain's Average % Change in Production and the Ireland mean % Change in Production
#H0: M=1.96
#H1: M not equal to 1.96
#Alpha=0.05
stats.ttest_1samp(x, 1.96)


# In[15]:


#One sample ttest Analysis to Test Satatistical Significance in the Difference between Spain's Average % Change in Nutrient Balance and the Ireland mean % Change in Nutrient Balance
#H0: M=1.96
#H1: M not equal to 2.89
#Alpha=0.05
stats.ttest_1samp(p, 2.89)


# In[16]:


#One sample ttest Analysis to Test Satatistical Significance in the Difference between Spain's Average % Change in Production and the Ireland mean % Change in Production
#H0: M=1.96
#H1: M not equal to 1.96
#Alpha=0.05
stats.ttest_1samp(j, -1.29)


# In[17]:


#Wilcoxon test to Compare Ireland and Spain with Regard to % Change in Production
wilcoxon(x,y)


# In[18]:


#Wilcoxon test to Compare Ireland and Spain with Regard to % Change in Nutrient Balance
wilcoxon(p,q)


# In[19]:


#Wilcoxon test to Compare Ireland and Spain with Regard to % Change in Labor Input
wilcoxon(j,k)


# In[21]:


#ANOVA to compare Ireland, Spain and Frnance with Regards to % Change in Production and Nutrient Balance
#H0: M1=M2=M3
#H1: M1 not equal to M2 not equal to M3
#Import ANOVA Data
nd = pd.read_csv('datajm_1.csv')


# In[22]:


#Viewing ANOVA Data
nd


# In[23]:


#Generating BoxPlots for the ANOVA Data
nd.boxplot('% Change in Production', by = 'Country')


# In[24]:


nd.boxplot('% Change in Nutrient Balance', by = 'Country')


# In[26]:


nd.boxplot('% Change in Labor Input', by='Country')


# In[27]:


#ANOVA 1: For Production
anova = pg.anova(dv='% Change in Production', between='Country', data=nd, detailed=True) 


# In[28]:


anova


# In[29]:


#ANOVA 2: For Nutrient Balance
anova2 = pg.anova(dv='% Change in Nutrient Balance', between='Country', data=nd, detailed=True) 


# In[30]:


anova2


# In[31]:


#ANOVA 3: For Labor Input
anova3= pg.anova(dv='% Change in Labor Input', between='Country', data=nd, detailed=True)


# In[32]:


anova3


# In[33]:


#Correlation Matrix
sns.heatmap(taskdata.corr(), annot=True)
plt.show()


# In[ ]:




