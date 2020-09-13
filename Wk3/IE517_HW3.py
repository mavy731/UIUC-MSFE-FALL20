import sys
import pandas as pd
import numpy as np
import pylab
import scipy.stats as stats
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('high_yield_corporate_bond_data.csv')

#sizing up the data

sys.stdout.write("Number of Columns of data " + str(len(df.columns))+'\n')
sys.stdout.write("Number of Rows of data " + str(len(df))+'\n')

n_row = len(df)
n_col = len(df.columns)

type = [0] *3
colCounts =[]

#determining the nature of attributes
for j in range(0,n_col-1):
    for i in range(0, n_row-1):
        try:
            a = float(df.iloc[i,j])
            if isinstance(a,float):
                type[0] += 1
        except ValueError:
            if len(df.iloc[i,j]) > 0:
                type[1] += 1
            else:
                type[2] += 1

    colCounts.append(type)
    type = [0]*3

sys.stdout.write("Col#" + '\t' + "Number" + '\t' +
"Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
                    str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1

#Summary Statistics

#generate summary statistics for column 9

col = 9
print(df.columns[9])
colData= []
for i in range(0,n_row):
    colData.append(float(df.iloc[i,col]))

colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
                "Standard Deviation = " + '\t ' + str(colsd) + "\n")

#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

col = 8
print(df.columns[8])
colData = []
for i in range(0,n_row):
    colData.append(df.iloc[i,col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)

#count up the number of elements having each value
#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0] * 28

for elt in colData:
    catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")

print(list(unique))
print(catCount)

#QQ-Plot for Issued Amount
type = [0]*3
colCounts = []
col = 10
colData2= []
for i in range(0,n_row):
    colData2.append(df.iloc[i,col])

stats.probplot(colData2, dist="norm", plot=pylab)
plt.savefig("QQ-plot", dpi=300)
pylab.show()

#Histogram of Industry
In_bin=df.Industry.unique()
sns.set()
fig = plt.hist(df['Industry'],bins=In_bin)
plt.xlabel('Industry')
plt.xticks(rotation=90)
plt.tick_params(labelsize=5)
plt.ylabel('Number of Bonds')
plt.tight_layout()
plt.savefig("Industry.png",dpi=300)
plt.show()

#Swarm plot of industry and liquidityscore
sns.swarmplot(x='Industry',y='LiquidityScore',data=df)
plt.xlabel('Industry')
plt.xticks(rotation=90)
plt.tick_params(labelsize=5)
plt.ylabel('Liquidity Score')
plt.tight_layout()
plt.savefig("Liquidity Score with Industry.png",dpi=300)
plt.show()

#Boxplot for industry and LIQ Score
sns.boxplot(x='Industry',y='LIQ SCORE', data=df)
plt.xlabel('Industry')
plt.ylabel('LIQ Score')
plt.xticks(rotation=90)
plt.tick_params(labelsize=5)
plt.tight_layout()
plt.savefig("boxplot.png",dpi=300)
plt.show()


#Presenting Attribute correlations
df1=pd.read_csv('high_yield_corporate_bond_data.csv', usecols = ['weekly_mean_volume','weekly_mean_ntrades'], low_memory = False)
corMat = DataFrame(df1.corr())
plt.pcolor(corMat)
plt.xlabel('Weekly Mean Volume')
plt.ylabel('Weekly Mean Ntrades')
plt.tight_layout()
plt.savefig("correlations.png",dpi=300)
plt.show()


#Pearson Correlation and heat map
df2 = df.iloc[:,20:35]
corr=df2.corr()
#print correlation matrix
print(corr)
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns)
plt.tight_layout()
plt.savefig("heatmap.png",dpi=300)
plt.show()


print("My name is Yue Ma")
print("My NetID is: yuema4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
