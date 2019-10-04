import numpy as np
import pandas as pd

df1 = pd.read_csv("lbnl.anon-ftp.03-01-11.csv")
df2 = pd.read_csv("lbnl.anon-ftp.03-01-14.csv")
df3 = pd.read_csv("lbnl.anon-ftp.03-01-18.csv")

print(df1.head())

def getNums(df):
	df1 = df.loc[df["Info"].str.contains('[SYN]',regex=False) ]
	df1 = df.drop_duplicates(subset='Info')
	return (df1['Source'].drop_duplicates().size,df1['Destination'].drop_duplicates().size)