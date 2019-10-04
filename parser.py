import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-11.csv"))
df2 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-14.csv"))
df3 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-18.csv"))

print(df1.head())

def cleanData(df):
	df['Info'] = df['Info'].apply(lambda x: x.strip())
	return df

def getTCP(df):
	df1 = df.loc[df["Protocol"] ==  "TCP"]
	df1["Source Port"] = df1['Info'].apply(lambda x: int(x.split()[0]))
	df1['Destination Port'] = df1['Info'].apply(lambda x: int(x.split()[2]))
	return df1

def getFTP(df):	
	return df.loc[df["Protocol"] ==  "FTP"]

def getSyn(df):
	return df.loc[df["Info"].str.contains('[SYN]',regex=False) ]

def getNums(df):
	df1 = getSyn(df)
	df1 = df.drop_duplicates(subset=['Info'])
	return (df1['Source'].drop_duplicates().size,df1['Destination'].drop_duplicates().size)

def getFlows(df):
	df1 = getTCP(getSyn(df))
	return df1.drop_duplicates(subset=["Source","Destination","Source Port","Destination Port"])

def numFlows(df):
	return getFlows(df).size

def plotFlows(df):
	df1 = getFlows(df)
	startTime = df1["Time"].iloc[0]
	endTime = df1["Time"].iloc[-1]
	binlist = list(range(startTime,endTime,3600))
	binlist.append(binlist[-1]+3600)
	counts = pd.cut(df1["Time"],bins = binlist, include_lowest=True).value_counts(sort=False)
	x = list(range(0,24))
	y = list(counts)
	plt.hist(x,x,weights=y)

	