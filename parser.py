import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cleanData(df):
	df['Info'] = df['Info'].apply(lambda x: x.strip())
	return df

def getTCP(df):
	df_tcp = df.loc[df["Protocol"] ==  "TCP"]
	df_tcp["Source Port"] = df_tcp['Info'].apply(lambda x: int(x.split()[0]))
	df_tcp['Destination Port'] = df_tcp['Info'].apply(lambda x: int(x.split()[2]))
	return df_tcp

def getFTP(df):	
	return df.loc[df["Protocol"] ==  "FTP"]

def getSyn(df):
	return df.loc[df["Info"].str.contains('[SYN]',regex=False) ]

def getEnd(df):
	df_new = df.loc[df["Info"].str.contains('FIN|RST')] 
	return df.loc[df["Info"].str.contains('FIN|RST')]

# Q1
def getNums(df):
	df_syn = getSyn(df)
	df_first_syn = df_syn.drop_duplicates(subset=['Info']) # Ack sent again (Considering that whole info can never match for a new connection)
	return (df_first_syn['Source'].drop_duplicates().size,df_first_syn['Destination'].drop_duplicates().size)

def getFlows(df):
	df_flow = getTCP(getSyn(df))
	return df_flow.drop_duplicates(subset=["Source","Destination","Source Port","Destination Port"])

# Q2
def numFlows(df):
	return getFlows(df).shape[0]
# Q3
def plotFlows(df):
	df_flow = getFlows(df)
	startTime = df_flow["Time"].iloc[0]
	endTime = df_flow["Time"].iloc[-1]
	binlist = list(range(int(startTime),int(endTime)+1,3600))
	binlist.append(binlist[-1]+3600)
	counts = pd.cut(df_flow["Time"],bins = binlist, include_lowest=True).value_counts(sort=False)
	x = list(range(0,25))
	y = list(counts)
	plt.hist(x[:-1],x,weights=y)
	plt.show()

def getConnections(df):
	df_syn = getSyn(df);
	df_end = getEnd(df);

if __name__ == "__main__":
	df1 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-11.csv"))
	df2 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-14.csv"))
	df3 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-18.csv"))

	print(getNums(df1))
	print(numFlows(df1))

