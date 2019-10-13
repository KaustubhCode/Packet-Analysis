import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def cleanData(df):
	df['Info'] = df['Info'].apply(lambda x: x.strip())
	return df

def getTCP(df):
	df_tcp = df.loc[df["Protocol"] ==  "TCP"].copy()
	df_tcp.loc[:,'Source Port'] = df_tcp['Info'].apply(lambda x: int(x.split()[0]))
	df_tcp.loc[:,'Destination Port'] = df_tcp['Info'].apply(lambda x: int(x.split()[2]))
	return df_tcp

def getFTP(df):	
	return df.loc[df["Protocol"] ==  "FTP"]

def getSyn(df): # Does not consider duplicate sent acks
	return df[df['Info'].apply(lambda x: 'SYN' in x and ' ACK' not in x)]
	# return df.loc[df["Info"].str.contains('[SYN]', regex=False) ]

def getFirstSyn(df):
	return getSyn(df).drop_duplicates(subset=['Info'])

def getEnd(df):
	return getTCP(df.loc[df["Info"].str.contains('FIN|RST')])

# Q1
def getNums(df):
	df_syn = getSyn(df)
	# Below step redundant
	# df_syn = df_syn.drop_duplicates(subset=['Info']) 
	# Ack sent again (Considering that whole info can never match for a new connection)
	return (df_syn['Source'].drop_duplicates().size,df_syn['Destination'].drop_duplicates().size)

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
	# Remove all duplicate flows by keep = False
	df_syn = getTCP(getFirstSyn(df)).drop_duplicates(subset=["Source","Destination","Source Port","Destination Port"], keep = False)
	# Incorrect - Deletes unnecessary connections with resent SYN packets 
	# df_syn = getTCP(getSyn(df)).drop_duplicates(subset=["Source","Destination","Source Port","Destination Port"], keep = False);
	df_end = getEnd(df);
	mask = (df_end["Destination Port"] != 21)
	df_end.loc[mask, ["Source","Destination","Source Port","Destination Port"]] = df_end.loc[mask, ["Destination","Source","Destination Port","Source Port"]].values
	df_end = df_end.drop_duplicates(subset=["Source","Destination","Source Port","Destination Port"])	
	# Connections which give following error are not counted
	# Response: 426 Transfer aborted. Data connection closed and other random errors. (3 in total)
	df_merge = pd.merge(df_syn,df_end,on=["Source","Destination","Source Port","Destination Port"])
	df_merge["Connection Time"] = df_merge["Time_y"] -  df_merge["Time_x"]
	return df_merge

# Q4
def plotConnections(df):
	df_merge = getConnections(df)	
	conn = df_merge["Connection Time"].to_numpy()
	cumu = np.cumsum(conn/conn.sum())
	conn.sort()

	print("Mean of connection times:",conn.mean())
	print("Median of connection times:",np.median(conn))

	# Remove last two values as outliers - very large
	plt.plot(conn[:-20],cumu[:-20])
	plt.show()

# Q5
def connectionBytes(df):
	df_merge = getConnections(df)[["Source","Destination","Source Port","Destination Port","Connection Time"]]
	df_group = getTCP(df).groupby(["Source","Destination","Source Port","Destination Port"])
	df_length = {name:group["Length"].sum() for name,group in df_group}

	df_final = {}
	for index, row in df_merge.iterrows():
		a = tuple(row[0:4])
		b = (row[1],row[0],row[3],row[2])
		df_final[a] = [row[4],df_length[a],df_length[b]]

	X = []
	Y = []
	Z = []
	key = []
	for i in df_final:
		key.append(i)
		X.append(df_final[i][0])
		Y.append(df_final[i][1])
		Z.append(df_final[i][2])
	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)
	key = np.array(key)
	sort = X.argsort()
	key = key[sort]
	X = X[sort]
	Y = Y[sort]
	Z = Z[sort]

	plt.subplot(1, 2, 1)
	plt.scatter(X[:-4],Y[:-4],c='r')
	plt.title("Bytes sent (to server)")
	plt.xlabel("Connection time")
	plt.ylabel("Bytes sent per connection")

	plt.subplot(1, 2, 2)
	plt.scatter(X[:-4],Z[:-4],c='g')
	plt.title("Bytes Received (from server)")
	plt.xlabel("Connection time")
	plt.ylabel("Bytes sent per connection")
	# plt.scatter(X[:-2],Y[:-2] + Z[:-2],c='b')
	plt.show()

	#To implement PEARSONS

def getBusyConn(df):
	df_merge = getConnections(df)[["Source","Destination","Source Port","Destination Port","Connection Time"]]
	df_group = getTCP(df).groupby(["Source","Destination","Source Port","Destination Port"])
	df_length = {name:group["Length"].sum() for name,group in df_group}

	df_final = {}
	for index, row in df_merge.iterrows():
		a = tuple(row[0:4])
		b = (row[1],row[0],row[3],row[2])
		df_final[a] = [row[4],df_length[a],df_length[b]]

	X = []
	Y = []
	Z = []
	key = []
	for i in df_final:
		key.append(i)
		X.append(df_final[i][0])
		Y.append(df_final[i][1])
		Z.append(df_final[i][2])
	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)
	key = np.array(key)
	sort = X.argsort()
	key = key[sort]
	X = X[sort]
	Y = Y[sort]
	Z = Z[sort]

	ind = Z.argsort()
	Z = Z[ind]
	key = key[ind]
	# print("Max Size ", Z[-1], " Min Size: ", Z[0])
	return np.flip(key,0)

# Q6
def plotInterArrival(df):
	df_syn = getFirstSyn(df)
	conn_times = df_syn["Time"].to_numpy()
	inter_arrival = np.diff(conn_times)

	np.savetxt("inter_arrival_3.csv",inter_arrival,delimiter=",")

	cumu = np.cumsum(inter_arrival/inter_arrival.sum())

	print("Mean of connection times:",inter_arrival.mean())
	print("Median of connection times:",np.median(inter_arrival))

	inter_arrival.sort()
	# Remove last two values as outliers - very large
	plt.plot(inter_arrival,cumu)
	plt.title("CDF of inter-arrival times")
	plt.xlabel("Time")
	plt.show()

# Q7
def plotPacketInterArrival(df):
	df_tcp = getTCP(df)
	df_to_server = df_tcp[df_tcp["Destination Port"] == 21]
	packet_conn_times = df_to_server["Time"].to_numpy()
	packet_inter_arrival = np.diff(packet_conn_times)

	cumu = np.cumsum(packet_inter_arrival/packet_inter_arrival.sum())

	print("Mean of inter arrival times:",packet_inter_arrival.mean())
	print("Median of inter arrival times:",np.median(packet_inter_arrival))

	packet_inter_arrival.sort()
	# print(packet_inter_arrival[-10:])
	# Remove last two values as outliers - very large
	plt.plot(packet_inter_arrival,cumu)
	plt.title("CDF of packet inter-arrival times")
	plt.xlabel("Time (in s)")
	plt.show()

# Q8
def plotPacketLengths(df):
	df_tcp = getTCP(df)
	df_incoming = df_tcp[df_tcp["Destination Port"] != 21]
	df_outgoing = df_tcp[df_tcp["Destination Port"] == 21]
	packet_incoming = df_incoming["Length"].to_numpy()
	packet_outgoing = df_outgoing["Length"].to_numpy()

	cumu_incoming = np.cumsum(packet_incoming/packet_incoming.sum())
	cumu_outgoing = np.cumsum(packet_outgoing/packet_outgoing.sum())

	packet_incoming.sort()
	packet_outgoing.sort()
	plt.subplot(1, 2, 1)
	plt.plot(packet_incoming,cumu_incoming,'r')
	plt.title("CDF for incoming packets lengths")
	plt.xlabel("Packet size")

	plt.subplot(1, 2, 2)
	plt.plot(packet_outgoing,cumu_outgoing,'b')
	plt.title("CDF for outgoing packets lengths")
	plt.xlabel("Packet size")

	plt.show()

#Q9
def getSingleFlow(df, df_sel):
	mask1 = (df["Source"] == df_sel[0]) & \
					(df["Destination"] == df_sel[1]) & \
					(df["Source Port"] == df_sel[2]) & \
					(df["Destination Port"] == df_sel[3])
	mask2 = (df["Source"] == df_sel[1]) & (df["Destination"] == df_sel[0]) & (df["Source Port"] == df_sel[3]) & (df["Destination Port"] == df_sel[2])
	mask = mask1 | mask2
	df = df.loc[mask,:]
	return df

def getSeqAckNum(df):
	# should be TCP 
	df = df.loc[ ~df["Info"].str.contains('[SYN]',regex=False) ]
	df.loc[:,'Seq Num'] = df['Info'].apply( lambda x: int([int(i) for i in re.split(' |=', x) if i.isdigit()][2]) )
	df.loc[:,'Ack Num'] = df['Info'].apply(lambda x: int([int(i) for i in re.split(' |=', x) if i.isdigit()][3]))
	return df

# def getAckNum(df):
# def selectFlow(df, ind):

def processQ9(df):
	df_tcp = getTCP(df)
	df_tcp = getSeqAckNum(df_tcp)

# Q9a
def plotSeqNoPlots(df,ind):
	# get TCP and populate source, destination ports
	df_tcp = getTCP(df)
	# get flows
	df_flows = getFlows(df_tcp)
	# get index of maximum bytes flow
	# sel_index = 0
	# sel_tuple = [ df_flows["Source"].iloc[sel_index], df_flows["Destination"].iloc[sel_index], df_flows["Source Port"].iloc[sel_index], df_flows["Destination Port"].iloc[sel_index] ]
	# print(sel_tuple)

	# Get Highest data flow TCP Flow
	sel_tuple = getBusyConn(df_tcp)[ind]
	sel_tuple = [sel_tuple[0], sel_tuple[1], int(sel_tuple[2]), int(sel_tuple[3])]
	# sel_tuple = [sel_tuple[1], sel_tuple[0], sel_tuple[3], sel_tuple[2]]
	print("Selected Flow: ", sel_tuple)
	sel_flow = getSingleFlow(df_tcp, sel_tuple)
	sel_flow = getSeqAckNum(sel_flow)

	# print(sel_flow)

	server_seq = sel_flow.loc[sel_flow["Source Port"] == 21]
	client_ack = sel_flow.loc[sel_flow["Destination Port"] == 21]
	y_seq = []
	x_seq = []
	y_ack = []
	x_ack = []
	for i in range(server_seq.shape[0]):
		if (int(server_seq["Seq Num"].iloc[i]) != 0):
			y_seq.append(int(server_seq["Seq Num"].iloc[i]))
			x_seq.append(int(server_seq["Time"].iloc[i]))

	for i in range(client_ack.shape[0]):
		if (int(client_ack["Ack Num"].iloc[i]) != 0):
			y_ack.append(int(client_ack["Ack Num"].iloc[i]))
			x_ack.append(int(client_ack["Time"].iloc[i]))

	plt.scatter(x_seq,y_seq,color='r',marker='x')
	plt.scatter(x_ack,y_ack,color='g',marker='+')
	plt.show()
	# select a flow
	# df_seq = getSeqNum(df_tcp)
	# return df_seq


if __name__ == "__main__":
	df1 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-11.csv"))
	df2 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-14.csv"))
	df3 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-18.csv"))

	print(getNums(df2))
	# print(numFlows(df1))
	# plotFlows(df1)
	# plotConnections(df1)
	# plotInterArrival(df3)
	# plotPacketInterArrival(df1)
	# df_new = getSeqNum(df1)
	# print(df_new.head())
	# plotSeqNoPlots(df1)
	# plotPacketLengths(df1)
	
	# Q9a (Plot High Traffic TCP Flows)
	# plotSeqNoPlots(df1,0)
	# plotSeqNoPlots(df2,0)
	# plotSeqNoPlots(df3,0)
