import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

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

def getFlows(df):
	df_flow = getTCP(getSyn(df))
	return df_flow.drop_duplicates(subset=["Source","Destination","Source Port","Destination Port"])

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

# Q1
def getNums(df):
	df_syn = getSyn(df)
	# Below step redundant
	# df_syn = df_syn.drop_duplicates(subset=['Info']) 
	# Ack sent again (Considering that whole info can never match for a new connection)
	return (df_syn['Source'].drop_duplicates().size,df_syn['Destination'].drop_duplicates().size)

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

# Q4
def plotConnections(df):
	df_merge = getConnections(df)	
	conn = df_merge["Connection Time"].to_numpy()
	cumu = np.cumsum(conn/conn.sum())
	conn.sort()

	print("Mean of connection times:",conn.mean())
	print("Median of connection times:",np.median(conn))

	# Remove last two values as outliers - very large
	plt.plot(conn[:-2],cumu[:-2])
	plt.show()

# Q5
def connectionBytes(df,k):
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
	plt.ylabel("Bytes received per connection")
	# plt.scatter(X[:-2],Y[:-2] + Z[:-2],c='b')
	plt.show()

	#To implement PEARSONS
	print("Pearson's coefficient of bytes sent to server and connection duration:",pearsonr(X,Y)[0])
	print("Pearson's coefficient of bytes received to server and connection duration:",pearsonr(X,Z)[0])
	print("Pearson's coefficient of bytes sent and received:",pearsonr(Y,Z)[0])

	X_o = []
	Y_o = []
	Z_o = []

	params = [[1250,40000,10000], [2500, 75000, 25000], [1700,75000,50000]]

	outlier_part = params[k-1]
	#Removing outliers - by outliers we mean unusual data connections - most connections are short and send less bytes
	for i in range(len(X)):
		if (X[i] < outlier_part[0] and Y[i] < outlier_part[1] and Z[i] < outlier_part[2]):
			X_o.append(X[i])
			Y_o.append(Y[i])
			Z_o.append(Z[i])

	print("Outlier removal - Pearson's coefficient of bytes sent to server and connection duration:",pearsonr(X_o,Y_o)[0])
	print("Outlier removal - Pearson's coefficient of bytes received to server and connection duration:",pearsonr(X_o,Z_o)[0])
	print("Outlier removal - Pearson's coefficient of bytes sent and received:",pearsonr(Y_o,Z_o)[0])

	plt.scatter(Y_o,Z_o)
	plt.title("Bytes sent vs bytes received")
	plt.xlabel("Bytes sent")
	plt.ylabel("Bytes received")
	plt.show()

# Q6
def plotInterArrival(df):
	df_syn = getFirstSyn(df)
	conn_times = df_syn["Time"].to_numpy()
	inter_arrival = np.diff(conn_times)

	# np.savetxt("inter_arrival_3.csv",inter_arrival,delimiter=",")

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

	# np.savetxt("packet_inter_arrival_3.csv",packet_inter_arrival,delimiter=",")

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

	pack_in_mean = packet_incoming.mean()
	pack_out_mean = packet_outgoing.mean()
	print("Incoming Packet Mean: ", pack_in_mean, " Outgoing packet Mean: ", pack_out_mean)
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

def getTupleHighFlow(df, ind):
	df_tcp = getTCP(df)
	# Get Highest data flow TCP Flow
	sel_tuple = getBusyConn(df_tcp)[ind]
	sel_tuple = [sel_tuple[0], sel_tuple[1], int(sel_tuple[2]), int(sel_tuple[3])]
	return sel_tuple

# Q9a
def plotSeqNoPlots(df,sel_tuple):
	# get TCP and populate source, destination ports
	df_tcp = getTCP(df)
	# get flows
	# df_flows = getFlows(df_tcp)
	# get index of maximum bytes flow
	# sel_index = 0
	# sel_tuple = [ df_flows["Source"].iloc[sel_index], df_flows["Destination"].iloc[sel_index], df_flows["Source Port"].iloc[sel_index], df_flows["Destination Port"].iloc[sel_index] ]
	# print(sel_tuple)

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

def getRetransmissions(df,skip):
	df_tcp = getTCP(df)
	df_flows = getFlows(df_tcp)
	retransmit_flow = []
	retransmit_time = []
	packet_no = []
	for sel_index in range(df_flows.shape[0]):
		sel_tuple = [ df_flows["Source"].iloc[sel_index], df_flows["Destination"].iloc[sel_index], df_flows["Source Port"].iloc[sel_index], df_flows["Destination Port"].iloc[sel_index] ]
		sel_flow = getSingleFlow(df_tcp, sel_tuple)
		sel_flow = getSeqAckNum(sel_flow)
		sel_flow = sel_flow.loc[sel_flow["Source Port"] == 21]
		retransmit_count = 0
		for i in range(sel_flow.shape[0]-1):
			if sel_flow["Seq Num"].iloc[i] == sel_flow["Seq Num"].iloc[i+1]:
				retransmit_count += 1
				re_time = sel_flow["Time"].iloc[i]
				re_time2 = sel_flow["Time"].iloc[i+1]
				pack_num = sel_flow["Seq Num"].iloc[i]
		if retransmit_count > 2:
			if skip > 0:
				skip-=1
			else:
				retransmit_flow.append(sel_index)
				retransmit_time.append(re_time)
				retransmit_time.append(re_time2)
				packet_no.append(pack_num)
				break
	# print("Connections with retransmissions: ", retransmit_flow)
	return (retransmit_flow[0], retransmit_time, packet_no)

def getSpuriousRetrans(df,skip):
	df_tcp = getTCP(df)
	df_flows = getFlows(df_tcp)
	# print(df_flows)
	retransmit_flow = []
	retransmit_time = []
	for sel_index in range(df_flows.shape[0]):
		sel_tuple = [ df_flows["Source"].iloc[sel_index], df_flows["Destination"].iloc[sel_index], df_flows["Source Port"].iloc[sel_index], df_flows["Destination Port"].iloc[sel_index] ]
		sel_flow = getSingleFlow(df_tcp, sel_tuple)
		sel_flow = getSeqAckNum(sel_flow)
		highestAck = 0
		prev_seq_num = 0
		for i in range(sel_flow.shape[0]):
			if sel_flow["Destination Port"].iloc[i] == 21:
				highestAck = sel_flow["Ack Num"].iloc[i]
			if sel_flow["Source Port"].iloc[i] == 21:
				seq_num = sel_flow["Seq Num"].iloc[i]
				if seq_num < highestAck and seq_num == prev_seq_num:
					if skip >0:
						skip-=1
					else:
						print("Seq Num: ", seq_num, " Ack Num: ", highestAck)
						return(sel_index)
				prev_seq_num = seq_num

def getDuplicateAcks(df, skip):
	df_tcp = getTCP(df)
	df_flows = getFlows(df_tcp)
	# print(df_flows)
	retransmit_flow = []
	retransmit_time = []
	ack_list = []
	for sel_index in range(df_flows.shape[0]):
		sel_tuple = [ df_flows["Source"].iloc[sel_index], df_flows["Destination"].iloc[sel_index], df_flows["Source Port"].iloc[sel_index], df_flows["Destination Port"].iloc[sel_index] ]
		sel_flow = getSingleFlow(df_tcp, sel_tuple)
		sel_flow = sel_flow.loc[sel_flow["Destination Port"] == 21]
		sel_flow = getSeqAckNum(sel_flow)
		prev_ack = 0
		dup_ack = False
		for i in range(sel_flow.shape[0]):
			ack_no = sel_flow["Ack Num"].iloc[i]
			if ack_no == prev_ack:
				dup_ack = True
				ack_list.append(ack_no)
				break
			prev_ack = ack_no
		if dup_ack == True and skip > 0:
			skip -= 1
		elif dup_ack == True:
			print("Dup Acks Time: ",sel_flow["Time"].iloc[i-1], " ", sel_flow["Time"].iloc[i])
			return sel_index, ack_list

def getTupleFromFlow(df,sel_index):
	df_tcp = getTCP(df)
	df_flows = getFlows(df_tcp)
	sel_tuple = [ df_flows["Source"].iloc[sel_index], df_flows["Destination"].iloc[sel_index], df_flows["Source Port"].iloc[sel_index], df_flows["Destination Port"].iloc[sel_index] ]
	return sel_tuple

# Q11
def exp_func(x, a):
	# return a * np.exp(-b * x) + c
	return 1 - np.exp(-a * x)

def getInterArrival(df):
	df_syn = getFirstSyn(df)
	conn_times = df_syn["Time"].to_numpy()
	inter_arrival = np.diff(conn_times)

	cumu = np.cumsum(inter_arrival/inter_arrival.sum())

	print("Mean of connection times:",inter_arrival.mean())
	print("Median of connection times:",np.median(inter_arrival))

	inter_arrival.sort()
	# Remove last two values as outliers - very large
	return (inter_arrival,cumu)

# Q8
def getMeanPacketLengths(df):
	df_tcp = getTCP(df)
	df_incoming = df_tcp[df_tcp["Destination Port"] != 21]
	df_outgoing = df_tcp[df_tcp["Destination Port"] == 21]
	packet_incoming = df_incoming["Length"].to_numpy()
	packet_outgoing = df_outgoing["Length"].to_numpy()

	pack_in_mean = packet_incoming.mean()
	pack_out_mean = packet_outgoing.mean()
	print("Incoming Packet Mean: ", pack_in_mean, " Outgoing packet Mean: ", pack_out_mean)
	cumu_incoming = np.cumsum(packet_incoming/packet_incoming.sum())
	cumu_outgoing = np.cumsum(packet_outgoing/packet_outgoing.sum())
	return [pack_in_mean, pack_out_mean]

if __name__ == "__main__":
	df1 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-11.csv"))
	df2 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-14.csv"))
	df3 = cleanData(pd.read_csv("lbnl.anon-ftp.03-01-18.csv"))

	# Change this for each dataset
	k = 1
	if (k == 1):
		df = df1
	elif (k == 2):
		df = df2
	else:
		df = df3

	# Q1
	# print(getNums(df))

	# Q2
	# print(numFlows(df))

	# Q3
	# plotFlows(df)

	# Q4
	# plotConnections(df)

	# Q5
	# connectionBytes(df,k)

	# Q6
	# plotInterArrival(df)

	# Q7
	# plotPacketInterArrival(df)
	
	# df_new = getSeqNum(df1)
	# print(df_new.head())
	# plotSeqNoPlots(df1)

	# Q8 
	# plotPacketLengths(df1)
	
	# Q9a (Plot High Traffic TCP Flows)
	# plotSeqNoPlots(df1,getTupleHighFlow(df1,0))
	# plotSeqNoPlots(df2,getTupleHighFlow(df2,0))
	# plotSeqNoPlots(df3,getTupleHighFlow(df3,0))

	# Q9b (Retransmissions)	(2 plots - df1, df3)
	# df = df3
	# [sel_index, time, pack_no] = getRetransmissions(df,0)
	# print("Packet No: ", pack_no, " Time: ", time)
	# if sel_index != None:
	# 	sel_tuple = getTupleFromFlow(df,sel_index)
	# 	plotSeqNoPlots(df,sel_tuple)

	# Q9c Spurious Retransmission
	# df = df3
	# sel_index = getSpuriousRetrans(df,0)
	# print(sel_index)
	# # sel_index = 602
	# if sel_index != None:
	# 	sel_tuple = getTupleFromFlow(df,sel_index)
	# 	plotSeqNoPlots(df,sel_tuple)

	# Q9d Duplicate ACKs (df1, df2)
	# df = df1
	# [sel_index, ack_list] = getDuplicateAcks(df,0)
	# if sel_index != None:
	# 	print("Duplicate Ack Number: ", ack_list)
	# 	sel_tuple = getTupleFromFlow(df,sel_index)
	# 	plotSeqNoPlots(df,sel_tuple)

	# Q9e Out-of-Order Delivery

	# ## Q11 
	# ## (Get lambda)
	# df = df1
	# [inter_arrival, cumu] = getInterArrival(df)
	# popt, pcov = curve_fit(exp_func, inter_arrival, cumu)
	# lamb = float(popt[0])
	# print("lambda: ", lamb)
	# print("Mean Inter Arrival Time:", 1/lamb)
	# plt.plot(inter_arrival, cumu)
	# plt.plot(inter_arrival, exp_func(inter_arrival, *popt), 'r-', label='exp fit: lambda=%5.3f' % tuple(popt))
	# plt.legend()
	# plt.show()
	# ## Get Mu
	# [pack_in_mean, pack_out_mean] = getMeanPacketLengths(df)
	# link_speed = 128 * 1000 / 8
	# mu = link_speed/pack_in_mean
	# print("Link Speed: ", link_speed, " bytes per second")
	# print("Mu: ", mu)
	# ## Get Utilization factor
	# rho = lamb / mu
	# print("Rho: ", rho)
	# ## Get Queue Size
	# avg_queue_size = lamb / (mu - lamb)
	# print("Average Queue Size: ", avg_queue_size)
	# ## Get Average Waiting Time
	# avg_wait_time = 1/(mu - lamb) - 1/mu
	# print("Average Wait Time: ", avg_wait_time)
	# ## plot lambda vs queue size and lambda vs wait time
	# la = [i / 100 * mu for i in range(0, 100)]
	# qu_plot = [i / (mu - i) for i in la]
	# wait_time_plot = [1/(mu - i) - 1/(mu) for i in la]
	# plt.plot(la,qu_plot,'r-',label='Avg Queue Size')
	# plt.xlabel('Lambda')
	# plt.ylabel('Avg Queue Size')
	# plt.legend()
	# plt.show()
	# plt.plot(la,wait_time_plot,'g-',label='Avg Wait Time')
	# plt.xlabel('Lambda')
	# plt.ylabel('Avg Wait Time')
	# plt.legend()
	# plt.show()
