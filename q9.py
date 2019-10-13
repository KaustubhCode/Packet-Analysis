import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read File
df = pd.read_csv("lbnl.anon-ftp.03-01-11.csv")
df.drop(['No.'],axis = 1)
# pick TCP packets
df_tcp = df[df.Protocol == "TCP"]
seq_num = []
ack_num = []
source_port= []
destination_port = []
# get seq num and ack num of all entries and append them
for i in range(len(df_tcp)):
	temp = df_tcp.iloc[i]['Info']
	numbers = re.findall(r'\d+', temp) 
	numbers_list = list(map(int, numbers)) 
	source_port.append(numbers_list[0])
	destination_port.append(numbers_list[1])
	seq_num.append(numbers_list[2])
	ack_num.append(numbers_list[3])

df_tcp['ack_num'] = ack_num
# df_tcp.loc[:,'ack_num'] = ack_num
df_tcp['seq_num'] = seq_num
df_tcp['source_port'] = source_port
df_tcp['destination_port'] = destination_port
df_tcp_server_side = df_tcp[df_tcp['source_port'] == 21]
df_tcp_client_side = df_tcp[df_tcp['destination_port'] == 21]
print(df_tcp_server_side.head(10))

def get_packet_tuple(df,ind):
	return df.iloc[ind]['Source'],df.iloc[ind]['Destination'],df.iloc[ind]['source_port'],df.iloc[ind]['destination_port']

seq_num_list = []
seq_num_set = set()
ack_num_list = []

def seq_num_plot(flow):
	rev_flow = (flow[1],flow[0],flow[3],flow[2])
	print(rev_flow)
	for ind in range(len(df_tcp_server_side)):
		if flow == tuple(get_packet_tuple(df_tcp_server_side,ind)):
			seq_num_list.append((df_tcp_server_side.iloc[ind]['Time'],df_tcp_server_side.iloc[ind]['seq_num']))
			seq_num_set.add(df_tcp_server_side.iloc[ind]['seq_num'])
	for ind in range(len(df_tcp_client_side)):
		if (rev_flow == tuple(get_packet_tuple(df_tcp_client_side,ind))) and (df_tcp_client_side.iloc[ind]['ack_num'] in seq_num_set):
			ack_num_list.append((df_tcp_client_side.iloc[ind]['Time'],df_tcp_client_side.iloc[ind]['ack_num']))
	print(len(seq_num_list))
	print(len(ack_num_list))
	x_axis_seq = [x[0] for x in seq_num_list]
	y_axis_seq = [y[1] for y in seq_num_list]	
	plt.scatter(x_axis_seq,y_axis_seq,color = 'g')
	x_axis_ack = [x[0] for x in ack_num_list]
	y_axis_ack = [y[1] for y in ack_num_list]
	plt.scatter(x_axis_ack,y_axis_ack,color = 'r')
	plt.show()
