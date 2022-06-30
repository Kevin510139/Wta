import time
import numpy as np
import csv
from prettytable import PrettyTable


start = time.time()
data = np.load("data/3.npy",allow_pickle= True).item()
weapon = data['weapon']['type']
tar_type = data['target']['type']
tar_threat = data['target']['threat']
tar = np.stack((tar_type, tar_threat))
file = open('data/sink.csv')
reader = csv.reader(file)
sink_data_list = list(reader)
file.close()
N = int(tar.shape[1])
sink_data = np.zeros((N,2))
for i in range(N):
    sink_data[i][0] = sink_data_list[tar_type[i]][0]
    sink_data[i][1] = sink_data_list[tar_type[i]][1]

orig_sink = sink_data.copy()

max_reward = 0
i = 0
for w1 in range(N):
    for w2 in range(N):
        for w3 in range(N):
            for w4 in range(N):
                for w5 in range(N):
                    for w6 in range(N):
                        for w7 in range(N):
                            reward = 0
                            sink_data[w1][weapon[0]]-=1
                            sink_data[w2][weapon[1]]-=1
                            sink_data[w3][weapon[2]]-=1
                            sink_data[w4][weapon[3]]-=1
                            sink_data[w5][weapon[4]]-=1
                            sink_data[w6][weapon[5]]-=1
                            sink_data[w7][weapon[6]]-=1
                            for i in range(N):
                                if sink_data[i][0] < 0 or sink_data[i][1] < 0:
                                    reward += tar_threat[i]
                                elif sink_data[i][0] == orig_sink[i][0] and sink_data[i][1] == orig_sink[i][1]:
                                    reward += 0
                                elif (orig_sink[i][0]-sink_data[i][0])/orig_sink[i][0] > (orig_sink[i][1]-sink_data[i][1])/orig_sink[i][1] :
                                        reward += 0.5 * (orig_sink[i][0]-sink_data[i][0])/orig_sink[i][0] * tar_threat[i]
                                else:
                                        reward += 0.5 * (orig_sink[i][1]-sink_data[i][1])/orig_sink[i][1] * tar_threat[i]
                            
                            if reward > max_reward:
                                max_reward = reward
                                attack_tar = [w1,w2,w3,w4,w5,w6,w7]
                            
                            sink_data = orig_sink.copy()


table =PrettyTable()
table.field_names = [
    "Weapon to target",
    "value"
]
table.add_row([attack_tar, max_reward])

with open(f"answer/Greedy_Results.txt","w") as f:
    f.write(str(table)+'\n')

with open(f"data/bruteanswer.txt","w") as f:
    f.write(str(max_reward))

print(f"Time to run : {round(time.time() - start, 3)}")
