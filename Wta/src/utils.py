import matplotlib.pyplot as plt
from numba import njit
from prettytable import PrettyTable
import numpy as np
import csv

def load():
    data = np.load("data/0.npy",allow_pickle= True).item()
    weapon = data['weapon']['type']
    tar_type = data['target']['type']
    tar_threat = data['target']['threat']
    tar = np.stack((tar_type, tar_threat))
    return tar, weapon

def load_sink_data():
    file = open('data/sink.csv')
    reader = csv.reader(file)
    sink_data_list = list(reader)
    file.close()
    return sink_data_list

@njit
def next_max_point(Q_table: np.ndarray, column:int, mask: np.ndarray):
    argmax = 0
    max_v = -np.inf
    idx = np.arange(Q_table.shape[0])
    np.random.shuffle(idx)
    for i in idx:
        if not (mask[i]):
            continue
        if Q_table[i,column]> max_v:
            argmax = i
            max_v = Q_table[i,column]
    return argmax, max_v

def compute_all_tar(Q_table: np.ndarray, weapon: np.ndarray, tar:np.ndarray,sink_data:np.ndarray):
    N = Q_table.shape[1]
    M = Q_table.shape[0]
    mask = np.array([True] * M)
    attack_tar = np.zeros((N,))
    new_sink_data = sink_rate(tar,sink_data)
    for i in range(N):
        current = attack_tar[i]
        next_tar, _ = next_max_point(Q_table, int(current), mask)
        sink = new_sink_data[next_tar][weapon[i]] -1
        new_sink_data[next_tar][weapon[i]] = sink
        if sink < 0:
            mask[next_tar] = False
        attack_tar[i] = next_tar
    return attack_tar

def sink_rate(tar:np.ndarray, sink_data:np.ndarray):
    N = tar.shape[1]
    new_sink_data = np.zeros((N,2))
    for i in range(N):
        tar_type = int(tar[0][i])
        new_sink_data[i][0] = sink_data[tar_type][0]
        new_sink_data[i][1] = sink_data[tar_type][1]
    return new_sink_data


def reward_calculate(tar:np.ndarray,nxt_tar:int,weapon:int,sink:float):
    if sink < 0:
        reward = tar[1][nxt_tar]
    else:
        sink_data = load_sink_data()
        orig_sink = float(sink_data[int(tar[0][nxt_tar])][weapon])
        reward = 0.5 * (orig_sink - sink)/orig_sink * tar[1][nxt_tar]
    return reward

def compute_value_of_q_table(Q_table:np.ndarray, tar:np.ndarray, weapon:np.ndarray, sink_data:np.ndarray):
    N = Q_table.shape[1]
    M = Q_table.shape[0]
    mask = np.array([True] * M)
    attack_tar = np.zeros((N,))
    new_sink_data = sink_rate(tar,sink_data)
    for i in range(N):
        current = attack_tar[i]
        next_tar, _ = next_max_point(Q_table, int(current), mask)
        sink = new_sink_data[next_tar][weapon[i]] -1
        new_sink_data[next_tar][weapon[i]] = sink
        if sink < 0:
            mask[next_tar] = False
        attack_tar[i] = next_tar
    greedy_value = 0
    orig_sink = sink_rate(tar,sink_data)
    for i in range(tar.shape[1]):
        if new_sink_data[i][0] < 0 or new_sink_data[i][1] < 0:
            greedy_value += tar[1][i]
        elif (orig_sink[i][0]-new_sink_data[i][0])/orig_sink[i][0] >= (orig_sink[i][1]-new_sink_data[i][1])/orig_sink[i][1] :
            greedy_value += 0.5 * (orig_sink[i][0]-new_sink_data[i][0])/orig_sink[i][0] * tar[1][i]
        else:
            greedy_value += 0.5 * (orig_sink[i][1]-new_sink_data[i][1])/orig_sink[i][1] * tar[1][i]
    return greedy_value


def trace_progress(values: np.ndarray, true_best: float, tag: str):
    plt.figure(figsize=(19, 7))
    plt.plot(values, label="Test values")
    plt.hlines(
        true_best,
        xmin=0,
        xmax=len(values),
        colors="r",
        label="Best distance"
    )
    plt.title(tag)
    plt.legend()
    plt.savefig(f"answer/Evolution_{tag}")

def write_reult(greedy_tar:np.ndarray, greedy_value: int, Q_table:np.ndarray):
    table =PrettyTable()
    table.field_names = [
        "Weapon to target",
        "all value"
    ]
    table.add_row([greedy_tar, greedy_value])
    with open(f"answer/Results.txt","w") as f:
        f.write(str(table)+'\n')
        f.write(str(Q_table))
