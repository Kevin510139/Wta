import numpy as np
from src.utils import next_max_point,sink_rate,reward_calculate,compute_value_of_q_table

def eps_update(
    Q_table: np.ndarray,
    mask:np.ndarray,
    tar:np.ndarray,
    weapon:np.ndarray,
    sink_data:np.ndarray,
    E: float,
    gamma: float,
    lr: float,
    N: int,
):
    
    new_sink_data = sink_rate(tar,sink_data)
    orig_sink_data = new_sink_data.copy()
    for i in range(N):
        reward = 0
        possible = np.where(mask == True)[0]
        u = np.random.random()
        if u < E:
            next_target = np.random.choice(possible)
        else:
            next_target,_ = next_max_point(Q_table,i,mask)
        new_sink_data[next_target][weapon[i]] -=1

        if new_sink_data[next_target][weapon[i]] < 0:
            mask[next_target] = False
            reward = tar[1][next_target]
        else:
            reward = reward_calculate(tar, next_target, weapon[i], new_sink_data, orig_sink_data)
        
        _, max_next = next_max_point(Q_table,i,mask)
        Q_table[next_target][i] = (1-lr)*(Q_table[next_target][i]) + lr * (reward + gamma * max_next) 
    return Q_table

def Q_learning(
    Q_table: np.ndarray,
    tar:np.ndarray,
    weapon:np.ndarray,
    sink_data:np.ndarray,
    E: float,
    gamma: float,
    Ir: float,
    e: int =100,
):
    N = Q_table.shape[1]
    M = Q_table.shape[0]
    AfterQ_table = Q_table.copy()
    mask = np.array([True] * M)
    value_best = np.zeros((e,))
    value_after = np.zeros((e,))
    for i in range(e):
        AfterQ_table = eps_update(
            AfterQ_table, mask, tar, weapon, sink_data, E, gamma, Ir, N
        )
        cost,_ = compute_value_of_q_table(Q_table,tar,weapon,sink_data)
        After_cost,_ = compute_value_of_q_table(AfterQ_table,tar,weapon,sink_data) 
        value_best[i] = cost
        value_after[i] = After_cost
        if After_cost > cost:
            Q_table[:, :] = AfterQ_table[:, :]
        mask[:] = True
    return Q_table, value_best, value_after
