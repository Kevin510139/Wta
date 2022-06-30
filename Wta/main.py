import time
import numpy as np
from src.model import Q_learning
from src.utils import load_greedy_ans, load_sink_data, write_reult, compute_all_tar, compute_value_of_q_table, load, trace_progress


def main():
    EPOCHS = 3000
    LEARNING_RATE = 0.2
    GAMMA = 0.95
    EPSILON = 0.1
    start = time.time()
    tar, weapon = load()
    sink_data = load_sink_data()
    data_ans = float(load_greedy_ans())
    Q_table = np.zeros((5,7))
    Q_table, value_best, value_after = Q_learning(
        Q_table,
        tar,
        weapon,
        sink_data,
        E = EPSILON,
        gamma = GAMMA,
        Ir = LEARNING_RATE,
        e = EPOCHS
    )
    trace_progress(
        value_after,
        data_ans,
        f"Wta_exploration"
    )
    trace_progress(
        value_best,
        data_ans,
        f"Wta_solution"
    )
    greedy_value,greedy_tar = compute_value_of_q_table(Q_table, tar, weapon, sink_data)
    write_reult(greedy_tar,greedy_value,Q_table)
    print(f"Time to run : {round(time.time() - start, 3)}")

    
    
if __name__ == "__main__":
    main()