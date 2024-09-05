import torch
import numpy as np


def viterbi_align(logit, labels, bound_pred):
    emission = torch.softmax(logit, dim=-1)
    probs = emission.numpy()
    T, N = probs.shape
    labels = labels.copy()

    C = len(labels)
    dp = np.zeros((T + 1, C + 1))
    path = np.zeros((T + 1, C + 1), dtype=int)

    # 初始化
    dp[0, :] = -np.inf
    dp[:, 0] = -np.inf
    dp[0, 0] = 0

    # 填充DP表
    for t in range(1, T + 1):
        for c in range(1, C + 1):
            choices = dp[t - 1, c - 1], dp[t - 1, c]
            if t >= 2 and bound_pred is not None and bound_pred[t - 1] > 0.8:
                path[t, c] = 0
            else:
                path[t, c] = np.argmax(choices)
            dp[t, c] = choices[path[t, c]] + np.log(probs[t - 1, labels[c - 1]])

    # 回溯找路径
    alignment = []
    t, c = T, C
    while t > 0 and c > 0:
        alignment.append((t, c))
        if path[t, c] == 0:  # 对应dp[t-1, c-1]
            t, c = t - 1, c - 1
        elif path[t, c] == 1:  # 对应dp[t-1, c]
            t -= 1

    alignment.reverse()
    # 将帧映射到标签
    frame_to_label = [0] * T
    for frame_index, label_index in alignment:
        frame_to_label[frame_index - 1] = labels[label_index - 1]

    return frame_to_label