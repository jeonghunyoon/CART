#-*- coding: utf-8 -*-

from numpy import *

class RegTree:
    def __init__(self):
        # feature : a features used for dividing this node
        self.feature = 0
        # value : a value used for dividing this node
        self.value = 0
        self.right = None
        self.left = None

    def bin_split_X(self, X, split_feat, split_val):
        sub_1 = X[nonzero(X[:, split_feat] > split_val)[0], :]
        sub_2 = X[nonzero(X[:, split_feat] <= split_val)[0], :]
        return sub_1, sub_2

    # ops의 첫번째 원소는 오차범위이고, ops의 2번째 원소는 분할된 node에 포함될 수 있는 최소 원소의 갯수이다
    def choose_best_split(self, X, leaf_type=reg_type, err_type=reg_err, ops=(1,4)):
        tol_s = ops[0]
        tol_n = ops[1]

        # 포함된 원소의 label값이 모두 같으면 분할하지 않는다
        if len(set(X[:, -1].T.tolist()[0])) == 1:
            return None, leaf_type(X)

        m, n = shape(X)

        # 현재 node의 disorder를 측정
        s = err_type(X)

        new_s = inf
        best_s = inf
        best_feat = 0
        best_val = 0

        # greedy 방식
        for idx in range(n-1):
            for val in set(X[:, idx]):
                sub_1, sub_2 = self.bin_split_X(X, idx, val)
                # tol_n 보다 작으면, split 할 필요가 없다.
                if (shape(sub_1)[0] < tol_n) or (shape(sub_2)[0] <tol_n):
                    continue
                new_s = err_type(sub_1) + err_type(sub_2)
                # 전체 제곱 오류가 작을수록 좋다.
                if new_s < best_s:
                    best_feat = idx
                    best_val = val
                    best_s = new_s

        # 전체 오류의 변동량이 tol_s 보다 작으면, split 할 필요가 없다.
        if abs(s-new_s) < tol_s:
            return None, leaf_type(X)

        # 위의 continue 문에 걸린 case만 존재할 때
        sub_1, sub_2 = self.bin_split_X(X, best_feat, best_val)
        if (shape(sub_1)[0] < tol_n) or (shape(sub_2)[0] < tol_n):
            return None, leaf_type(X)

        return best_feat, best_val


    def create_tree(self, X, leaf_type=reg_type, err_type=reg_err, ops=(1,4)):
        split_feat, split_val = self.choose_best_split(X, leaf_type, err_type, ops)
        if split_feat == None:
            return split_val

        l_set, r_set = self.bin_split_X(X, split_feat, split_val)

        reg_tree = {}
        reg_tree['split_feat'] = split_feat
        reg_tree['split_val'] = split_val
        reg_tree['left'] = self.create_tree(l_set, leaf_type, err_type, ops)
        reg_tree['right'] = self.create_tree(r_set, leaf_type, err_type, ops)

        return reg_tree


def data_loader(file_name):
    training_set = []
    file_reader = open(file_name)
    for line in file_reader.readlines():
        tokens = line.strip().split('\t')
        float_tokens = map(float, tokens)
        training_set.append(float_tokens)
    return training_set


def reg_type(X):
    return mean(X[:, -1])

def reg_err(X):
    return var(X[:, -1]) * shape(X)[0]