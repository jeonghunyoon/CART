#-*- coding: utf-8 -*-

from numpy import *

def reg_type(X):
    return mean(X[:, -1])


def reg_err(X):
    return var(X[:, -1]) * shape(X)[0]


def linear_solver(data_set):
    m, n = shape(data_set)
    X = matrix(ones((m, n)))
    y = matrix(ones((m, 1)))

    X[:, 1:n] = data_set[:, 0:n-1]
    y = data_set[:, -1]

    if linalg.det(X.T * X) == 0:
        raise NameError('This matrix is singular, cannot do inverse, \n \
                         try increasing the second value of options')

    w = linalg.inv(X.T * X) * X.T * y

    return w, X, y


def model_type(X):
    w, X, y = linear_solver(X)
    return w


def model_err(X):
    w, X, y = linear_solver(X)
    y_hat = X*w

    return sum(power(y - y_hat, 2))


# check whether leaf node is
def isTree(object):
    if (type(object).__name__ == 'dict'):
        return True
    else:
        return False


def getMean(tree):
    if (isTree(tree['left'])):
        tree['left'] = getMean(tree['left'])
    if (isTree(tree['right'])):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2


# 모델을 평가하기 위한 함수 (regression tree)
def reg_tree_eval(model, input_data):
    return float(model)


# 모델을 평가하기 위한 함수 (model tree)
def model_tree_eval(model, input_data):
    n = shape(input_data)[1]
    # (1, x_1, x_2, ..., x_n)의 형태가 되어야 함.
    X = matrix(ones((1, n+1)))
    X[:, 1:n+1] = input_data

    return float(X*model)


# input data 1개에 대하여 예측값을 계산해준다.
def tree_fore_cast(tree, input_data, model_eval = reg_tree_eval):
    if not isTree(tree):
        return tree
    if input_data[tree['split_feat']] > tree['split_val']:
        if isTree(tree['left']):
            return tree_fore_cast(tree['left'], input_data, model_eval)
        else:
            return model_eval(tree['left'], input_data)
    else:
        if isTree(tree['right']):
            return tree_fore_cast(tree['right'], input_data, model_eval)
        else:
            return model_eval(tree['right'], input_data)


# input data 전체에 대하여 예측값을 계산한다. 즉 내부적으로 tree_fore_cast를 호출한다.
def create_fore_Cast(tree, test_data, model_eval = reg_tree_eval):
    m = len(test_data)
    y_hat = matrix(zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_fore_cast(tree, test_data[i], model_eval)

    return y_hat


class ModelTree:
    def __init__(self):
        # feature : a features used for dividing this node
        self.feature = 0
        # value : a value used for dividing this node
        self.value = 0
        self.right = None
        self.left = None


    def bin_split_X(self, X, split_feat, split_val):
        sub_1 = X[nonzero(X[:, split_feat] > split_val)[0], :][0]
        sub_2 = X[nonzero(X[:, split_feat] <= split_val)[0], :][0]
        return sub_1, sub_2


    # ops의 첫번째 원소는 오차범위이고, ops의 2번째 원소는 분할된 node에 포함될 수 있는 최소 원소의 갯수이다
    def choose_best_split(self, X, leaf_type, err_type, options=(1,4)):
        tol_s = options[0]
        tol_n = options[1]

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
            # python 2.7+ 에서는 set(matrix)가 unhashable type 에러를 발생시킴
            # idxs = []
            # col = X[:, idx].tolist()
            # for i in range(len(col)):
            #    idxs.append(col[i][0])
            # for val in idxs:

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


    def create_tree(self, X, leaf_type=reg_type, err_type=reg_err, options=(1,4)):
        split_feat, split_val = self.choose_best_split(X, leaf_type, err_type, options)
        if split_feat == None:
            return split_val

        l_set, r_set = self.bin_split_X(X, split_feat, split_val)

        reg_tree = {}
        reg_tree['split_feat'] = split_feat
        reg_tree['split_val'] = split_val
        reg_tree['left'] = self.create_tree(l_set, leaf_type, err_type, options)
        reg_tree['right'] = self.create_tree(r_set, leaf_type, err_type, options)

        return reg_tree


    def prune(self, tree, valid_data):
        if shape(valid_data)[0] == 0:
            return getMean(tree)
        l_set, r_set = self.bin_split_X(valid_data, tree['split_feat'], tree['split_val'])
        if isTree(tree['right']):
            tree['right'] = self.prune(tree['right'], r_set)
        if isTree(tree['left']):
            tree['left'] = self.prune(tree['left'], l_set)
        # print tree
        # print l_set
        # print r_set
        if not isTree(tree['right']) and not isTree(tree['left']):
            error_no_merge = sum(power((l_set[:, -1] - tree['left']), 2)) + \
                             sum(power((r_set[:, -1] - tree['right']), 2))
            tree_mean = (tree['left'] + tree['right']) / 2
            # merge가 no_merge보다 좋을 때는 leaf node를 합친다.
            if error_no_merge > tree_mean:
                print ('merging')
                return tree_mean
            # no_merge가 좋을 때는 leaf node를 합치지 않은채로 값을 return한다.
            else:
                return tree
        else:
            return tree


def data_loader(file_name):
    training_set = []
    file_reader = open(file_name)
    for line in file_reader.readlines():
        tokens = line.strip().split('\t')
        # python 2.7+ 에서는 아래와 같이 사용
        # float_tokens = list(map(float, tokens))
        float_tokens = map(float, tokens)
        training_set.append(float_tokens)
    return matrix(training_set)