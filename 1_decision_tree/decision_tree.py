"""
    decision tree
"""

import copy
from collections import Counter
from math import log


def get_formatted_data(fname):
    """
        get raw data and convert to proper format for the algorithm
    """
    with open(fname, 'r') as f:
        raw_text = f.readlines()
        # get the header.  assume in format like "(Occupied, Price, Music, Location, VIP, Favorite Beer, Enjoy)"
        header = (raw_text[0].strip())[1:-1].replace(' ', '').split(',')
        data_set = []

        for data in raw_text[2:]:  # skip header and blank second line
            # clean text and get list of attributes
            raw_stats_list = data.replace(' ', '').split(':')[-1][:-2].split(',')
            data_set.append(raw_stats_list)
        # for data in data_set:
        #     print(data)
        return header, data_set


def calc_entropy(data):
    """ return the base entropy of the given data set"""
    # get the total number of instances
    inst_count = len(data)
    # get the occurrences of the different results, could also use Collection.Counter
    res_dict = dict()
    for ins in data:
        result = ins[-1]
        if result not in res_dict.keys():
            res_dict[result] = 0
        res_dict[result] += 1
    entropy = 0.0
    # entropy = - sum(p*log(p)) from all results
    for res in res_dict:
        res_prob = float(res_dict[res] / inst_count)
        entropy -= res_prob * log(res_prob, 2)

    # print('entropy:{}'.format(entropy))
    return entropy


def calc_info_gain(data_set, header):
    """ return the info gain of given index """
    para_count = len(header) - 1
    base_entropy = calc_entropy(data_set)

    most_info_gain = 0.0

    # its okay even best feature index never get updated (when info_gain == 0).
    # since it will return 'Tie' immediately afterward
    best_feature_index = -1

    for i in range(para_count):
        # get list of unique parameters
        cur_para_list = [ins[i] for ins in data_set]
        unique_para_set = set(cur_para_list)
        # para_counter = Counter(cur_para_list)
        cur_entropy = 0.0

        for val in unique_para_set:
            sub_data_set = split_data_set(data_set, i, val)

            prob = len(sub_data_set) / float(len(data_set))
            cur_entropy += prob * calc_entropy(sub_data_set)
        info_gain = base_entropy - cur_entropy

        # tie breaker for attributes, select the one closer to the front
        # if info_gain > most_info_gain or info_gain == most_info_gain and i > best_feature_index:
        if info_gain > most_info_gain or info_gain == most_info_gain and i < best_feature_index:
            most_info_gain = info_gain
            best_feature_index = i

    return best_feature_index, most_info_gain


def split_data_set(data_set, index, value):
    """
        return the tree contains the parts we need only,
        index for locating the attributes, like index = 2, header[index] = 'Price'
        value for selecting which outcome to follow, like value='expensive' (one of the outcomes for Price)
    """
    res_data_list = []
    for ins in data_set:
        if ins[index] == value:
            temp_list = copy.deepcopy(ins)
            temp_list.pop(index)
            res_data_list.append(temp_list)
    return res_data_list


def create_tree(data_set, labels):
    res_list = [data[-1] for data in data_set]

    # check end game scenario
    # if the leaf is pure return the result
    if res_list.count(res_list[0]) == len(res_list):
        # print('pure leaf with res_list[0] = {}'.format(res_list[0]))
        return res_list[0]  # stop splitting when all of the classes are equal/result set is pure

    # return most common result if not enough data/ SHOULD THIS RETURN TIE???
    if len(data_set[0]) == 1:
        print(data_set)
        print('when len(data_set(0)) == 1, return {}'.format(Counter(res_list).most_common(1)[0][0]))
        # return Counter(res_list).most_common(1)[0][0]
        return 'Tie'

    # # calc info gain to determine which attribute to split
    del_index, info_gain = calc_info_gain(data_set, labels)

    # return tie if no info gain (two same situation gives different answer)
    if info_gain == 0:
        # print('zero info gain.\ndata:{}\nheader:{}\n'.format(data_set, labels))
        return 'Tie'

    # add highest info gain into result set,
    selected_attr = labels[del_index]

    # keep building new tree/branch
    my_tree = {selected_attr: {}}

    # get unique values of selected attr in data set
    unique_attr_val_set = set([ins[del_index] for ins in data_set])

    # delete select data for such info gain, prep for new trees
    labels.pop(del_index)

    for val in unique_attr_val_set:
        new_labels = copy.deepcopy(labels)  # copy a new instance of the lables
        new_data_set = split_data_set(data_set, del_index, val)  # delete the branches that unnecessary
        my_tree[selected_attr][val] = create_tree(new_data_set, new_labels)

    return my_tree


def predict(pred_header, pred_values, my_tree):
    """
        given a decision tree, predict a given dataset's outcome
    """
    # find the best feature to split
    split_feature = list(my_tree.keys())[0]
    # print('best feature to split: {}'.format(split_feature))

    feature_index = pred_header.index(split_feature)
    feature_val = pred_values[feature_index].replace(' ', '')

    # deal with possible placeholder Ties.
    try:
        cur_res = my_tree[split_feature][feature_val]
        if cur_res == 'No' or cur_res == 'Yes' or cur_res == 'Tie':
            # print(cur_res)
            return cur_res
    except KeyError:
        # print('no attributes in tree')
        return 'Tie'  # kinda cheesy, but works :P

    except Exception as e:
        print('exception :{}'.format(e))  # should not happen

    # delete the feature from pred_header
    # only keep the trees with that values and call recursively

    new_tree = my_tree[split_feature][feature_val]
    # print('decision tree:{}\n'.format(new_tree))

    pred_header.remove(split_feature)
    pred_values.pop(feature_index)

    final_res = predict(pred_header, pred_values, new_tree)

    return final_res


def tree_plotter_linear(dt, depth=-1, res_list=[[] for _ in range(7)]):
    """ save the decision tree in desired format , trace by depth, for print afterward
    :param dt: a nested dictionary, decision tree
    :param res_list: list to track what to print, init for 7 levels (max depth possible in this case)
    :param depth: use to track which level to print the variables.
    """
    depth += 1

    # leaf nodes reached, add to list
    if isinstance(dt, str):
        res_list[depth].append(dt)
    else:
        for val in list(dt.keys()):
            res_list[depth].append(val)

            for branch in list(dt[val].keys()):
                tree_plotter_linear(dt[val][branch], depth, res_list)

    return res_list


# get the data from dt-data.txt
fname = 'dt-data.txt'
header_, data_set_, = get_formatted_data(fname)

# create decision tree from the data
decision_tree = create_tree(data_set_, header_)
print('Print Generated Decision Tree in dictionary mode:\n{}\n'.format(decision_tree))

# making prediction
pred_header = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer']
pred_values = ['Moderate', 'Cheap', 'Loud', ' City-Center', ' No', 'No']
result = predict(pred_header, pred_values[:], decision_tree)
print('Prediction mode\nFor params:{}\nResult :{}\n'.format(pred_values, result))

# print tree in plain text
results = tree_plotter_linear(decision_tree)
print('\nDecision tree in print mode:')
for result in results:
    if result:  # not print empty lines
        print(','.join(result))

# generate and show graph of the tree
print('\n\nGraph output: (note that matplotlib part of code was from third party)')
import tree_plotter

tree_plotter.createPlot(decision_tree)
