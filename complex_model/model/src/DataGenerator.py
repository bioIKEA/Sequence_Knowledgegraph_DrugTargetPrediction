import numpy as np

import random


def readIdx(idx_file=''):
    with open(idx_file) as f:
        e_idx = []

        for line in f:
            e_idx.append(line.strip())

        return e_idx


def readTrain(train_file='', drug_idx_file='', target_idx_file=''):
    print('read idx: ', drug_idx_file)

    global_drugs = readIdx(drug_idx_file)

    global_targets = readIdx(target_idx_file)

    # global_disease = readIdx()

    # global_sideEffect = readIdx()

    existing_triples_set = set()

    train_negative_triples_set = set()

    train_positive_triples_set = set()

    drug_set = set()

    target_set = set()

    print('read train: ', train_file)

    with open(train_file) as f:

        for line in f:
            items = line.split()

            train_positive_triples_set.add((items[0], items[1]))

            existing_triples_set.add((items[0], items[1]))

            drug_set.add(items[0])

            target_set.add(items[1])

    local_drugs = []

    local_targets = []

    for drug in drug_set:
        local_drugs.append(drug)

    for target in target_set:
        local_targets.append(target)

    while len(train_negative_triples_set) < len(train_positive_triples_set):

        # print (len(train_negative_triples_set), ' < ', len(train_positive_triples_set) )

        drug_idx = random.randint(0, len(drug_set) - 1)

        target_idx = random.randint(0, len(target_set) - 1)

        _drug = local_drugs[drug_idx]

        _target = local_targets[target_idx]

        if (_drug, _target) not in existing_triples_set:

            if (_drug in global_drugs and _target in global_targets):
                train_negative_triples_set.add((_drug, _target))

                existing_triples_set.add((_drug, _target))

    data_set = np.zeros((len(train_positive_triples_set) + len(train_negative_triples_set), 3), dtype=int)

    print('data_set: ', data_set.shape)

    count = 0

    for pair in train_positive_triples_set:

        if (pair[0] in global_drugs and pair[1] in global_targets):
            d_idx = global_drugs.index(pair[0])

            t_idx = global_targets.index(pair[1])

            data_set[count][0] = d_idx

            data_set[count][1] = t_idx

            data_set[count][2] = 1

            count += 1

    for pair in train_negative_triples_set:

        if (pair[0] in global_drugs and pair[1] in global_targets):
            d_idx = global_drugs.index(pair[0])

            t_idx = global_targets.index(pair[1])

            data_set[count][0] = d_idx

            data_set[count][1] = t_idx

            data_set[count][2] = 0

            count += 1

    print('count: ', count)

    return data_set


def readTest(test_file='', drug_idx_file='', target_idx_file=''):
    global_drugs = readIdx(drug_idx_file)

    global_targets = readIdx(target_idx_file)

    # global_disease = readIdx()

    # global_sideEffect = readIdx()

    test_negative_triples_set = set()

    test_positive_triples_set = set()

    with open(test_file) as f:

        for line in f:

            items = line.split()

            if (items[2] == 'true'):

                test_positive_triples_set.add((items[0], items[1]))

            elif (items[2] == 'false'):

                test_negative_triples_set.add((items[0], items[1]))

    data_set = np.zeros((len(test_positive_triples_set) + len(test_negative_triples_set), 3), dtype=int)

    count = 0

    for pair in test_positive_triples_set:

        if (pair[0] in global_drugs and pair[1] in global_targets):
            d_idx = global_drugs.index(pair[0])

            t_idx = global_targets.index(pair[1])

            data_set[count][0] = d_idx

            data_set[count][1] = t_idx

            data_set[count][2] = 1

            count += 1

    for pair in test_negative_triples_set:

        if (pair[0] in global_drugs and pair[1] in global_targets):
            d_idx = global_drugs.index(pair[0])

            t_idx = global_targets.index(pair[1])

            data_set[count][0] = d_idx

            data_set[count][1] = t_idx

            data_set[count][2] = 0

            count += 1

    return data_set