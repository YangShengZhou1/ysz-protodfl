import logging
import math
import os
import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from .datasets import CIFAR10_truncated

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        net_cls_counts.append (tmp)
    return net_cls_counts

def record_part(y_test, train_cls_counts,test_dataidxs, logger):
    test_cls_counts = []

    for net_i, dataidx in enumerate(test_dataidxs):
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        test_cls_counts.append ( tmp)
        logger.debug('DATA Partition: Train %s; Test %s' % (str(train_cls_counts[net_i]), str(tmp) ))
    return


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()
    download = not os.path.isdir(os.path.join(datadir, "cifar-10-batches-py"))
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=download, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=download, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def _allocate_by_priors(y_train, n_client, cls_priors, logger, method_name):
    """Allocate every training sample exactly once using client class priors.

    The implementation is finite by construction: each draw removes one index
    from a class pool, and classes whose pool is empty are excluded. It also
    distributes the integer-rounding remainder so that all samples are used.
    """
    total = len(y_train)
    n_cls = cls_priors.shape[1]
    raw_counts = np.random.lognormal(mean=np.log(total / n_client), sigma=0, size=n_client)
    counts = np.floor(raw_counts / raw_counts.sum() * total).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        for client in np.random.permutation(n_client)[:remainder]:
            counts[client] += 1
    if counts.sum() != total:
        raise RuntimeError(f"{method_name}: client counts do not cover all samples")

    idx_list = [np.where(y_train == c)[0] for c in range(n_cls)]
    for indices in idx_list:
        np.random.shuffle(indices)
    cursors = [0] * n_cls
    net_dataidx_map = {client: [] for client in range(n_client)}

    for client in range(n_client):
        for _ in range(int(counts[client])):
            available = [c for c in range(n_cls) if cursors[c] < len(idx_list[c])]
            if not available:
                raise RuntimeError(f"{method_name}: no class samples left before allocation completed")
            probabilities = np.asarray([cls_priors[client, c] for c in available], dtype=float)
            if probabilities.sum() <= 0:
                probabilities = np.ones(len(available), dtype=float)
            probabilities /= probabilities.sum()
            cls_label = int(np.random.choice(available, p=probabilities))
            net_dataidx_map[client].append(int(idx_list[cls_label][cursors[cls_label]]))
            cursors[cls_label] += 1

    used = sum(len(indices) for indices in net_dataidx_map.values())
    if used != total or len(set().union(*(set(v) for v in net_dataidx_map.values()))) != total:
        raise RuntimeError(f"{method_name}: allocation is incomplete or contains duplicates")
    logger.info(f"{method_name}: allocated {used}/{total} samples across {n_client} clients")
    return net_dataidx_map


def partition_data( datadir, partition, n_nets, alpha, logger):
    logger.info("*********partition data***************")
    net_dataidx_map = {}
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]

    if partition == 'n_cls':
        n_shards = 10
        n_client = n_nets
        n_cls = 10
        # 每个客户端的类别数目
        n_classes_per_group = int(alpha)
        if n_classes_per_group == 22:
            n_classes_per_group = 2

        if n_client < n_shards or n_client % n_shards != 0:
            raise ValueError("n_cls requires client count >= 10 and divisible by 10")
        cls_priors = np.zeros(shape=(n_client, n_cls))
        for i in range(n_shards):
            selected_classes = random.sample(range(n_cls), n_classes_per_group)
            shard_clients = range(i * (n_client // n_shards), (i + 1) * (n_client // n_shards))
            for client in shard_clients:
                cls_priors[client][selected_classes] = 1.0 / n_classes_per_group

        net_dataidx_map = _allocate_by_priors(y_train, n_client, cls_priors, logger, "n_cls")

        '''
        for i in range(n_client):
            cls_priors[i][random.sample(range(n_cls), int(alpha))] = 1.0 / alpha

        prior_cumsum = np.cumsum(cls_priors, axis=1)
        # 生成分片的类别先验概率
        #cls_priors_tmp = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_shards)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = np.random.randint(0, len(idx_list[cls_label]))
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])

                break
        '''
        
            
    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logger.info("N = " + str(N))
        net_dataidx_map = {}
        attempts = 0

        while min_size < 10:
            attempts += 1
            if attempts > 100:
                raise RuntimeError("hetero partition failed to produce clients with >=10 samples")
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == 'dir':
        n_client = n_nets
        n_cls = 10

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_client)
        net_dataidx_map = _allocate_by_priors(y_train, n_client, cls_priors, logger, "dir")

    elif partition == 'my_part':
        n_shards = 10
        n_client = n_nets
        n_cls = 10
        if n_client < n_shards or n_client % n_shards != 0:
            raise ValueError("my_part requires client count >= 10 and divisible by 10")

        cls_priors = np.zeros(shape=(n_client, n_cls))
        cls_priors_tmp = np.random.dirichlet(alpha=[alpha] * n_cls, size=int(n_shards))
        clients_per_shard = n_client // n_shards
        for i in range(n_client):
            cls_priors[i] = cls_priors_tmp[i // clients_per_shard]

        net_dataidx_map = _allocate_by_priors(y_train, n_client, cls_priors, logger, "my_part")

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader_cifar10(datadir, train_bs, test_bs, dataidxs=None,test_idxs=None, cache_train_data_set=None,cache_test_data_set=None,logger=None):
    transform_train, transform_test = _data_transforms_cifar10()
    dataidxs=np.array(dataidxs)
    logger.info("train_num{}  test_num{}".format(len(dataidxs),len(test_idxs)))
    download = not os.path.isdir(os.path.join(datadir, "cifar-10-batches-py"))
    train_ds = CIFAR10_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=download,cache_data_set=cache_train_data_set)
    test_ds = CIFAR10_truncated(datadir, dataidxs=test_idxs, train=False, transform=transform_test, download=download,
                      cache_data_set=cache_test_data_set)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False)
    return train_dl, test_dl

def _split_train_validation(indices, y_train, validation_ratio):
    """Stratified per-client split held out before training."""
    indices = np.asarray(indices, dtype=np.int64)
    if validation_ratio <= 0 or len(indices) < 2:
        return indices, np.asarray([], dtype=np.int64)
    train_parts, val_parts = [], []
    for label in np.unique(y_train[indices]):
        cls_indices = indices[y_train[indices] == label]
        cls_indices = np.random.permutation(cls_indices)
        n_val = int(round(len(cls_indices) * validation_ratio))
        if n_val == 0 and len(cls_indices) > 1:
            n_val = 1
        n_val = min(n_val, len(cls_indices) - 1)
        val_parts.append(cls_indices[:n_val])
        train_parts.append(cls_indices[n_val:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def load_partition_data_cifar10( data_dir, partition_method, partition_alpha, client_number, batch_size, logger, validation_ratio=0.0):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha, logger)
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    val_data_local_num_dict = dict()
    val_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_cifar10()
    download = not os.path.isdir(os.path.join(data_dir, "cifar-10-batches-py"))
    cache_train_data_set=CIFAR10(data_dir, train=True, transform=transform_train, download=download)
    cache_test_data_set = CIFAR10(data_dir, train=False, transform=transform_test, download=download)
    idx_test = [[] for i in range(10)]
    # checking
    for label in range(10):
        idx_test[label] = np.where(y_test == label)[0]
    test_dataidxs = [[] for i in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    for client_idx in range(client_number):
        for label in range(10):
            # each has 100 pieces of testing data
            label_num = math.ceil(traindata_cls_counts[client_idx][label] / sum(traindata_cls_counts[client_idx]) * tmp_tst_num)
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]]))
        dataidxs, validx = _split_train_validation(
            net_dataidx_map[client_idx], y_train, validation_ratio
        )
        train_data_local, test_data_local = get_dataloader_cifar10( data_dir, batch_size, batch_size,
                                                 dataidxs,test_dataidxs[client_idx] ,cache_train_data_set=cache_train_data_set,cache_test_data_set=cache_test_data_set,logger=logger)
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logger.info("client_idx = %d, local_sample_number = %d, validation_number = %d" % (
            client_idx, local_data_num, len(validx)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        if validation_ratio > 0:
            val_ds = CIFAR10_truncated(
                data_dir, dataidxs=validx, train=True, transform=transform_test,
                download=download, cache_data_set=cache_train_data_set
            )
            val_data_local_dict[client_idx] = data.DataLoader(
                dataset=val_ds, batch_size=batch_size, shuffle=False, drop_last=False
            )
            val_data_local_num_dict[client_idx] = len(val_ds)

    record_part(y_test, traindata_cls_counts, test_dataidxs, logger)

    if validation_ratio > 0:
        return None, None, None, None, \
               data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts, \
               val_data_local_num_dict, val_data_local_dict
    return None, None, None, None, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts
