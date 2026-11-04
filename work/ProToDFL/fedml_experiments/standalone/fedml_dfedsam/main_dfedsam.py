import argparse
import logging
import os
import random
import sys
import pdb
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.abspath("/media/sysu/computer2/LXL/DisPFL-master/"))#/media/sysu/computer2/LXL/DisPFL-master/fedml_experiments/standalone/fedml_dfedsam/
from fedml_api.model.cv.vgg import vgg11
#from fedml_api.model.cv.AlexNet import AlexNetCIFAR, AlexNet
#from fedml_api.model.cv.resnet_ham import wide_resnet50_2
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import (
    load_partition_data_cifar100,
)
from fedml_api.data_preprocessing.tiny_imagenet.data_loader import (
    load_partition_data_tiny,
)
from fedml_api.model.cv.cnn_cifar10 import cnn_cifar10, cnn_cifar100
from fedml_api.model.cv.lenet5 import LeNet5
from fedml_api.standalone.dfedsam.dfedsam_api import DFedSAMAPI
from fedml_api.standalone.dfedsam.dfedsam_muti_api import DFedSAMMutiAPI
from fedml_api.standalone.dfedsam.dfedsam_model_trainer import DFedSAMMModelTrainer
#from fedml_api.data_preprocessing.HAM10000.data_loader import (
#    load_partition_data_HAM10000,
#)
from fedml_api.model.cv.resnet import (
    customized_resnet18,
    tiny_resnet18#,
#    customized_resnet50,
)
from fedml_api.utils.common_args import add_common_args
import fedml_api.utils.config as config_loader
#from fedml_api.model.cv.wideresnet import wideresnet28_4


def merge_datasets(dataset1, dataset2, client_num):
    train_data_num1, test_data_num1, train_data_global1, test_data_global1, \
    train_data_local_num_dict1, train_data_local_dict1, test_data_local_dict1, \
    class_num1 = dataset1

    train_data_num2, test_data_num2, train_data_global2, test_data_global2, \
    train_data_local_num_dict2, train_data_local_dict2, test_data_local_dict2, \
    class_num2 = dataset2

    train_data_num = None
    test_data_num = None
    train_data_global = None
    test_data_global = None
    class_num = class_num1 + class_num2

    train_data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    half_clients = client_num // 2

    for i in range(half_clients):
        train_data_local_num_dict[i] = train_data_local_num_dict1[i]
        train_data_local_dict[i] = train_data_local_dict1[i]
        test_data_local_dict[i] = test_data_local_dict1[i]

    for i in range(half_clients, client_num):
        train_data_local_num_dict[i] = train_data_local_num_dict2[i - half_clients]
        train_data_local_dict[i] = train_data_local_dict2[i - half_clients]
        test_data_local_dict[i] = test_data_local_dict2[i - half_clients]

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num



def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode="w", encoding="UTF-8")
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser = add_common_args(parser)
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        metavar="N",
        help="network architecture, supporting 'alexnet', 'resnet18', 'vgg11'",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="N",
        help="momentum used for DFedAvgM and DfedSAM",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/sysu/computer2/LXL/DisPFL-master/data/",
        help="data directory, please feel free to change the directory to the right place",
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="dir",
        metavar="N",
        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
        "one called 'n_cls' short for how many classes allocated for each client"
        "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=0.3,
        metavar="PA",
        help="available parameters for data partition method",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="local batch size for training",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )

    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.998,
        metavar="LR_decay",
        help="learning rate decay (default: 0.998)",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=5e-4
    )
    parser.add_argument(
        "--rho",
        help="control the variablity/smoothness of gradients in SAM;",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="local training epochs for each client",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=100,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--frac",
        type=float,
        default=0.1,
        metavar="NN",
        help="selection fraction each round",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=500,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=1,
        help="the frequency of the algorithms",
    )

    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--active", type=float, default=1.0)
    parser.add_argument("--cs", type=str, default="random", help="topology, include ring, random, cyclic")

    parser.add_argument("--ci", type=int, default=0, help="CI")
    parser.add_argument("--tag", type=str, default="test")

    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
    else:
        if dataset_name == "cifar100":
            args.data_dir += "cifar100"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_cifar100(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)
        elif dataset_name == "tiny":
            args.data_dir += "tiny_imagenet"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_tiny(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)

        elif dataset_name == "cifar110":
            args.data_dir_cifar10 = args.data_dir + "cifar10"
            args.data_dir_cifar100 = args.data_dir + "cifar100"
            client_num_half = args.client_num_in_total // 2

            dataset_cifar10 = load_partition_data_cifar10(args.data_dir_cifar10, args.partition_method,
                                                        args.partition_alpha, client_num_half, args.batch_size, logger)
            dataset_cifar100 = load_partition_data_cifar100(args.data_dir_cifar100, args.partition_method,
                                                            args.partition_alpha, client_num_half, args.batch_size, logger)

            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = merge_datasets(dataset_cifar10, dataset_cifar100, args.client_num_in_total)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args, model_name, class_num):
    model = None
    if model_name == "cnn_cifar10":
        model = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
    elif model_name == "resnet18" and args.dataset != "tiny":
        model = customized_resnet18(class_num=class_num)
    elif model_name == "resnet50" and args.dataset != "tiny":
        model = customized_resnet50(class_num=class_num)
    elif model_name == "resnet18" and args.dataset == "tiny":
        model = tiny_resnet18(class_num=class_num)
    elif model_name == "vgg11":
        model = vgg11(class_num)
    elif model_name == "alex":
        model = AlexNet(class_num)
    elif model_name == "alex_cifar10":
        model = AlexNetCIFAR()
    elif model_name == "wide_resnet28_4":
        model = wideresnet28_4(class_num)
    else:
        raise Exception("Error: no model " + model_name)
    return model


def create_combined_model(model_name, class_num_cifar10, class_num_cifar100):
    model_cifar10 = None
    model_cifar100 = None

    if model_name == "cnn_cifar10":
        model_cifar10 = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model_cifar100 = cnn_cifar100()
    elif model_name == "resnet18":
        model_cifar10 = customized_resnet18(class_num=class_num_cifar10)
        model_cifar100 = customized_resnet18(class_num=class_num_cifar100)
    elif model_name == "vgg11":
        model_cifar10 = vgg11(class_num_cifar10)
        model_cifar100 = vgg11(class_num_cifar100)
    return model_cifar10, model_cifar100

def custom_model_trainer(args, model, logger):
    if isinstance(model, tuple):
        model_cifar10, model_cifar100 = model
        trainer_cifar10 = DFedSAMMModelTrainer(model_cifar10, args, logger)
        trainer_cifar100 = DFedSAMMModelTrainer(model_cifar100, args, logger)
        return trainer_cifar10, trainer_cifar100
    else:
        return DFedSAMMModelTrainer(model, args, logger)


# def custom_model_trainer(args, model, logger):
#     return DFedSAMMModelTrainer(model, args, logger)


if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description="DFedAvgM-standalone"))
    args = parser.parse_args()
    # print("torch version{}".format(torch.__version__))

    config_loader.Config.initialize(args)

    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
    data_partition = args.partition_method
    if data_partition != "homo":
        data_partition += str(args.partition_alpha)
    args.identity = "dfedsam" + "-" + data_partition
    args.client_num_per_round = int(args.client_num_in_total * args.frac)
    args.identity += "-mdl" + args.model
    args.identity += (
        "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    )
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += "-rho" + str(args.rho)
    args.identity += "-seed" + str(args.seed)
    args.identity += '-cs'+args.cs

    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    log_path = os.path.join(
        cur_dir,
        "dfedsam/"
        + "LOG/"
        + args.dataset
        + "/"
        + current_datetime_str
        + args.identity
        + ".log",
    )
    logger = logger_config(log_path=log_path, logging_name=args.identity)
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu))
    logger.info(device)
    logger.info("running at device{}".format(device))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset)

    if args.dataset == "cifar110":
            model_cifar10, model_cifar100 = create_combined_model( model_name=args.model, class_num_cifar10=10, class_num_cifar100=100)
            model_trainer_cifar10, model_trainer_cifar100 = custom_model_trainer(args, (model_cifar10, model_cifar100), logger)
            logger.info("CIFAR-10 model: {}".format(model_cifar10))
            logger.info("CIFAR-100 model: {}".format(model_cifar100)) 

            dfedsamAPI = DFedSAMMutiAPI(dataset, device, args, (model_trainer_cifar10, model_trainer_cifar100), logger) 
            dfedsamAPI.train()
    else:
            # create model.
            model = create_model(args, model_name=args.model, class_num=len(dataset[-1][0]))
            # print(model)
            model_trainer = custom_model_trainer(args, model, logger)
            logger.info(model)

            dfedsamAPI = DFedSAMAPI(dataset, device, args, model_trainer, logger)
            dfedsamAPI.train()
