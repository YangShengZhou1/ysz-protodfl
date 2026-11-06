import argparse
import logging
import os
import random
import sys
import pdb
import numpy as np
import torch
import copy
from torch import nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from fedml_api.model.cv.vgg import vgg11
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.tiny_imagenet.data_loader import load_partition_data_tiny
from fedml_api.model.cv.resnet import  customized_resnet18, tiny_resnet18, BaseHeadSplit
from fedml_api.model.cv.cnn_cifar10 import cnn_cifar10, cnn_cifar100
from fedml_api.standalone.ProToDFL.protoDFL_api import ProToDFLAPI
from fedml_api.standalone.ProToDFL.my_model_trainer import MyModelTrainer

def logger_config(log_path, logging_name):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100', 'resnet18', 'vgg11'")

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--momentum', type=float, default=0, metavar='N',
                        help='momentum')

    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data'),
                        help='root directory containing dataset-specific subdirectories')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'outputs'),
                        help='directory for run logs and topology figures')

    parser.add_argument('--partition_method', type=str, default='dir', metavar='N',
                        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
                             "one called 'n_cls' short for how many classes allocated for each client"
                             "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution")

    parser.add_argument('--partition_alpha', type=float, default=0.3, metavar='PA',
                        help='available parameters for data partition method')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='local batch size for training')
    
    parser.add_argument('--personal', type=int, default=0,
                        help='whether belongs to personalized FL')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay (default: 0.998)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='local training epochs for each client')

    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--frac', type=float, default=0.1, metavar='NN',
                        help='selection fraction each round')

    parser.add_argument('--comm_round', type=int, default=500,
                        help='how many round of communications we shoud use')
    
    parser.add_argument('--k', type=int, default=2,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')
    parser.add_argument("--cs", type = str, default='v0')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--dynamic', type=int, default=0,
                        help='static or dynamic')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    parser.add_argument('--step', type=int, default=0,
                        help='0-whole ProtoDFL, 1-FedAvg, 2-without 2-hop, 3-without wait')
    parser.add_argument('--oracle_mode', type=int, default=0,
                        help='0-off, 1-top-m global prototype oracle, 2-all clients oracle')
    parser.add_argument('--oracle_m', type=int, default=2,
                        help='number of collaborators for oracle_mode=1')
    parser.add_argument('--max_neighbors', type=int, default=0,
                        help='hard cap on collaborators for non-oracle runs; 0 disables')
    parser.add_argument('--aggregate_scope', type=str, default='all', choices=['all', 'base'],
                        help='aggregate all parameters or only the feature extractor (keep local head)')
    parser.add_argument('--self_weight_scale', type=float, default=1.0,
                        help='multiply the local sample count when weighting self in aggregation')
    parser.add_argument('--proto_update_each', type=int, default=0,
                        help='0-keep original behaviour (prototype computed at round 0 only), 1-refresh every round')
    parser.add_argument('--validation_ratio', type=float, default=0.0,
                        help='stratified per-client validation fraction held out from local training data')
    parser.add_argument('--save_models_round', type=str, default='-1',
                        help='comma-separated formal rounds to save client models/prototypes, e.g. 1,100,300,500; -1 disables')
    parser.add_argument('--save_models_dir', type=str, default='',
                        help='checkpoint root; defaults to output_dir/checkpoints')
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument('--t_max', type=int, default=0, help='max wait time t_max')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--momentums", type=float, default=0.1)
    parser.add_argument("--times", type=int, default=1)
    return parser


def load_data(args, dataset_name):
    dataset_subdirs = {"cifar10": "cifar10", "cifar100": "cifar100", "tiny": "tiny_imagenet"}
    if dataset_name not in dataset_subdirs:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    data_dir = os.path.join(args.data_dir, dataset_subdirs[dataset_name])
    if dataset_name == "cifar10":
        result = load_partition_data_cifar10(
            data_dir, args.partition_method, args.partition_alpha,
            args.client_num_in_total, args.batch_size, logger,
            validation_ratio=args.validation_ratio
        )
    elif dataset_name == "cifar100":
        result = load_partition_data_cifar100(
            data_dir, args.partition_method, args.partition_alpha,
            args.client_num_in_total, args.batch_size, logger,
            validation_ratio=args.validation_ratio
        )
    else:
        if args.validation_ratio > 0:
            raise ValueError("validation_ratio is currently implemented for CIFAR-10 only")
        result = load_partition_data_tiny(
            data_dir, args.partition_method, args.partition_alpha,
            args.client_num_in_total, args.batch_size, logger
        )
    base = list(result[:8])
    val_parts = list(result[8:10]) if len(result) == 10 else [None, None]
    return base + val_parts

def configure_model(model,model_name):#classifier
    # 拷贝原始的分类层并替换为 Identity，分离出特征提取部分
    if model_name == "vgg11":
        original_head = copy.deepcopy(model.classifier)
        model.classifier = nn.Identity()
    else:
        original_head = copy.deepcopy(model.linear)
        model.linear = nn.Identity()
    
    # 使用 BaseHeadSplit 重构模型
    configured_model = BaseHeadSplit(base=model, head=original_head)
    return configured_model

def create_model(args, model_name,class_num,logger):
    logger.info("create_model. model_name = %s" % (model_name))
    model = None
    if model_name == "lenet5":
        model = LeNet5(class_num)
        model = configure_model(model,model_name)
    elif model_name == "cnn_cifar10":
        model = cnn_cifar10()
        model = configure_model(model,model_name)
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
        model = configure_model(model,model_name)
    elif model_name =="resnet18" and args.dataset != 'tiny':
        model = customized_resnet18(class_num=class_num)
        model = configure_model(model,model_name)
    elif model_name == "resnet18" and args.dataset == 'tiny':
        model = tiny_resnet18(class_num=class_num)
        model = configure_model(model,model_name)
    elif model_name == "vgg11":
        model = vgg11(class_num)
        model = configure_model(model,model_name)
    return model


def custom_model_trainer(args, model, logger):
    return MyModelTrainer(model, args, logger)


if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description='ProToDFL-standalone'))
    args = parser.parse_args()
    # print("torch version{}".format(torch.__version__))

    data_partition = args.partition_method
    if data_partition != "homo":
        data_partition += str(args.partition_alpha)
    #args.identity = "FedAvg_baseline"  + "-"+data_partition
    args.identity = "ProToDFL"  + "-data-"+data_partition
    args.identity += "-ti" + "-" + str(args.times)
    args.identity += "-step" + "-" + str(args.step)
    args.identity += "-dy" + "-" + str(args.dynamic)#t_max
    args.identity += "-t_max" + "-" + str(args.t_max)
    args.identity += "-cs" + "-" + str(args.cs)
    args.identity += "-hops" + "-" + str(args.k)
    args.identity += '-mo' + str(args.momentums)
    args.identity += '-per' + "-"+ str(args.personal)
    args. client_num_per_round = int(args.client_num_in_total* args.frac)
    args.identity += "-mdl" + args.model
    args.identity += "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += '-seed' + str(args.seed)
    args.identity += '-tag' + str(args.tag)

    log_path = os.path.join(args.output_dir, 'logs', args.dataset, args.identity + '.log')
    logger = logger_config(log_path=log_path, logging_name=args.identity)

    logger.info(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
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
    for i in range(args.times):

        # create model.
        model = create_model(args, model_name=args.model, class_num=len(dataset[7][0]), logger=logger)
        # print(model)
        model_trainer = custom_model_trainer(args, model, logger)
        logger.info(model)

        fedAvgAPI = ProToDFLAPI(dataset, device, args, model_trainer, logger)
        fedAvgAPI.train()
