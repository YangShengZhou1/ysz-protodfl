import argparse
import logging
import os
import random
import sys
import pdb
import numpy as np
import torch
import tqdm
import math
#sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath("/media/sysu/computer2/LXL/DisPFL-master"))
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.model.cv.vgg import vgg16, vgg11
from fedml_api.model.cv.cnn_cifar10 import cnn_cifar10, cnn_cifar100
from fedml_api.standalone.DisPFL.dispfl_api_muti import dispflAPI
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.tiny_imagenet.data_loader import load_partition_data_tiny
from fedml_api.model.cv.resnet import  customized_resnet18, tiny_resnet18
from fedml_api.standalone.DisPFL.my_model_trainer import MyModelTrainer

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100', 'resnet18', 'vgg11'")

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='/media/sysu/computer2/LXL/DisPFL-master/data/',
                        help='data directory, please feel free to change the directory to the right place')

    parser.add_argument('--partition_method', type=str, default='dir', metavar='N',
                        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
                             "one called 'n_cls' short for how many classes allocated for each client"
                             "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution")

    parser.add_argument('--partition_alpha', type=float, default=0.3, metavar='PA',
                        help='available parameters for data partition method')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='local batch size for training')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')

    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay')

    parser.add_argument('--wd', help='weight decay parameter', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='local training epochs for each client')

    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--frac', type=float, default=0.1, metavar='NN',
                        help='available communication fraction each round')

    parser.add_argument('--momentum', type=float, default=0, metavar='NN',
                        help='momentum')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='total communication rounds')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the test algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--dense_ratio', type=float, default=0.5,
                        help='local density ratio')

    parser.add_argument('--anneal_factor', type=float, default=0.5,
                        help='anneal factor for pruning')
    #parser.add_argument('--device', type=str, default='cpu', help='设备使用: "cuda" 或 "cpu"')

    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--cs", type = str, default='v0')
    parser.add_argument("--active", type=float,default=1.0)

    parser.add_argument("--public_portion", type=float, default=0)
    parser.add_argument("--erk_power_scale", type=float, default=1 )
    parser.add_argument("--dis_gradient_check", action='store_true')
    parser.add_argument("--strict_avg", action='store_true')
    parser.add_argument("--static", action='store_true')
    parser.add_argument("--uniform", action='store_true')
    parser.add_argument("--save_masks", action='store_true')
    parser.add_argument("--different_initial", action='store_true')
    parser.add_argument("--record_mask_diff", action='store_true')
    parser.add_argument("--diff_spa", action='store_true')
    parser.add_argument("--global_test", action='store_true')
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--times", type=int, default=2)
    return parser


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


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
    elif dataset_name == "cifar100":
        args.data_dir += "cifar100"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar100(args.data_dir, args.partition_method,
                                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
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
        trainer_cifar10 = MyModelTrainer(model_cifar10, args, logger)
        trainer_cifar100 = MyModelTrainer(model_cifar100, args, logger)
        return trainer_cifar10, trainer_cifar100
    else:
        return MyModelTrainer(model, args, logger)



        
def create_model(args, model_name,class_num):
    model = None
    if model_name == "cnn_cifar10":
        model = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
    elif model_name == "resnet18" and args.dataset != 'tiny':
        model = customized_resnet18(class_num=class_num)
    elif model_name == "resnet18" and args.dataset == 'tiny':
        model = tiny_resnet18(class_num=class_num)
    elif model_name == "vgg11":
        model = vgg11(class_num)
    return model




def custom_model_trainer(args, model, logger):
    if isinstance(model, tuple):
        model_cifar10, model_cifar100 = model
        trainer_cifar10 = MyModelTrainer(model_cifar10, args, logger)
        trainer_cifar100 = MyModelTrainer(model_cifar100, args, logger)
        return trainer_cifar10, trainer_cifar100
    else:
        return MyModelTrainer(model, args, logger)

def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description='DisPFL-standalone'))
    args = parser.parse_args()

    print("torch version{}".format(torch.__version__))
    # 根据用户输入选择设备
    data_partition=args.partition_method
    if data_partition!="homo":
        data_partition+=str(args.partition_alpha)
    args.identity = "DisPFL_muti" + "-" + args.dataset + "-" + data_partition
    args.identity += "-ti" + "-" + str(args.times)
    args.identity+="-mdl" + args.model
    args.identity+="-cs"+args.cs

    if args.save_masks:
        args.identity+="-masks"

    if args.diff_spa:
        args.identity += "-diff_spa"

    if args.uniform:
        args.identity += "-uniform_init"
    else:
        args.identity += "-ERK_init"

    if args. different_initial:
        args.identity += "-diff_init"
    else:
        args.identity += "-same_init"

    if args. global_test:
        args.identity += "-g"

    if args.static:
        args.identity += '-RSM'
    else:
        args.identity += '-DST'

    args.identity += "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    args. client_num_per_round = int(args.client_num_in_total* args.frac)
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += "-dr" + str(args.dense_ratio)
    args.identity += "-active" + str(args.active)
    args.identity += '-seed' + str(args.seed)

    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    log_path = '/media/sysu/computer2/LXL/DisPFL-master/fedml_experiments/standalone/DisPFL/LOG/' + args.dataset + '/' + args.identity + '.log'
    logger = logger_config(log_path=log_path, logging_name=args.identity)

    device = torch.device("cuda:" + str(args.gpu) )
    logger.info(args)
    logger.info(device)

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
    
    for i in range(args.times+1):
        # create model.
        print(i)
        if args.dataset == "cifar110":
            model_cifar10, model_cifar100 = create_combined_model( model_name=args.model, class_num_cifar10=10, class_num_cifar100=100)
            model_trainer_cifar10, model_trainer_cifar100 = custom_model_trainer(args, (model_cifar10, model_cifar100), logger)
            logger.info("CIFAR-10 model: {}".format(model_cifar10))
            logger.info("CIFAR-100 model: {}".format(model_cifar100))    
            dispflAPis = dispflAPI(dataset, device, args, (model_trainer_cifar10, model_trainer_cifar100), logger)
            if i != 0 and i !=1:
                dispflAPis.train()
        else:
            model = create_model(args, model_name=args.model, class_num=len(dataset[-1][0]))
            model_trainer = custom_model_trainer(args, model, logger)
            logger.info(model)
            dispflAPis = dispflAPI(dataset, device, args, model_trainer, logger)
            if i != 0 and i !=1:
                dispflAPis.train()
