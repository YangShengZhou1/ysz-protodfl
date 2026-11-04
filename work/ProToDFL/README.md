# DisPFL

然后接口代码写在fedml_api文件里面，有关于各个联邦去中心化方法，fedml_experiments里面就是跑这些联邦学习方法
路径是DisPFL-master\fedml_api\standalone

\DisPFL-master\fedml_api\data_preprocessing这个路径控制各个数据异构

\DisPFL-master\fedml_api\utils里面改一些拓扑结构


This repository contains the official implementation for the manuscript:

> [DisPFL: Towards Communication-Efficient Personalized Federated learning via Decentralized Sparse Training](https://arxiv.org/pdf/2206.00187.pdf)

Personalized federated learning is proposed to handle the data heterogeneity problem amongst clients by learning dedicated tailored local models for each user. However, existing works are often built in a centralized way, leading to high communication pressure and high vulnerability when a failure or an attack on the central server occurs. In this work, we propose a novel personalized federated learning framework in a decentralized (peer-to-peer) communication protocol named Dis-PFL, which employs personalized sparse masks to customize sparse local models on the edge. To further save the communication and computation cost, we propose a decentralized sparse training technique, which means that each local model in Dis-PFL only maintains a fixed number of active parameters throughout the whole local training and peer-to-peer communication process. Comprehensive experiments demonstrate that Dis-PFL significantly saves the communication bottleneck for the busiest node among all clients and, at the same time, achieves higher model accuracy with less computation cost and communication rounds. Furthermore, we demonstrate that our method can easily adapt to heterogeneous local clients with varying computation complexities and achieves better personalized performances.

# Experiments

The implementations of each method are provided in the folder `/fedml_api/standalone`, while experiments are provided in the folder `/fedml_experiments/standalone`.

Use dataset corresponding bash file to run the experiments.

```
cd /fedml_experiments/standalone/DisPFL
sh cifar10.sh
```

# Citation

If you find this repo useful for your research, please consider citing the paper

```
@InProceedings{pmlr-v162-dai22b,
  title = 	 {{D}is{PFL}: Towards Communication-Efficient Personalized Federated Learning via Decentralized Sparse Training},
  author =       {Dai, Rong and Shen, Li and He, Fengxiang and Tian, Xinmei and Tao, Dacheng},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {4587--4604},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/dai22b/dai22b.pdf},
  url = 	 {https://proceedings.mlr.press/v162/dai22b.html},
  abstract = 	 {Personalized federated learning is proposed to handle the data heterogeneity problem amongst clients by learning dedicated tailored local models for each user. However, existing works are often built in a centralized way, leading to high communication pressure and high vulnerability when a failure or an attack on the central server occurs. In this work, we propose a novel personalized federated learning framework in a decentralized (peer-to-peer) communication protocol named DisPFL, which employs personalized sparse masks to customize sparse local models on the edge. To further save the communication and computation cost, we propose a decentralized sparse training technique, which means that each local model in DisPFL only maintains a fixed number of active parameters throughout the whole local training and peer-to-peer communication process. Comprehensive experiments demonstrate that DisPFL significantly saves the communication bottleneck for the busiest node among all clients and, at the same time, achieves higher model accuracy with less computation cost and communication rounds. Furthermore, we demonstrate that our method can easily adapt to heterogeneous local clients with varying computation complexities and achieves better personalized performances.}
}
```

[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
