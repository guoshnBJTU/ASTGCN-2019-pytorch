# ASTGCN

Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (ASTGCN)

<img src="fig/ASTGCN architecture.png" alt="image-20200103164326338" style="zoom:50%;" />

This is a Pytorch implementation of ASTGCN and MSTCGN. The pytorch version of ASTGCN released here only consists of the  recent component, since the other two components have the same network architecture. 

# Reference

```latex
@inproceedings{guo2019attention,
  title={Attention based spatial-temporal graph convolutional networks for traffic flow forecasting},
  author={Guo, Shengnan and Lin, Youfang and Feng, Ning and Song, Chao and Wan, Huaiyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={922--929},
  year={2019}
}
```

# Configuration

Step 1: The loss function and metrics can be set in the configuration file in ./configurations

Step 2: The last three lines of the configuration file are as follows:

  ```c++
  loss_function = masked_mae
  metric_method = mask
  missing_value = 0.0
  ```

loss_function can choose 'masked_mae',  'masked_mse',  'mae',  'mse'. The loss function with a mask does not consider  missing values.

metric_method can choose 'mask', 'unmask'. The metric with a mask does not evaluate missing values.

The missing_value is the missing identification, whose default value is 0.0

# Datasets

Step 1: Download PEMS04 and PEMS08 datasets provided by [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). 

Step 2: Process dataset

- on PEMS04 dataset

  ```shell
  python prepareData.py --config configurations/PEMS04_astgcn.conf
  ```

- on PEMS08 dataset

  ```shell
  python prepareData.py --config configurations/PEMS08_astgcn.conf
  ```



# Train and Test

- on PEMS04 dataset

  ```shell
  python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf
  ```

- on PEMS08 dataset

  ```shell
  python train_ASTGCN_r.py --config configurations/PEMS08_astgcn.conf
  ```

  

  



