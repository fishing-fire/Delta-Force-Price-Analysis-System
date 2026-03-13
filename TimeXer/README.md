# TimeXer

This repo is the official implementation for the paper: [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2402.19072).

## Introduction
This paper focuses on forecasting with exogenous variables which is a practical forecasting paradigm applied extensively in real scenarios. TimeXer empower the canonical Transformer with the ability to reconcile endogenous and exogenous information without any architectural modifications and achieves consistent state-of-the-art performance on twelve real-world forecasting benchmarks.

<p align="center">
<img src=".\figures\Introduction.png" width = "800" height = "" alt="" align=center />
</p>

## Overall Architecture
TimeXer employs patch-level and variate-level representations respectively for endogenous and exogenous variables, with an endogenous global token as a bridge in-between. With this design, TimeXer can jointly capture intra-endogenous temporal dependencies and exogenous-to-endogenous correlations.

<p align="center">
<img src=".\figures\TimeXer.png" width = "800" height = "" alt="" align=center />
</p>

## Usage 

1. Short-term Electricity Price Forecasting Dataset have alreadly included in "./dataset/EPF". Multivariate datasets can be obtained from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy).

2. Install Pytorch and other necessary dependencies.
```
pip install -r requirements.txt
```
3. Train and evaluate model. We provide the experiment scripts under the folder ./scripts/. You can reproduce the experiment results as the following examples:

```
bash ./scripts/forecast_exogenous/EPF/TimeXer.sh
```

## Main Results
We evaluate TimeXer on short-term forecasting with exogenous variables and long-term multivariate forecasting benchmarks. Comprehensive forecasting results demonstrate that TimeXer effectively ingests exogenous information to facilitate the prediction of endogenous series.

### Forecasting with Exogenous

<p align="center">
<img src=".\figures\Result_EPF.png" width = "800" height = "" alt="" align=center />
</p>

### Multivariate Forecasting

<p align="center">
<img src=".\figures\Result_Multivariate.png" width = "800" height = "" alt="" align=center />
</p>

## Experiments on Large-scale Meteorology Dataset
In this paper, we build a large-scale weather dataset for forecasting with exogenous variables, where the endogenous series is the hourly temperature of 3,850 stations worldwide obtained from the National Centers for Environmental Information (NCEI), and the exogenous variables are meteorological indicators of its adjacent area from the ERA5 dataset. You can obtain this meteorology dataset from [[Google Drive]](https://drive.google.com/file/d/1EuEedepUV2A_cia1plAHwA6fJXNio47i/view?usp=drive_link).

<p align="center">
<img src=".\figures\ERA5.png" width = "800" alt="" align=center />
</p>

## Citation
If you find this repo helpful, please cite our paper.

```
@article{wang2024timexer,
  title={Timexer: Empowering transformers for time series forecasting with exogenous variables},
  author={Wang, Yuxuan and Wu, Haixu and Dong, Jiaxiang and Liu, Yong and Qiu, Yunzhong and Zhang, Haoran and Wang, Jianmin and Long, Mingsheng},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.

Reformer (https://github.com/lucidrains/reformer-pytorch)

Informer (https://github.com/zhouhaoyi/Informer2020)

Autoformer (https://github.com/thuml/Autoformer)

Stationary (https://github.com/thuml/Nonstationary_Transformers)

Time-Series-Library (https://github.com/thuml/Time-Series-Library)

## Concat

If you have any questions or want to use the code, please contact wangyuxu22@mails.tsinghua.edu.cn
