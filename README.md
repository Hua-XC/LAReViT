# Local-Aware Residual Attention Vision Transformer (LAReViT)
Pytorch code for paper "**Local-Aware Transformers with Attention Residual Connections for Visible-Infrared Person Re-Identification**"
**\*Accept by ACM Transactions on Multimedia Computing, Communications, and Applications(TOMM)**

### 1. Results
We adopt the Transformer as backbone respectively.

|   Datasets   | Backbone | Rank@1  | Rank@10| Rank@20  | mAP     |  mINP |  Model |  - |
|:-------:|:---:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|#SYSU-MM01  | Transformer  |  76.71% |  97.33% | 99.05% | 72.95% | 60.18% |[GoogleDrive](https://drive.google.com/file/d/1oR6o6TnNCCfEEdgOu08n4NmSt0DN8SoK/view?usp=sharing)|[Baidu Netdisk](https://pan.baidu.com/s/1r2UwL95RtvFgZ_RUCSPLEw?pwd=0508)|

**\*The results may exhibit fluctuations due to random splitting, and further improvement can be achieved by fine-tuning the hyperparameters.**

### 2. Datasets

- RegDB [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html).

- SYSU-MM01 [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

  - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

    ```
    python pre_process_sysu.py
    ```

- LLCM [5]: The LLCM dataset can be downloaded by sending a signed [dataset release agreement](https://github.com/ZYK100/LLCM/blob/main/Agreement/LLCM%20DATASET%20RELEASE%20AGREEMENT.pdf) copy to zhangyk@stu.xmu.edu.cn. 


### 3. Training


**Train LAReViT by**

```
python train.py --dataset sysu --gpu 0
```
- `--dataset`: which dataset "sysu", "regdb" or "llcm".

- `--gpu`: which gpu to run.

*You may need manually define the data path first.*



### 4. Testing

**Test a model on SYSU-MM01 dataset by**

```
python test.py --dataset 'sysu' --mode 'all' --resume 'model_path'  --gpu 0
```
  - `--dataset`: which dataset "sysu".
  - `--mode`: "all" or "indoor"  (only for sysu dataset).
  - `--resume`: the saved model path.
  - `--gpu`: which gpu to use.



**Test a model on RegDB dataset by**

```
python test.py --dataset 'regdb' --resume 'model_path'  --tvsearch True --gpu 0
```

  - `--tvsearch`:  whether thermal to visible search  True or False (only for regdb dataset).


**Test a model on LLCM dataset by**

```
python test.py --dataset 'llcm' --resume 'model_path'  --gpu 0
```

### 5. Citation

Most of the code of our backbone are borrowed from [AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [3] and [CAJ](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [4]. Most of the code related to LLCM dataset are borrowed from [DEEN](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [5]. 

Thanks a lot for the author's contribution.

Please cite the following paper in your publications if it is helpful:

```
Xuecheng Hua, Ke Cheng,Gege Zhu, Hu Lu, Yuanquan Wang, Shitong Wang,
Local-Aware Transformers with Attention Residual Connections for Visible-Infrared Person Re-Identification,
ACM Transactions on Multimedia Computing, Communications, and Applications,
2025,

**https://doi.org/10.1016/j.patcog.2024.111090.**
```

###  6. References.

[1] Lu H, Zou X, Zhang P. Learning progressive modality-shared transformers for effective visible-infrared person re-identification[C]//Proceedings of the AAAI conference on Artificial Intelligence. 2023, 37(2): 1835-1843.

[2] He S, Luo H, Wang P, et al. Transreid: Transformer-based object re-identification[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 15013-15022.

[3] Diko A, Avola D, Cascio M, et al. ReViT: Enhancing Vision Transformers with Attention Residual Connections for Visual Recognition[J]. arXiv preprint arXiv:2402.11301, 2024.

[4] Ni H, Li Y, Gao L, et al. Part-aware transformer for generalizable person re-identification[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 11280-11289.

