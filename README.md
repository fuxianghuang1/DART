
# <p align=center> Unsupervised Robust Domain Adaptation: Paradigm, Theory and Algorithm </p> # 


[NEWS.20251112] **The related [paper](https://arxiv.org/pdf/2511.11009) has been accepted by International Journal of Computer Vision.**

# Quick Usage
DART is applicable to various UDA algorithms; therefore, the code implementation primarily begins with core "Step-2 (Robustification)".


The command to run the Office31 dataset is as follows:

```
CUDA_VISIBLE_DEVICES=0 python DART.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/APDA_mdd/i_fgsm/try/Office31_A2W
```

```
CUDA_VISIBLE_DEVICES=0 python DART.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/APDA_mdd/i_fgsm/try/Office31_D2W
```

```
CUDA_VISIBLE_DEVICES=0 python DART.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/APDA_mdd/i_fgsm/try/Office31_W2D
```

```
CUDA_VISIBLE_DEVICES=0 python DART.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/APDA_mdd/i_fgsm/try/Office31_A2D
```

```
CUDA_VISIBLE_DEVICES=0 python DART.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/APDA_mdd/i_fgsm/try/Office31_D2A
```

```
CUDA_VISIBLE_DEVICES=0 python DART.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/APDA_mdd/i_fgsm/try/Office31_W2A
```
The code primarily relies on previous work from tllib (https://github.com/thuml/Transfer-Learning-Library) and advertorch (https://github.com/BorealisAI/advertorch), for which we thank them for their contributions to the UDA and adversarial defense communities.  Users can install them via `pip install -i https://test.pypi.org/simple/tllib==0.4` and `pip install advertorch`.

If you find this repository is useful for you, please cite our paper:
```
@article{huang2026unsupervised,
  title={Unsupervised Robust Domain Adaptation: Paradigm, Theory and Algorithm: F. Huang etal},
  author={Huang, Fuxiang and Fu, Xiaowei and Ye, Shiyu and Ma, Lina and Li, Wen and Gao, Xinbo and Zhang, David and Zhang, Lei},
  journal={International Journal of Computer Vision},
  volume={134},
  number={1},
  pages={5},
  year={2026},
  publisher={Springer}
}
```
