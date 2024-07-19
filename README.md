# Pixel-level Semantic Correspondence through Layout-aware Representation (LPMFlow)

This is the official code for LPMFlow implemented with PyTorch.

# Environment Settings
```
git clone https://github.com/YXSUNMADMAX/LPMFlow
cd LPMFlow
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U scikit-image
pip install git+https://github.com/albumentations-team/albumentations
pip install tensorboardX termcolor timm tqdm requests pandas info-nce-pytorch
```

# Evaluation
- Download pre-trained weights on [Link](链接: https://pan.baidu.com/s/1tM0cyenXE6x6V5MM5vRWSQ) (Keys: 5jnk)

- Result on SPair-71k:
      python test.py --datapath "/path_to_dataset" --pretrained "/path_to_pretrained_model/spair" --benchmark spair

- Results on PF-PASCAL:
  python test.py --datapath "/path_to_dataset" --pretrained "/path_to_pretrained_model/pfpascal" --benchmark pfpascal
  
- Results on PF-WILLOW:
  
  python test.py --datapath "/path_to_dataset" --pretrained "/path_to_pretrained_model/pfwillow" --benchmark pfwillow --thres img

# Acknowledgement 
We borrow code from public projects (Thanks a lot !!!). We mainly borrow code from [CATs ](https://github.com/SunghwanHong/Cost-Aggregation-transformers) and [OSTrack](https://github.com/botaoye/OSTrack). 

### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@inproceedings{sun2024pixel,
  title={Pixel-level Semantic Correspondence through Layout-aware Representation Learning and Multi-scale Matching Integration},
  author={Sun, Yixuan and Yin, Zhangyue and Wang, Haibo and Wang, Yan and Qiu, Xipeng and Ge, Weifeng and Zhang, Wenqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17047--17056},
  year={2024}
}
````
