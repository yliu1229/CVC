## Contrastive Visual Clustering for Improving Instance-level Contrastive Learning as a Plugin

This is the PyTorch implementation of Contrastive Visual Clustering (CVC).

It provides a baseline self-supervised learning framework integrated with CVC for image and video representation learning respectively.

For image, training and evaluating are using ./Train/main_2d.py and ./Evaluate/test_2d.py; 
For video, training and evaluating are using ./Train/main_3d.py and ./Evaluate/test_3d.py.

### Prepare

Please follow the instruction in ./ProcessData/readme.md to prepare the datasets.

### Self-supervised pre-training with CVC

Edit arguments with your own settings and simply run: python main_2d/main_3d.

* example: train baseline with CVC using 1 GPUs on ImageNet1K dataset for 300 epochs
  ```
  python main_2d.py --net vit_small --dataset imagenet1k --epochs 300
  ```

Please be noted that CVC is prefered to be integrated with well-pretrained models.

### Linear probing evaluation

Edit arguments with your own settings and simply run: python test_2d/test_3d.

* example: test the pre-trained model with linear probing (replace `{model.pth.tar}` with your pre-trained model)
  ```
  python test_2d.py --net vit_small --dataset imagenet1k --train_what last --test {model.pth.tar}
  ```

### Citation
If you like our work please cite it as follows:

```bibtex
@misc{CVC,
  author = {Yue Liu, Xiangzhen Zan, Xianbin Li, Wenbin Liu, Gang Fang},
  title = {Contrastive visual clustering for improving instance-level contrastive learning as a plugin},
  journal = {Pattern Recognition},
	volume = {154},
	pages = {108710},
	year = {2024},
	url   = {https://doi.org/10.1016/j.patcog.2024.110631}
}
```






