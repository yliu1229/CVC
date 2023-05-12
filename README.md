## Contrastive Visual Clustering for Improving Instance-level Contrastive Learning as a Plugin

This is the PyTorch implementation of Contrastive Visual Clustering (CVC).

It contains a baseline self-supervised learning framework integrated with CVC for image and video representation learning respectively.

For image, training and evaluating are using ./Train/main_2d.py and ./Evaluate/test_2d.py; 
For video, training and evaluating are using ./Train/main_3d.py and ./Evaluate/test_3d.py.

### Prepare

Please follow the instruction in ./ProcessData/readme.md to prepare the datasets.

We provide pre-trained ViT-S and Uniformer-S for quick test.

### Self-supervised pre-training with CVC

Edit arguments with your own settings and simply run: python main_2d/main_3d.

* example: train baseline with CVC using 1 GPUs on ImageNet1K dataset for 300 epochs
  ```
  python main_2d.py --net vit_small --dataset imagenet1k --epochs 300
  ```

### Linear probing evaluation

Edit arguments with your own settings and simply run: python test_2d/test_3d.

* example: test the pre-trained model with linear probing (replace `{model.pth.tar}` with your pre-trained model)
  ```
  python test_2d.py --net vit_small --dataset imagenet1k --train_what last --test {model.pth.tar}
  ```









