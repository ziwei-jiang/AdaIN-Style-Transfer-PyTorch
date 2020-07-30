AdaIN Style Transfer-PyTorch
============================
A Pytorch implementation of Style Transfer with Adaptive Instance Normalization based on the paper [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
](https://arxiv.org/abs/1703.06868).

Requirement
----------------------------
* Argparse
* Numpy
* Pillow
* Python 3.7
* PyTorch
* TorchVision
* tqdm


Usage
----------------------------

### Training

Download the data to the ./data/ folder. The network is trained using MSCOCO and wikiart dataset. Download the [weight of the vggnet](https://drive.google.com/file/d/1UcSl-Zn3byEmn15NIPXMf9zaGCKc2gfx/view?usp=sharing) to build the encoder.
Run the script train.py
```
$ python train.py --trainset_dir $TRAINDIR --cuda

usage: train.py [-h] [--content_dir CONTENT_DIR] [--style_dir STYLE_DIR]
                [--epochs EPOCHS] [--resume RESUME] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --content_dir CONTENT_DIR
                        content data set path
  --style_dir STYLE_DIR
                        style data set path
  --epochs EPOCHS       training epoch number
  --resume RESUME       continues from epoch number
  --cuda                Using GPU to train
```

### Testing

Download the [decoder weight](https://drive.google.com/file/d/18JpLtMOapA-vwBz-LRomyTl24A9GwhTF/view?usp=sharing).

Run the script test_image.py

```
$ python test_image.py --input_image $IMG --style_image $STYLE --weight $WEIGHT --cuda

usage: test_style_transfer.py [-h] [--input_image INPUT_IMAGE]
                              [--style_image STYLE_IMAGE] [--weight WEIGHT]
                              [--alpha {Alpha Range}] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --input_image INPUT_IMAGE
                        test image
  --style_image STYLE_IMAGE
                        style image
  --weight WEIGHT       decoder weight file
  --alpha {Alpha Range}
                        Level of style transfer, value between 0 and 1
  --cuda                Using GPU to train
```

