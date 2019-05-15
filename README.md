# Visual Explanations for CNNs

Some techniques for providing visual explanations of CNN model prediction.

I have Implemented Grad-Cam[1] and Grad-Cam++[2] with PyTorch and I may implement other methods, if I need them someday.

### Requirements

- torch >= 1.0
- torchvision >= 0.2
- numpy >= 1.16
- matplotlib >= 3.0
- PIL >= 6.0
- cv2 >= 4.0

### Usage

```bash
$ python main.py images/mybaby.jpeg
```

You will get output like this

![世界一かわいい猫](https://raw.githubusercontent.com/mori97/visual-explanations/master/images/mybaby_result.jpeg)

**Note**: He was a stray cat and I picked him up 5 years ago. There is no need to explain that He is ***the most beautiful cat*** in the world.

Here is his photos.

![世界一かわいい猫](https://raw.githubusercontent.com/mori97/visual-explanations/master/images/mybaby1.jpeg)

![世界一かわいい猫](https://raw.githubusercontent.com/mori97/visual-explanations/master/images/mybaby2.jpeg)

### Reference

1. [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
1. [Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/abs/1710.11063)
