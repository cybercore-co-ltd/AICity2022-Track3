import numpy as np
import cv2
from mmaction.datasets.builder import PIPELINES
from PIL import Image, ImageFilter
from mmaction.datasets.pipelines import ColorJitter
from mmaction.datasets.pipelines import Normalize, FormatShape


@PIPELINES.register_module
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max, p=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = p

    def __call__(self, results):
        imgs = results['imgs']
        out = []
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        for img in imgs:
            img = Image.fromarray(np.uint8(img)).convert('RGB')
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            img = np.array(img)
            out.append(img)

        if np.random.rand() > (1 - self.prob):
            results['imgs'] = out
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class ProbColorJitter(ColorJitter):
    """Inherit ColorJitter function from mmaction2 -version 0.17.0 with adding probability for augmentation.

    Args:
        brightness (float | tuple[float]): The jitter range for brightness, if
            set as a float, the range will be (1 - brightness, 1 + brightness).
            Default: 0.5.
        contrast (float | tuple[float]): The jitter range for contrast, if set
            as a float, the range will be (1 - contrast, 1 + contrast).
            Default: 0.5.
        saturation (float | tuple[float]): The jitter range for saturation, if
            set as a float, the range will be (1 - saturation, 1 + saturation).
            Default: 0.5.
        hue (float | tuple[float]): The jitter range for hue, if set as a
            float, the range will be (-hue, hue). Default: 0.1.
        p (np.float | 0.5): probability of using this function. Default: 0.5.
    """

    def __init__(self, p=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = p

    def __call__(self, results):
        if np.random.rand() > (1 - self.prob):
            results = super().__call__(results)
        return results


@PIPELINES.register_module
class GrayScale(object):

    def __init__(self, p=None):
        self.p = None

    def __call__(self, results):
        results['imgs'] = [cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
                           for _img in results['imgs']]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Stack3Gray2RGB(object):

    def __init__(self, p=None):
        self.p = None

    def __call__(self, results):
        imgs = [cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
                for _img in results['imgs']]
        imgs = [np.stack(imgs[i:i + 3], axis=2)
                for i in range(0, len(imgs), 3)]
        results['imgs'] = imgs
        results['clip_len'] = results['clip_len'] // 3
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class GrayNormalize(Normalize):
    """
    Inherit from class Normalize of mmaction 
    but adjusting mean and standard deviation to be compatible with grayscale image
    """

    def __init__(self, *args, **kwargs):
        super(GrayNormalize, self).__init__(*args, **kwargs)
        self.mean_gray = np.mean(self.mean.copy())
        self.std_gray = np.mean(self.std.copy())

    def __call__(self, results):
        tmp_imgs = np.array(results['imgs'])
        tmp_imgs = (tmp_imgs - self.mean_gray) / self.std_gray
        results['imgs'] = np.expand_dims(tmp_imgs, axis=-1)  # [b, h, w, C=1]

        return results


@PIPELINES.register_module()
class GrayFormatShape(FormatShape):
    """Inherit from FormatShape 
    but converting NCTHW to NTHW format for Stacked Frames"""

    def __init__(self, *args, **kwargs):
        super(GrayFormatShape, self).__init__(*args, **kwargs)

    def __call__(self, results):
        super().__call__(results)
        imgs = results['imgs']
        results['imgs'] = imgs.reshape(-1, imgs.shape[1]
                                       * imgs.shape[2], imgs.shape[3], imgs.shape[4])
        results['imgs'] = np.float32(results['imgs'])
        return results


@PIPELINES.register_module
class SuperImage(object):
    """Transform each sub-clip to super image 
    by concatenating images with a layout (sh, sw)"""

    def __init__(self, width=3, resize=None):
        self.width = width
        self.resize = resize

    def __call__(self, results):
        imgs = results['imgs']
        N, C, T, H, W = imgs.shape
        num_odd_images = self.width - (T % self.width)
        padding_images = np.zeros((N, C, num_odd_images, H, W))
        new_imgs = np.concatenate([imgs, padding_images], axis=2)
        rows = [
            new_imgs[:, :, i * self.width: (i + 1) * self.width, :, :] for i in range(self.width)]
        new_imgs = np.concatenate(
            [np.concatenate([row[:, :, i, :, :] for i in range(self.width)], axis=3) for row in rows], axis=2)

        results['imgs'] = np.asarray(new_imgs, dtype=np.float32)
        if self.resize:
            B, C, H, W = results['imgs'].shape
            results['imgs'] = np.transpose(results['imgs'], (0, 2, 3, 1))
            new_images = np.zeros((B, self.resize, self.resize, C))
            for i in range(len(results['imgs'])):
                new_images[i] = cv2.resize(results['imgs'][i],
                                           (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
            results['imgs'] = np.asarray(np.transpose(
                new_images, (0, 3, 1, 2)), np.float32)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class SuperGrayImage(SuperImage):
    """Inherit from SuperImage class. 
    Stacking Gray_frames images as RGB images before calling SuperImage"""

    def __init__(self, super_feature=False, with_focal_window=False, num_stack=3, *args, **kwargs):
        self.super_feature = super_feature
        self.with_focal_window = with_focal_window
        self.num_stack = num_stack
        self.focalwindow_super_image = SuperImage(width=4, resize=None)
        super(SuperGrayImage, self).__init__(*args, **kwargs)

    def _stack_gray2rgb(self, imgs):
        # stack gray images
        if len(imgs.shape) == 5:
            N, C, T, H, W = imgs.shape
        elif len(imgs.shape) == 4:
            # gray stacked: C=1
            N, T, H, W = imgs.shape
        else:
            # do nothing
            pass

        if T % self.width != 0:
            raise ValueError("Num_clip_len is not relevant to width argument")
        else:
            window = T // self.num_stack
        imgs = imgs.reshape(N, window, -1, H, W)
        imgs = np.transpose(imgs, (0, 2, 1, 3, 4))
        return imgs
        # shuffle_spatial = np.arange(window)
        # np.random.shuffle(shuffle_spatial)
        # shuffle_chanels = np.arange(self.num_stack)
        # np.random.shuffle(shuffle_chanels)
        # imgs= imgs[:,shuffle_chanels,:,:, :]
        # return imgs[:,:,shuffle_spatial,:, :]
        # return imgs

    def __call__(self, results):
        results['imgs'] = self._stack_gray2rgb(results['imgs'])
        if self.super_feature: return results
        if not self.with_focal_window:
            results = super().__call__(results)

        # ----------------- run with focal window
        else:
            imgs = results['imgs']
            center_size_h, center_size_w = imgs.shape[3] * 2, imgs.shape[4] * 2
            surround_size_h, surround_size_w = imgs.shape[3], imgs.shape[4]

            # get center img
            center_img = np.zeros(
                (imgs.shape[0], center_size_h, center_size_w, imgs.shape[1]))
            for i in range(len(imgs)):
                _img = np.transpose(
                    imgs[:, :, (imgs.shape[2] // 2)][i], (1, 2, 0))
                center_img[i] = cv2.resize(_img,
                                           (center_size_w, center_size_h), interpolation=cv2.INTER_LINEAR)
            center_img = np.transpose(center_img, (0, 3, 1, 2))

            # ------------ construct super image with center region is substitute imgs (4 zeros substitute imgs)
            center_substitutes = np.zeros(
                (imgs.shape[0], imgs.shape[1], 2, imgs.shape[3], imgs.shape[4]))  # 2 frames
            # tmp_imgs = np.concatenate([imgs[:,:,:5], center_substitutes, imgs[:,:,6:8], center_substitutes, imgs[:,:,8:]], axis=2)
            tmp_imgs = np.concatenate([imgs[:, :, :5], center_substitutes, imgs[:, :, np.newaxis, 5],
                                      imgs[:, :, np.newaxis, 7], center_substitutes, imgs[:, :, 8:]], axis=2)
            results['imgs'] = tmp_imgs
            del tmp_imgs
            results = self.focalwindow_super_image.__call__(results)

            # replace substitute imgs by center_img
            results['imgs'][:, :, surround_size_h:(
                surround_size_h + center_size_h), surround_size_w:(surround_size_w + center_size_w)] = center_img

            if self.resize:
                B, C, W, H = results['imgs'].shape
            results['imgs'] = np.transpose(results['imgs'], (0, 2, 3, 1))
            new_images = np.zeros((B, self.resize, self.resize, C))
            for i in range(len(results['imgs'])):
                new_images[i] = cv2.resize(results['imgs'][i],
                                           (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
            results['imgs'] = np.asarray(np.transpose(
                new_images, (0, 3, 1, 2)), np.float32)

        return results
