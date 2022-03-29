import timm.data as tdata
import torch
import random
import numpy as np
import mmcv
from mmaction.datasets import PIPELINES
from mmaction.datasets.pipelines import Resize

@PIPELINES.register_module()
class RandomErasing(tdata.random_erasing.RandomErasing):
    def __init__(self, device='cpu', **args):
        super().__init__(device=device, **args)

    def __call__(self, results):
        in_type = results['imgs'][0].dtype.type

        rand_state = random.getstate()
        torchrand_state = torch.get_rng_state()
        numpyrand_state = np.random.get_state()
        # not using cuda to preserve the determiness

        out_frame = []
        for frame in results['imgs']:
            random.setstate(rand_state)
            torch.set_rng_state(torchrand_state)
            np.random.set_state(numpyrand_state)
            frame = super().__call__(torch.from_numpy(frame).permute(2, 0, 1)).permute(1, 2, 0).numpy()
            out_frame.append(frame)

        results['imgs'] = out_frame
        img_h, img_w, _ = results['imgs'][0].shape

        out_type = results['imgs'][0].dtype.type
        assert in_type == out_type, \
            ('Timmaug input dtype and output dtype are not the same. ',
             f'Convert from {in_type} to {out_type}')

        if 'gt_bboxes' in results:
            raise NotImplementedError('only support recognition now')
        assert results['img_shape'] == (img_h, img_w)

        return results
    
    
@PIPELINES.register_module()
class CCResize(Resize):
    def __init__(self,
                 crop_driver=False,
                 *args, **kwargs):
        self.crop_driver = crop_driver
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        if self.crop_driver:
            h,w = results['img_shape']
            crop_driver_imgs = []
            for img in results['imgs']:
                if 'Dashboard' in results['filename']:
                    vr_frame = img[:,380:w-120]
                elif 'Right_side' in results['filename']:
                    vr_frame = img[:, 750:]
                else:
                    vr_frame = img[:, 750:]
                crop_driver_imgs.append(mmcv.imresize(vr_frame, (w,h)))
            results['imgs'] = crop_driver_imgs
        return super().__call__(results)