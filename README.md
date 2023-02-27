# FreeCOS
FreeCOS: Self-Supervised Learning from Fractals and Unlabeled Images for Curvilinear Object Segmentation


## Usage

#### 1.generate the curvilinear object images

```bash
# generate the curvilinear object images in Data/XCAD/make_fakevessel.py
python Data/XCAD/make_fakevessel.py #make_fakevessel.py is an example python script.
```

#### 2. Training scripts

```bash

CUDA_VISIBLE_DEVICES=0 python train_DA_contrast_liot_finalversion.py (CUDA_VISIBLE_DEVICES=0 python train_DA_contrast_liot_DRIVE_finalversion.py for DRIVE)

```

#### 3. Evaluation scripts

```bash

CUDA_VISIBLE_DEVICES=0 python test_DA_thresh.py

```

## Future Work

FreeCOS will be continuously updated.
Thanks for the parts of LIOT codes from "Local Intensity Order Transformation for Robust Curvilinear Object Segmentation" (https://github.com/TY-Shi/LIOT), we changes the LIOT to a online way in FreeCOS codes.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- shitianyihust@hust.edu.cn
- shitianyihust@126.com
