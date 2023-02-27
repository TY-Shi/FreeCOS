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

#### 4. Trained models
Trained models can be downloaded from here. [[Google Drive](https://drive.google.com/file/d/1wtATuEFbZPZ06k_C_T5gV59u-_eaH3cJ/view?usp=sharing)] [[Baidu Drive](https://pan.baidu.com/s/1_r3CFhW-qjJZD2nE5iIBzw) (download code: urc2) ].   
Put the weights in the "logs/" directory.  

#### 4. Trained Data
Trained Data can be down from here. (you can generate different curvilinear data by our method or use the same generated curvilinear data as our experiment. Different generated curvilinear data will effect the performance)

## Future Work

FreeCOS will be continuously updated.
Thanks for the parts of LIOT codes from "Local Intensity Order Transformation for Robust Curvilinear Object Segmentation" (https://github.com/TY-Shi/LIOT), we changes the LIOT to a online way in FreeCOS codes.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- shitianyihust@hust.edu.cn
- shitianyihust@126.com
