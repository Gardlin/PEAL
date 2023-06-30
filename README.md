# PEAL:  Prior-embedded Explicit Attention Learning for Low-overlap Point Cloud Registration [CVPR-2023]
This is the official repo of CVPR 2023 paper :  '' PEAL: Prior-embedded Explicit Attention Learning for Low-overlap Point Cloud Registration ''

Junle Yu, Luwei Ren,  Wenhui Zhou, Yu Zhang, Lili Lin, Guojun Dai

<div  align="center">  
<img src="https://github.com/Gardlin/PEAL/blob/main/assets/iter_sample.gif" alt="show" align=center  />
</div>  


# Install packages and other dependencies

Follow [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) to install the dependencies.

```
pip install -r requirements.txt
python setup.py build develop
```

# DATA 
[3D prior data for trainging](https://drive.google.com/file/d/1voIJkBUr1uvl0QPZN03S4XrNgaBt3n4U/view?usp=sharing) 

[3D prior data for testing/validating](https://drive.google.com/file/d/1COsTRJu48xz-qakfMnA7DZsDlfGVYbxa/view?usp=sharing) Unzip these files, you can get two folders, one is for training, the other is for testing/validating.

[Point cloud data](https://github.com/prs-eth/OverlapPredator)
The dataset is downloaded from [PREDATOR](https://github.com/prs-eth/OverlapPredator).

## Training
The code for 3DMatch/3DLoMatch is in `experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn`. 
You need to modify the source paths in  config.py.
```
_C.data.dataset_root -> point cloud data root
_C.train.geo_train_prior ->  3D prior data for training
_C.train.geo_prior  ->  3D prior data for testing/validating
```
Use the following command for training.
```bash
 python trainval.py
```

## Testing
```
sh eval_all_finetune.sh 3DMatch(or 3DLoMatch)
```

## [Pretrained models](https://drive.google.com/file/d/1BMiymZgTXBCbcqh7h-2KdF8lHWgJCvXQ/view?usp=sharing)

## PEAL-2dprior
Stay tuned, to be released.

## Human-guided prior for low-overlap point cloud registration
Using the model of PEAL-3dprior for interactive low-overlap point cloud registration. Stay tuned.

## [PREDATOR](https://github.com/prs-eth/OverlapPredator) backbone
Stay tuned.

## Acknowledgements
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)

We thank the respective authors for open sourcing their methods, the code is heavily borrowed from [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
