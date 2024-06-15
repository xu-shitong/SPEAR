## SPEAR: Receiver-to-Receiver Acoustic Neural Warping Field

[Yuhang He](https://yuhanghe01.github.io/),
[Shitong Xu](https://github.com/xu-shitong),
[Jiaxing Zhong](https://www.cs.ox.ac.uk/people/jiaxing.zhong/),
[Sangyun Shin](https://www.cs.ox.ac.uk/people/sangyun.shin/),
[Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/),
[Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/)<br>
Department of Computer Science, University of Oxford. Oxford. UK.

**TL:DR**: a novel framework **SPEAR** for receiver-to-receiver spatial acoustic effects prediction. Contrary to traditonal methods that model spatial acoustic effects from source-to-receiver, **SPEAR** warps the spatial acoustic effects from one reference receiver to the target receiver.

<img src=res/SPEAR_mot.jpg></a>
**SPEAR Motivation**: A stationary audio source is emitting audio in 3D space. Requiring neither source position nor 3D space acoustic properties, SPEAR simply requires two microphones to actively record the spatial audio independently at discrete positions. During training, SPEAR takes as input a pair of receiver positions and outputs a warping field potentially warping the recorded audio on reference position to target position. Minimizing the discrepancy between the warped audio and recorded audio enforces SPEAR to acoustically characterise the 3D space from receiver-to-receiver perspective. The learned \emph{SPEAR} is capable of predicting spatial acoustic effects at arbitrary positions.

> **Abstract**: 
We present *SPEAR*, a continuous receiver-to-receiver acoustic neural warping field for spatial acoustic effects prediction in an acoustic 3D space with a single stationary audio source. Unlike traditional source-to-receiver modelling methods that require prior space acoustic properties knowledge to rigorously model audio propagation from source to receiver, we propose to predict by warping the spatial acoustic effects from one reference receiver position to another target receiver position, so that the warped audio essentially accommodates all spatial acoustic effects belonging to the target position. *SPEAR* can be trained in a data much more readily accessible manner, in which we simply ask two robots to independently record spatial audio at different positions. We further theoretically prove the universal existence of the warping field if and only if one audio source presents. Three physical principles are incorporated to guide *SPEAR* network design, leading to the learned warping field physically meaningful. We demonstrate *SPEAR* superiority on both synthetic, photo-realistic and real-world dataset, showing the superiority of *SPEAR*.
> 
<!-- Details of the model architecture and experimental results can be found in [our paper](https://arxiv.org/abs/2312.11269). -->

### Challenge
<img src=res/warpfield_irregu_vis_v2.jpg></a>
The complex behavior of spatial audio propagation in an enclosed 3D space often results in significantly different spatial acoustic effects even for audio recorded at neighboring positions. Plot C above indicates that a 0.3-meter position change results in a significantly more pronounced warping field variation (much lower SSIM score) compared to the RGB images.

TODO: explain the task using Fig A B in this section, or simply remove the image A B.

### Main results
<div style="display: flex; align-items: flex-start;">

  <!-- Metric section -->
  <div style="margin-left: 0px;">
    <img src="TODO.jpg"></a>
  </div>

  <!-- Visualization section -->
  <div style="margin-left: 260px;">
    <img src=res/warpfield_vis.jpg></a>
  </div>

</div>

We qualitatively compare our model with three other baseline methods on 5 metrics, from which we can see that *SPEAR* outperforms all the three comparing methods by a large margin (except for one PSNR metric). 

We also provide qualitative comparison of the predicted warping field, in which we visualize the warping field (real-part) on a synthetic dataset (A) and real-world dataset (B) gnererated by all 4 methods. We can clearly observe that our *SPEAR* is more capable of generating the irregularity pattern from the input position pair than the other baseline models. 

### Create envirionment
The experiment environment is given in file `environment.txt`. The code has been tested on Ubuntu 22.04.

### Generate synthetic data
To generate the synthetic train and test data using Pyroomacoustic, run the following command

```python
python data/R2RGenerator.py train
python data/R2RGenerator.py test
```

### Train
To train a model, run 
```shell
python train.py config/Hyperparameter.yaml
```

### Test
To evaluate a model's test performance metric, uncomment corresponding lines in `test.py` and run the file. 
```shell
python test.py
```

<!-- ### Pretrained model
The pretrained models on all 4 scenes could be found in the `pretrained_models` folder. -->


<!-- 
### [ScanNetv2](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d?metric=ap)

| Dataset | AP | AP_50 | Config | Checkpoint
|:-:|:-:|:-:|:-:|:-:|
| ScanNet val | 62.6 | 81.9 | [config](configs/scannetv2/spherical_mask.yaml) | [checkpoint](https://drive.google.com/file/d/1WJtBr3nxaCaGCA_z1_dpu9bISnPAoxoL/view?usp=drive_link) 
     
* December, 2023: Spherical Mask achieves state-of-the-art in ScanNet-V2 3D instance segmentation. [[Link]](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d?metric=ap) [[Screenshot]](docs/leaderboard_2204.png)
  
For the best training result, we recommend initializing the encoder with the pretrained-weights checkpoint([Download](https://drive.google.com/file/d/1OeHRgkEkxvPkUOrFacmNUevrrAjHW6DA/view?usp=drive_link)) from [ISBNet](https://arxiv.org/abs/2303.00246). 
After downloading the pre-trained weights, please specify the path in configs/scannetv2/spherical_mask.yaml
```shell
# train 
python tools/train.py configs/scannetv2/spherical_mask.yaml --trainall --exp_name defaults
# test
python tools/test.py configs/scannetv2/spherical_mask.yaml --ckpt path_to_ckpt.pth
```
The code has been tested using torch==1.12.1 and cuda==11.3 on Ubuntu 20.04.  -->

**Please CITE** our paper if you found this repository helpful for producing publishable results or incorporating it into other software.
```bibtext
todo
```

<!-- ## Datasets :floppy_disk:

- [x] ScanNetV2 -->

## Acknowledgements :clap:
todo

## Contacts :email:
If you have any questions or suggestions about this repo, please feel free to contact me (todo).



<!-- # Receiver2Receiver-Warp-Field

## Create envirionment
The experiment environment is given in file `environment.txt` 

## Generate synthetic data
To generate the synthetic train and validation data using pyroomacoustic, run the following command
```
python R2RGenerator.py train
python R2RGenerator.py val
```

## Train
To train a model, run 
```
python main_fft.py Hyperparameter_1d_fft.yaml
```




Generate the 


only need to change line 13 to 26 line, and line 88, line 119 and line 120

step_size: distance between mics, meansured in meters

audio_name: file name of source audio, file is "res/{audio_name}.wav" 

scale: times source audio by certain scale
- **change this term so that during training, the loss term `waveform_l` is between 5000 to 10000.**
- **to achieve this scale, audio waveform amplitude should mostly be at 0.1**

uncomment line 20-22 to generate train data

uncomment line 24-26 to generate validation data

line 88: define sound source

line 119-120: define mic grid

# run training
python useful_files/main_fft.py useful_files/Hyperparameter_1d_fft.yaml

## Hyperparameter_1d_fft.yaml

### important parameters
1. scene_x: 5 scene_y: 3: scene size in meters
  - **this defines the region covered by the grid feature**
2. line 16-21: NAF model
3. line 23-30: our transformer model
  - **line 17, line 24**: grid_size: [192, 8, 16], grid feature of [192, 8, 16] is used as position encoding grid. 
    - **16 correspond to x axis, 8 correspond to y axis**
  - line 30: pred_range: 16384: our model predicts 16384 length warp field, and being mirrored to get 32k length warp field
4. epoch_num: 6000: in totol, train for 6000 epoch
5. dataset_tag: "yyy": "yyy_train_data" is the training data folder, "yyy_val_data" is the validation data folder
6. line 36-37: sample_size: 3000, val_sample_size: 128: limit model to use 3000 sample for training and 128 sample for validation

## less important parameters
1. gpu: "2": using gpu node 2
2. save_epoch: 2000: save model each 2000 epoch
3. load_model: "xxx.pt": load model in initialization and train
4. scene_z: size of only used by model that predict mic with varying z. can ignore

 -->
