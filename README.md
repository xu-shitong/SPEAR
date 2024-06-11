# Receiver2Receiver-Warp-Field

## generate synthetic data
To generate the synthetic train and validation data using pyroomacoustic, run the following command
```


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


