## SPEAR: Receiver-to-Receiver Acoustic Neural Warping Field

### Generate synthetic data
To generate the synthetic train and test data using Pyroomacoustic, run the following command
```shell
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
