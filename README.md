# kilter.net

Using a Convolutional, Conditional Variational Autoencoder for Kilter Board analysis conditional on the difficulty grade and angle.

Models and training scripts provided by [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)


To generate the training data, go into the data_preprocessing folder and run: `python clean_kilter_df.py`

### Required packages
To install requirements:
```
mamba create -n kilternet python=3.9 -y
mamba activate kilternet
mamba install -y -c conda-forge --file requirements.txt
```
