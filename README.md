# Thermal2Thermal

This repo releases part of the code for the paper: [Cross-Spectrum Thermal Face Pattern Generator
](https://ieeexplore.ieee.org/document/9684443).

The Dataset used for the T2T conversion is [Carl Database](http://splab.cz/en/download/databaze/carl-database). As this database is closed-source, this repo doesn't include any 'npz' file or trained model.

- model.py file is the script to train the cGAN model to generate thermal images.
- temperaturePredictor.py is the script to train the temperature predictor.
- faceRecognition directory contains the files of face recognition.
- T2T.gif is a result fig from the paper: ![img](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6287639/9668973/9684443/cao7ab-3144308-large.gif)