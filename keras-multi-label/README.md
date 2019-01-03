# Multi-label classification with Keras

This tutorial, with minor tweaks: https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/

## Get your Python environment setup

Python 3.6 is required in order to use Tensorflow. Python 3.7 is not yet supported as of Jan 2, 2019. 

```
conda create --name keras-multi-label python=3.6
source activate keras-multi-label
pip install -r requirements.txt
```

### (optional) Special Tensorflow setup for GPU

Getting your GPU working with tensorflow may be tricky. 
Ideally, you could just do a `pip install tensorflow-gpu`.

Unfortunately, I had to build tensorflow from source then install.
After installing requirements as shown above, I then run:

```
pip uninstall tensorflow
pip install /tmp/tensorflow_pkg/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
```

## Train the model

```
python train.py --dataset ignored --model setgame.model --labelbin setgame.pickle
```

This outputs a Tensorflow model, as well as a pickle file of labels (e.g. "red", "squiggle"), which is required to interpret the results of running the model.

In my testing, the resulting `.model` file is about 100 megabytes.

## Classify an image

```
python classify.py --model setgame.model --labelbin setgame.pickle --image ../images/single-card-from-internet/2-red-diamond-lines.png
```

