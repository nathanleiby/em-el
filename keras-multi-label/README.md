# Multi-label classification with Keras

This tutorial, with minor tweaks: https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/

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
