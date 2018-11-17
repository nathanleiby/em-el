# em-el
Learning Machine Learning

# Running locally

Install VirtualEnv, then setup your Python environment.

```
mkvirtualenv em-el
workon em-el
pip install -r requirements.txt
```

## Running the Notebook

Run notebook
```
jupyter notebook
```

If needed, setup a kernel to access from Jupyter Notebook.
More info: https://stackoverflow.com/a/43234969/950683
```
python -m ipykernel install --user --name=em-el
```

## Running Image Segmentation

```
python3 image_segmentation.py
```


# Tracking large files in Git with Git LFS

[Installing Git LFS](https://help.github.com/articles/installing-git-large-file-storage/)

[Using Git LFS](https://help.github.com/articles/configuring-git-large-file-storage/)


# Set Identification

We like that this has a "did you do it correctly" evaluatability.

[Scikit Image Tutorial](https://github.com/scikit-image/skimage-tutorials)

## Stages of the project:

    # (0) Prepare training data
         # take more pictures (how many instances of each card in the deck do we need?)
         # [a] take pictures with many cards - @nick
         # [b] take pictures with a single card - @nate
         # goal - 3 images of each card
         # differences - lighting/shadows, oblique camera angle, background, crappy camera image quality
             # automated approaches to get variation: add noise
    # (1) Given a picture with cards in it, extract the cards
    # (2) Given an image of a card, tell us what card it is in a digital representation
         # perfectly cropped, high quality
         # low quality image
    # Solve the set problem (tell me the complete list of all sets in the picture)
    # Submit a picture to a website and solve the problem (available via web API or similar)
        # Make the web app work nicely from a phone (make it shareable)
        # How much can we make this interesting or interactive- real time boxing of cards, supervision, etc.


    # continued learning from user-submitted images ("that's wrong" type feedback from users)
    # meta problem: identify any card from any card game
    # at each step of the way, statistics to evaluate success of the algorithm.  False Positive rate, algorithm success
    # be able to solve arbitrary images of set game

## Questions
    # Can we throw a whole image into a NN and get it to pull out the individual cards?

    # should we pre-process the image to ID cards and run them into a model then

    # map vector of each card: slot for each

    # namedtuple? Card(color, shape, count, pattern)
        Red, Green, Purple (RGP)
        Diamond, Oval, Squiggle (DOS)
        Empty, Half, Full (EHF)
        1 2 3 (123)
        ex. "rde1.png"

    # any image format that's better?
        => png for now

    # mapping between image files with cards and instances of labels

## General Problems to Solve

- removing shadows (or any major bright / dark spots)
    - https://stackoverflow.com/questions/42918752/how-to-remove-the-shadows-from-these-pictures-using-python-opencv
    - https://stackoverflow.com/questions/9081930/how-to-remove-the-shadow-in-image-by-using-opencv
    - http://aqua.cs.uiuc.edu/site/projects/shadow.html
    - http://dhoiem.web.engr.illinois.edu/publications/cvpr11_shadow.pdf

