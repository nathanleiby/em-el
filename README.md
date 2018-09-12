# em-el
Learning Machine Learning

# Tracking large files in Git with Git LFS

[Installing Git LFS](https://help.github.com/articles/installing-git-large-file-storage/)

[Using Git LFS](https://help.github.com/articles/configuring-git-large-file-storage/)


# Set Identification

We like that this has a "did you do it correctly" evaluatability.

[Scikit Image Tutorial](https://github.com/scikit-image/skimage-tutorials)

## Stages of the project:

    # Given a picture with cards in it, extract the cards
    # Given an image of a card, tell us what card it is in a digital representation
         # perfectly cropped, high quality
         # low quality image
    # Solve the set problem (tell me the complete list of all sets in the picture)
    # Submit a picture to a website and solve the problem (available via web API or similar)
        # Make the web app work nicely from a phone (make it shareable)
        # How much can we make this interesting or interactive- real time boxing of cards, supervision, etc.
    

    # continued learning from user-submitted images ("that's wrong" type feedback from users)
    # meta problem: identify any card from any card game
    # at each step of the way, statistics to evaluate success of the algorithm.  False Positive rate, algorithm success
    
    
## Questions
    # Can we throw a whole image into a NN and get it to pull out the individual cards?
    
    # should we pre-process the image to ID cards and run them into a model then
    
    # map vector of each card: slot for each 
    
    # namedtuple? Card(color, shape, count, pattern)
    
    # mapping between image files with cards and instances of labels