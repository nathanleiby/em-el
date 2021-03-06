1. Improve the single-image model

- [ ] create more labeled training data
- [ ] think about model architecture- particularly are we using kernels big enough to capture shape?  Are dimensions to collapsed to capture shape
- [ ] think about loss function ([more notes](https://github.com/nathanleiby/em-el/blob/master/keras-multi-label/train.py#L76-L89))
- [ ] experiment. iterate and document performance of various models
- [ ] hyperparameter search: grid search learning rate, image dimensions, data augmentation constants


2. Image segmentation model

- [ ] label the images - how many cards are present
- [ ] capture our performance here

3. Ship the single-image, and setup a website that uses it

- [ ] push today's model + labels to heroku site, so the API returns Set game classification results
- [ ] add "image segmentation" support to the API, so it can process images with many cards
- [ ] write logic for: "given X cards, which sets are possible?"
- [ ] minimal UI: file upload, spinning, then showing results
- [ ] phone UI: upload an image from a phone camera, since that's a more natural use case
- [ ] (stretch) get it working from a phone video feed, so we're processing multiple images and taking best estimate across them (ideally: treat images as linked, pay attention to temporal relationship)
- [ ] (stretch) use tensorflow via Pixel 3 GPU

---

- [x] Take more Ikea table photos that are easy to segment... segment them.
- [x] pull minimal image_segmentation logic into a .py file
- [x] Label the segmented output file in our standardized format. => instead, using labelbox JSON file
- [x] Start trying out Keras to process the single image files.
- [x] What's the appropriate input format ... matrix of pixels? other?
- [x] Figure out approx `n` of single card images we need. => so far we have ~200.. I think ideally we have a couple hundred per bucket

