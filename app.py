# SEE: https://blog.keras.io/category/tutorials.html
# 
# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
import keras.models
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import os

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
graph = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model

    # TODO: Replace with custom model for Set
    #model = ResNet50(weights="imagenet")
    model = keras.models.load_model('./first_try.model')


    # https://github.com/keras-team/keras/issues/2397#issuecomment-254919212
    import tensorflow as tf
    global graph
    graph = tf.get_default_graph()

def prepare_image(image, target):

    # TODO: Add any relevant pre-processing for Set images
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            print("HERE")
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            #image = prepare_image(image, target=(224, 224))
            image = prepare_image(image, target=(150, 150))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                preds = model.predict(image)

                # TODO: USE CUSTOM PREDICTIONS INSTEAD
                # results = imagenet_utils.decode_predictions(preds)
                # data["predictions"] = []

                # # loop over the results and add them to the list of
                # # returned predictions
                # for (imagenetID, label, prob) in results[0]:
                #     r = {"label": label, "probability": float(prob)}
                #     data["predictions"].append(r)

                print(preds)
                class_num = int(preds[0][0]) + 1
                data["predictions"] = class_num

                # indicate that the request was a success
                data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
print(("* Loading Keras model and Flask starting server..."
    "please wait until server has fully started"))
load_model()

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
