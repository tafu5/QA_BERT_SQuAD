# Developed by Valentin Tafura - e: valentintafura@hotmail.com

# -----------------------
# Loading model from disk
# -----------------------

import tensorflow as tf
import os

def LoadModel():
    # Get the current directory
    CURRENT_DIRECTORY = os.getcwd()
    # Get the Model Path
    MODELPATH = os.path.join(CURRENT_DIRECTORY, "\..\Model\tf2\tensorflow\1")
    # Load model from the .h5
    loaded_model = tf.compat.v2.saved_model.load(MODELPATH)

    print("MODEL_H5_FILE succesfully loaded")

    return loaded_model
