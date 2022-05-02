import pathlib
import tensorflow as tf


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_fname = model_name.split('/')[-1]
    model_dir = tf.keras.utils.get_file(fname=model_fname, origin=base_url + model_file, untar=True)
    print('model directory: ', model_dir)

    model_dir = pathlib.Path(model_dir) / 'saved_model'

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def load_model_from_file(model_dir):
    model = tf.saved_model.load(model_dir)
    return model
