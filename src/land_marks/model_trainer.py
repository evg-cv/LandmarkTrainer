import dlib

from settings import PROFILE_MOUTH_MODEL, PROFILE_FACE_XML


def train_model(name, xml):
    """requires: the model name, and the path to the xml annotations.
    It trains and saves a new model according to the specified
    training options and given annotations"""
    # get the training options
    options = dlib.shape_predictor_training_options()
    options.tree_depth = 8
    options.nu = 0.3
    options.cascade_depth = 18
    options.feature_pool_size = 400
    options.num_test_splits = 50
    options.oversampling_amount = 5
    #
    options.be_verbose = True  # tells what is happening during the training
    options.num_threads = 4  # number of the threads used to train the model

    # finally, train the model
    dlib.train_shape_predictor(xml, name, options)


def measure_model_error(model, xml_annotations):
    """requires: the model and xml path.
    It measures the error of the model on the given
    xml file of annotations."""
    error = dlib.test_shape_predictor(xml_annotations, model)
    print("Error of the model: {} is {}".format(model, error))


if __name__ == '__main__':

    train_model(name=PROFILE_MOUTH_MODEL, xml=PROFILE_FACE_XML)
