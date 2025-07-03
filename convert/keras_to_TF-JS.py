import tensorflowjs as tfjs
import keras


def convert_h5_to_tfjs(h5_model_path, tfjs_output_dir):
    """
    Convert a Keras HDF5 model to TensorFlow.js format.

    Args:
        h5_model_path (str): Path to your .h5 model file.
        tfjs_output_dir (str): Output directory for TF.js model.
    """
    # Load the Keras model
    model = keras.models.load_model(h5_model_path)

    # Export to TensorFlow.js format
    tfjs.converters.save_keras_model(model, tfjs_output_dir)
    print(f"Model saved to {tfjs_output_dir}")


# Example usage
if __name__ == "__main__":
    h5_model_path = "convert/mobilenetv4_dance_classifier.h5"
    tfjs_output_dir = "convert/mobilenetv4_tfjs"
    convert_h5_to_tfjs(h5_model_path, tfjs_output_dir)