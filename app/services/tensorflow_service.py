from pathlib import Path

import numpy as np


def convert_to_saved_model(
    onnx_path: Path,
    output_dir: Path
):

    from onnx2tf import convert

    convert(
        input_onnx_file_path=str(
            onnx_path
        ),
        output_folder_path=str(
            output_dir
        )
    )

    return output_dir


def representative_dataset(tensor_shape):
    shape = tuple(1 if dimension is None else int(dimension) for dimension in tensor_shape)
    sample = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32)
    sample = sample.reshape(shape)

    for _ in range(100):
        yield [sample]


def quantize_to_int8(
    saved_model_dir: Path,
    output_file: Path,
    input_shape=(1, 3, 224, 224),
):

    import tensorflow as tf

    converter = (
        tf.lite.TFLiteConverter
        .from_saved_model(
            str(saved_model_dir)
        )
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    signature = tf.saved_model.load(str(saved_model_dir)).signatures["serving_default"]
    input_specs = list(signature.structured_input_signature[1].values())
    if len(input_specs) != 1 or input_specs[0].shape.rank != 4:
        raise ValueError("INT8 calibration currently requires one 4-D SavedModel input")
    converter.representative_dataset = lambda: representative_dataset(input_specs[0].shape)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = (
        tf.int8
    )

    converter.inference_output_type = (
        tf.int8
    )

    quantized_model = (
        converter.convert()
    )

    with open(
        output_file,
        "wb"
    ) as f:

        f.write(
            quantized_model
        )

    return output_file