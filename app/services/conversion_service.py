
from app.services.job_service import update_job
from app.services.onnx_service import export_to_onnx
from app.services.onnx_validator import validate_onnx
from app.services.pytorch_service import load_pytorch_model
from app.services.report_service import generate_report
from app.services.tensorflow_service import convert_to_saved_model, quantize_to_int8


def run_conversion(
    job_id,
    job_folder,
    model_path,
    architecture,
    input_shape=(1, 3, 224, 224),
):

    try:

        update_job(
            job_id,
            "exporting_onnx",
            20
        )

        onnx_path = (
            job_folder /
            "model.onnx"
        )

        model = load_pytorch_model(model_path, architecture)
        export_to_onnx(
            model_path,
            architecture,
            onnx_path,
            input_shape,
        )

        update_job(
            job_id,
            "validating_onnx",
            40
        )

        validation = validate_onnx(onnx_path, model, input_shape)

        update_job(
            job_id,
            "converting_tensorflow",
            60
        )

        saved_model_dir = (
            job_folder /
            "saved_model"
        )

        convert_to_saved_model(
            onnx_path,
            saved_model_dir
        )

        update_job(
            job_id,
            "quantizing_int8",
            80
        )

        tflite_path = (
            job_folder /
            "model_int8.tflite"
        )

        quantize_to_int8(saved_model_dir, tflite_path, input_shape)

        report_path = (
            job_folder /
            "report.json"
        )

        generate_report(
            report_path,
            model_path,
            onnx_path,
            tflite_path,
            validation,
        )

        update_job(
            job_id,
            "completed",
            100
        )

    except Exception as exc:  # noqa: BLE001

        update_job(
            job_id,
            f"failed: {exc!s}",
            0
        )