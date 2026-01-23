from pneunomia import PneumoniaModel
import os

model = PneumoniaModel("grad_model.keras")

for dirpath, _, filenames in os.walk("sample_images"):
    for filename in filenames:
        input_image_path = os.path.join(dirpath, filename)
        output_image_path = os.path.relpath(input_image_path, "sample_images")
        output_image_path = os.path.join("sample_outputs", output_image_path)

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        model.pipeline(input_image_path, output_image_path)