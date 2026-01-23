from pneunomia import PneumoniaModel

model = PneumoniaModel("grad_model.keras")
predicted = model.pipeline("image.png", "output.png")

print("PNEUMONIA" if predicted > 0.5 else "NORMAL")