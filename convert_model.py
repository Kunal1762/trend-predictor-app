from tensorflow.keras.models import load_model

model = load_model("keras_model.h5")

model.save("keras_model.keras")

print("Model converted successfully!")