import theano
from keras.models import load_model

# Ensure you have set Theano as the backend in the keras.json configuration file

# Load the model
model = load_model('/home/nurupo/Desktop/dev/ASAM-master/software/_tmp_weights/WSJ0_left_00040.h5')

# Print the model architecture
model.summary()

# Analyze the weights
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}")
        print(f"Weights shape: {weights[0].shape}")
        # Additional analysis can be done here
