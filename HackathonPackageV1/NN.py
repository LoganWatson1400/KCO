from tensorflow import keras as k

INPUT_SIZE = 3 # Number of inputs IE initial POs, inventory, etc

# Initialize Input Layer
model = k.Sequential()
model.add(k.Input(shape=INPUT_SIZE))

# Initialize Hidden Layer
model.add(k.layers.Dense(INPUT_SIZE*2, activation="relu"))
model.add(k.layers.Dense(INPUT_SIZE*2, activation="relu"))
model.add(k.layers.Dense(INPUT_SIZE*2, activation="relu"))

# Initialize Output Layer


# OPTIONAL model check
model.summary()