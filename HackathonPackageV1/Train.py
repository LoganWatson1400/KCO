# Training Program for NN model
from tensorflow import keras as k
import Loss_Function as lf
import DataProcess as dp
import numpy as np

BATCH_SIZE = 1
EPOCHS = 100

date = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
    ]

## Grab Data
data = dp.processData(date[0])
INPUT_SIZE = len(data.columns) # Number of inputs IE initial POs, inventory, etc
## show data in cvs file
dp.toCV(data)

x_train = np.array(data, dtype=np.float32).reshape(BATCH_SIZE,len(data.columns))
y_train = x_train.astype(np.float32)
situationRoot='HackathonPackageV1\DataCache\OptimizerSituations'

####MODEL####

# Initialize Input Layer
model = k.Sequential()
model.add(k.Input(shape=INPUT_SIZE))

# Initialize Hidden Layer
model.add(k.layers.Dense(INPUT_SIZE/2, activation="relu"))
model.add(k.layers.Dense(INPUT_SIZE/2, activation="relu"))
model.add(k.layers.Dense(INPUT_SIZE/2, activation="relu"))

# Initialize Output Layer
model.add(k.layers.Dense(INPUT_SIZE, activation="relu"))

# OPTIONAL model check
model.summary()

####TRAIN####

## Training Loop (3 epochs = 3 weeks)

autoscore = lf.autoscore_loss(situationRoot, date[0]) #Loss Function

n.model.compile(loss=autoscore, optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
