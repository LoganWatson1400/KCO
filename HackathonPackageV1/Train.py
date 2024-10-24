import NN as n
import pandas as pd
from tensorflow import keras as k


batch_size = 128
epochs = 15
situationPath = 'HackathonPackageV1\DataCache\OptimizerSituations\2024-09-06 Week 1\planningSchedule.json'
planningSchedule=pd.read_json(situationPath)
print(planningSchedule)

print(k.datasets)
# (x_train, y_train), (x_test, y_test) = k.datasets

# n.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# n.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
