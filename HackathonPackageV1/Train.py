# Training Program for NN model
from tensorflow import keras as k
import Loss_Function as lf
import DataProcess as dp
import numpy as np

BATCH_SIZE = 1
EPOCHS = 10

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
predictRoot='HackathonPackageV1\PredDataCache\OptimizerSituations'
####MODEL####

# Initialize Input Layer
model = k.Sequential()
model.add(k.Input(shape=INPUT_SIZE))

# Initialize Hidden Layer
model.add(k.layers.Dense(INPUT_SIZE/2, activation="tanh"))
model.add(k.layers.Dense(INPUT_SIZE/2, activation="tanh"))
model.add(k.layers.Dense(INPUT_SIZE/2, activation="tanh"))

# Initialize Output Layer
model.add(k.layers.Dense(INPUT_SIZE, activation="tanh"))

# OPTIONAL model check
model.summary()

####TRAIN####

## Training Loop (3 epochs = 3 weeks)

autoscore = lf.autoscore_loss(predictRoot, date[0]) #Loss Function

model.compile(loss=autoscore, optimizer="adam", metrics=["accuracy"])

# model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
for i in range(EPOCHS):
    epoch_data = model.train_on_batch(x_train, y_train, return_dict= True)
    predictions = model.predict_on_batch(x_train)
    print(f"\nEpoch: {i+1} --- Loss: {epoch_data['loss']} --- Accuracy: {epoch_data['accuracy']} --- prediction[0][0]: {predictions[0][0]}\n")
    # print(len(predictions[0]))
    # print("\n\n")
    # print(epoch_data)
    # print("\n\n")
    # print(predictions[0][0])
    # print("\n\n")
    dp.updatejson(date[0],predictions)
    weights = model.get_weights()
    model.compile(loss=autoscore, optimizer="adam", metrics=["accuracy"])
    model.set_weights(weights)
#     # # print(len(predictions))
#     # for x in predictions:
#     #     for y in x:
#     #         print(f'{y}')
#     # # print(predictions[0][0])

