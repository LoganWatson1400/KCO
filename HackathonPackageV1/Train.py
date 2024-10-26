# Training Program for NN model
import NN as n
import Loss_Function as lf
import DataProcess as dp
import numpy as np

## Grab Data
data = dp.x
temp = []
for key,val in data:
    if isinstance(val, int):
        temp.append(val)

print(len(temp))
x_train = np.array(temp, dtype=np.float32).reshape(n.BATCH_SIZE,len(temp))
y_train = x_train.astype(np.float32)

date = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
    ]

situationRoot='HackathonPackageV1\DataCache\OptimizerSituations'
## Training Loop (3 epochs = 3 weeks)

autoscore = lf.autoscore_loss(situationRoot, date[0]) #Loss Function

n.model.compile(loss=autoscore, optimizer="adam", metrics=["accuracy"])

n.model.fit(x_train, y_train, batch_size=n.BATCH_SIZE, epochs=n.EPOCHS)
