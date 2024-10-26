# Training Program for NN model
import NN as n
import Loss_Function as lf
import DataProcess as dp
import numpy as np

date = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
    ]

## Grab Data
data = dp.processData(date[0])
## show data in cvs file
# dp.toCV(data)

x_train = np.array(data, dtype=np.float32).reshape(n.BATCH_SIZE,len(data.columns))
y_train = x_train.astype(np.float32)
situationRoot='HackathonPackageV1\DataCache\OptimizerSituations'
## Training Loop (3 epochs = 3 weeks)

autoscore = lf.autoscore_loss(situationRoot, date[0]) #Loss Function

n.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

n.model.fit(x_train, y_train, batch_size=n.BATCH_SIZE, epochs=n.EPOCHS)
