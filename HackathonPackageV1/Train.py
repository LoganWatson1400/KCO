# Training Program for NN model
import NN as n
import Loss_Function as lf

## Grab Data

## Load data
x_train = None #PlaceHolder
y_train = None #PlaceHolder

date = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
    ]

situationRoot='HackathonPackageV1\DataCache\OptimizerSituations'
## Training Loop (3 epochs = 3 weeks)

autoscore = lf.autoscore_loss(situationRoot, date[0]) #Loss Function

n.model.compile(loss=autoscore, optimizer="adam", metrics=["accuracy"])

n.model.fit(x_train, y_train, batch_size=n.BATCH_SIZE, epochs=n.EPOCHS, validation_split=0.1)
