# Training Program for NN model
import NN as n


## Grab Data

## Load data
x_train = None #PlaceHolder
y_train = None #PlaceHolder

## Training Loop (3 epochs = 3 weeks)
n.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

n.model.fit(x_train, y_train, batch_size=n.BATCH_SIZE, epochs=n.EPOCHS, validation_split=0.1)
