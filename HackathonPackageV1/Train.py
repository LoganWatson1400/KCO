import NN as n


batch_size = 128
epochs = 15


n.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

n.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
