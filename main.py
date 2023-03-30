import LSTMdiff2D
import tensorflow as tf
from matplotlib import pyplot as plt
import utilities

#Load and process the data
midi_folder = "./JSBChorales"
seed_length = 40
beatDiv = 1
X, y, Xtest, ytest, maxSpace = LSTMdiff2D.getData(midi_folder, beatDiv, seed_length)

#Load pre-trained model
model = tf.keras.models.load_model("JSB_305_BeatDiv1_4000ep.h5")

#Or build your own model
model = LSTMdiff2D.get_model(seed_length, maxSpace)

#train the model. If you loaded existing model, skip to generation step below
history = model.fit(X, y, epochs=4000, verbose=1, batch_size=7124)

#save the trained model
model.save("path/to/file.h5")

#plot the training loss and evaluate
plt.plot(history.history['loss'])
model.evaluate(Xtest, ytest, batch_size=7124)

#generating music
proll = LSTMdiff2D.compose(model=model, seed=Xtest[0], steps=300,
                           sampleLen=seed_length, maxSpace=maxSpace,
                           Hmode='max', Vmode='thresh',
                           temp=3, p=0.5, k=5, n=4,
                           fixConvergence=True)

#saving to MIDI file
utilities.proll2midi(proll, "output.mid", beatDiv)
