#Differential Music  
This ML model can compose music by learning from MIDI files. The output is also a MIDI file.
The data encoding is based on intervals between notes in the time and pitch space,
hence the name "Differential Music". For details on the encoding as well as model specifics, see my [paper](https://arxiv.org/ftp/arxiv/papers/2108/2108.10449.pdf).  

You can listen to a music piece created by this AI tool at [this link](https://soundcloud.com/hooman-rafraf/fitting-music-relatively-classical).

You need MIDI files for using this model. If you train the model yourself the model will split the MIDIs with 80-20 ratio for training and testing.  

**Note:** Even if you use the pretrained model you still need MIDI files because in the generation phase, the model needs
a prompt that it will then continue adding its own notes to. Put the MIDI files in a folder
with nothing else inside. This repository contains the ***JSBChorales*** dataset introduced in [this](https://arxiv.org/ftp/arxiv/papers/1206/1206.6392.pdf) paper.
This is also the dataset that was used for the pretrained model. It is a collection of 382 chorales by
J.S. Bach.

The MIDI parsing and MIDI writing is done using [MidiToolkit](https://github.com/YatingMusic/miditoolkit)

##Getting Started
This project has been run and tested on Python 3.7 with the package versions listed in
the _**requirements.txt**_ file. If you want to train the model yourself, GPU support is recommended.
If your system has GPU support and the dependencies are installed correctly, the code will
use GPU without any need for code modification.



Install the project dependencies:

    pip3 install -r requirements.txt

##Preparing the Data
Load the data using the code below. This will parse the MIDI files and perform the pre-processing to encode them
for feeding into the model. For details of the encoding see the [paper](https://arxiv.org/ftp/arxiv/papers/2108/2108.10449.pdf).

```Python
import LSTMdiff2D
import tensorflow as tf
from matplotlib import pyplot as plt
import utilities

midi_folder = "./JSBChorales"
seed_length = 40
beatDiv = 4
X, y, Xtest, ytest, maxSpace = LSTMdiff2D.getData(midi_folder, beatDiv, seed_length)
```

You can modify ***midi_folder*** to refer to your own folder containing MIDI files.
***beatDiv*** technically specifies the number of timesteps per MIDI beat. 4 is a good number for most cases
and can encode with the resolution of 16<sup>th</sup> notes. If your music is very fast you can increase this number.
For example if the music contains a lot of 32<sup>nd</sup> notes, set ***beatDiv = 8***.

***seed_length*** specifies the number of timesteps for the prompt which will also affect the LSTM input shape.
40 proved to be a good working number but feel free to experiment with it.

##Preparing the Model 
There are two ways to use this tool, you can either use the pretrained model or 
train your own. The pretrained model is included in the repository as ***JSB_305_BeatDiv1_4000ep.h5*** which is standalone Tensorflow saved model format. Follow one of the two sections below depending on your choice. 
###Loading pretrained model

```python
model = tf.keras.models.load_model("JSB_305_BeatDiv1_4000ep.h5")
```

###Training your own model

```python
model = LSTMdiff2D.get_model(seed_length, maxSpace)
history = model.fit(X, y, epochs=2000, verbose=1, batch_size=4693)
model.save('path/to/file.h5')   #if you wish to save the model for later use
plt.plot(history.history['loss'])   #to plot the training loss
model.evaluate(Xtest, ytest, batch_size=1335)   #to evaluate the model
```

This will build and train the model using the provided MIDI dataset. For model details
you can see the papar.

##Generating music

```python
proll = LSTMdiff2D.compose(model=model, seed=Xtest[0], steps=1000, 
                           sampleLen=seed_length, maxSpace=maxSpace, 
                           Hmode='max', Vmode='thresh', 
                           temp=3, p=0.5, k=5, n=4, 
                           fixConvergence=True)
utilities.proll2midi(proll, "output.mid", beatDiv)
```

The **compose** method has several parameters that let you customize the generation process.  
* ***model:*** this is either the loaded model or the one that you trained.  
* ***seed:*** this is the musical prompt that the model will add notes to. We can use one of the slices from out test set as prompt.  
* ***steps:*** specifies how many timesteps the model should compose.  
* ***sampleLen:*** This is the length of the seed and one of the dimensions of the LSTM input. Do not change this; it must be equal to seed_length.  
* ***maxSpace:*** The largest harmonic interval in the entire training set. Do not change this.  
* ***Hmode*** and ***Vmode:*** The model activates notes in the output based on the predicted probabilities in the output layer of the model.
This system supports several sampling methods that allow for customization of the music creation.
  
  ***Hmode***  specifies the sampling mode for notes of the melody (soprano, i.e. the highest active note at each timestep).  
Possible values for Hmode:  

  - **"max"** activates the note with the highest probability in the model's output.  
  - **"temp"** uses temperature sampling. Temperature should be a value greater than 0. If temperature is 1, this means a note will be activated based on the probability distribution in the model's output. As temperature gets larger, the probability distribution gets closer to uniform. As temperature gets smaller and approaches zero, the model's behavior becomes more deterministic, becoming more likely to pick the note with the highest probability. A very small value for temperature is almost equivalent to setting Hmode to "max"  
    
  ***Vmode*** determines the sampling method in vertical space (harmony, i.e. chord notes that accompany the soprano)  
  Possible values for Vmode:  
  -  **"thresh"** activates any note that has the probability greater than ***p***.  
  -  **"topK"** takes the top ***k*** highest-probability notes and randomly activates ***n*** of them, but activation only happens if the probability is greater than ***p***.  

The **proll2midi** method will decode the result and convert it to actual MIDI file and saves it to disk. You can modify the second parameter to the path and file name you wish.