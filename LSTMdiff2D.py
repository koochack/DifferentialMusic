import utilities
import tensorflow as tf
import random
import numpy as np
import math
from math import fmod

def getData(path, beatDiv, sampleLen):
    #sparses = utilities.loadSparses()
    sparses = utilities.midis2Sparses(path, beatDiv)

    maxVoices = 4
    sampleDistances = 1

    sparsesTRP = utilities.trp(sparses, maxVoices)
    sparsesTRP = [[ts for ts in sparse if len(np.nonzero(ts)[0]) > 0] for sparse in sparsesTRP]#remove silent timesteps
    melodySparses = [[[ts[0]] for ts in sparse] for sparse in sparsesTRP]
    for sparse in range(len(sparsesTRP)):#change padded zeros to min of the ts
        for ts in range(len(sparsesTRP[sparse])):
            temp = np.array(sparsesTRP[sparse][ts])
            temp[temp==0] = min(temp[temp.nonzero()])
            sparsesTRP[sparse][ts] = temp.tolist()

    #vdiffs can contain all-zero timesteps. these are timesteps that don't have chords. just one note
    vdiffs = [[np.abs((np.array(ts)-ts[0])[1:]).tolist() for ts in sparse]for sparse in sparsesTRP]#C-E-G becomes [3,7]
    maxSpace = max([max([max(ts) for ts in sparse]) for sparse in vdiffs])#max interval in all chords
    vdrolls = [utilities.sparse2VDroll(diff, maxSpace) for diff in vdiffs]

    hdiffs = []
    for sparse in melodySparses:
        temp = np.diff(np.array(sparse).reshape((len(sparse))))
        hdiffs.append(temp.reshape((len(sparse)-1,1)).tolist())

    #bring large leaps within octave. insert a [0] at the beginning of each sequence so 'hdiffs' timesteps
    #will match the correct 'vdiffs' timesteps
    hdiffs = [[[0]]+[[int(fmod(x[0],12))] for x in midi] for midi in hdiffs]

    hdrolls = [np.array(utilities.sparse2HDroll(diff)).T.tolist() for diff in hdiffs]

    combinedDrolls = [[hdrolls[midi][ts]+vdrolls[midi][ts] for ts in range(len(hdrolls[midi]))] for midi in range(len(hdrolls))]

    split = math.ceil(len(combinedDrolls)*0.8)
    [X, y] = utilities.getSamples(combinedDrolls[:split], sampleLen, sampleDistances, 'uint8')
    [Xtest, ytest] = utilities.getSamples(combinedDrolls[split:], sampleLen, sampleDistances, 'uint8')
    return X, y, Xtest, ytest, maxSpace

def get_model(sampleLen, maxSpace):
    input = tf.keras.Input(shape=(sampleLen, maxSpace+23))
    x = tf.keras.layers.LSTM(300, activation="tanh", return_sequences=True)(input)
    x = tf.keras.layers.LSTM(200, activation="tanh")(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    HDoutput = tf.keras.layers.Dense(23, activation="softmax")(x)
    VDoutput = tf.keras.layers.Dense(maxSpace, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=input, outputs=tf.keras.layers.concatenate([HDoutput, VDoutput]))
    model.compile(optimizer='adam', loss='bce')
    return model


def compose(model, seed, steps, sampleLen, maxSpace, Hmode, Vmode, temp, p, k, n, fixConvergence):
    combinedDroll = np.array(continueSeqOnehotDiffs2D(model, seed, steps, sampleLen, maxSpace, Hmode, Vmode, temp, p, k, n, fixConvergence), dtype='uint8')
    startNote = 74
    hdroll = combinedDroll[:, :23]
    vdroll = combinedDroll[:, 23:]
    vdroll = np.array([list(reversed(ts)) for ts in vdroll])
    hdparse = np.array([np.nonzero(ts)[0]-11 for ts in hdroll]).transpose().reshape((len(hdroll),1))
    hsparse = [[x] for x in np.r_[startNote, [ts[0] for ts in hdparse]].cumsum()]
    hsparse = bringInRange(hsparse, vdroll)
    proll = utilities.sparse2Proll(hsparse)
    proll = np.array(proll).T
    for i in range(len(combinedDroll)):
        startNote = hsparse[i+1][0]
        proll[i+1, max(startNote-maxSpace, 0):startNote] = vdroll[i, -min(startNote, maxSpace):]
    return proll

def continueSeqOnehotDiffs2D(model, seed, steps, sampleLen, maxSpace, Hmode = 'max', Vmode = 'thresh', temp=1, p = 0.5, k = 4, n = 4, fixConvergence = True):
    seq = np.zeros((sampleLen+steps, maxSpace+23), dtype='uint8')
    seq[0:len(seed), 0:len(seed[0])] = seed.copy()
    for i in range(steps):
        pred = model.predict(seq[i:i+sampleLen,:].reshape((1,sampleLen,maxSpace+23)))[0]

        #pick the max for the melody difference part
        if Hmode == 'max':
            seq[i+sampleLen, np.argmax(pred[:23])] = 1##########

        #temperature sampling for melody
        if Hmode == 'temp':
            seq[i+sampleLen, utilities.sample(pred[:23] - min(pred[:23]), temp)] = 1###########

        #use threshold activation for the chord part
        if Vmode == 'thresh':
            seq[i+sampleLen, np.concatenate((np.array([False]*23), (pred[23:] > p)))] = 1#######

        #top-K sampling: pick random n from top k, but only activate if threshold is > p
        if Vmode == 'topK':
            seq[i + sampleLen, [x + 23 for x in (np.random.choice(pred[23:].argsort()[-k:][::-1], n, False)) if pred[23:][x] > p]] = 1

        if fixConvergence:
            seq[i+sampleLen, 23:][random.sample(range(maxSpace), 10)] = 0
    #fill the gaps
    if fixConvergence:
        still = True
        while still:
            still = False
            for i in range(maxSpace):
                for j in range(sampleLen+steps-1):
                    if seq[j-1][23:][i] == 1 and seq [j][23:][i] == 0 and seq[j+1][23:][i] == 1:
                        seq[j][23:][i] = 1
                        still = True

    return seq.tolist()

#if notes start to exceed the standard range, increase or decrease pitch by 2 octaves
def bringInRange(hsparse, vdroll):
    hsparse = np.array(hsparse)
    for i in range(len(hsparse)-1):
        if hsparse[i+1][0] > 87: hsparse[i+1:] -= 24
        space = 0
        if len(np.nonzero(vdroll[i])[0]) > 0: space = max(np.nonzero(vdroll[i])[0])
        if hsparse[i+1][0] - space < 1: hsparse[i+1:] += 24
    return hsparse.tolist()
