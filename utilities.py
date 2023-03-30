#opens a pianoroll collection stored in a .mat file created by the corresponding Matlab function
#and converts it to a 3D list (#of midis, #of pitches, #number of timesteps)
#and saves it (if save is True) to a .pickle file
import numpy as np
import easygui
import scipy.io
from scipy.io import loadmat
import pickle
from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
from miditoolkit.midi import containers as ct
import os

def midis2Sparses(path, beatDiv):
    sparses = []
    for filename in os.listdir(path):
        midi = mid_parser.MidiFile(os.path.join(path, filename))
        tracks = []
        for inst in midi.instruments:
            tracks.append(pr_parser.notes2pianoroll(inst.notes))
        max_ticks = max([len(tr) for tr in tracks])

        pianoroll = np.zeros((max_ticks, 128), dtype='int8')
        for tr in tracks:
            pianoroll[np.pad(tr, ((0, max_ticks - tr.shape[0]), (0, 0))) > 0] = 1

        hop_size = int(midi.ticks_per_beat / beatDiv)  # ticks per timestep
        proll = pianoroll[:-1:hop_size, 21:-19]
        sparses.append([np.flip(np.nonzero(proll[i])[0]).tolist() for i in range(len(proll))])
    return sparses

def proll2midi(proll, filename, beatDiv):
    mido_obj = mid_parser.MidiFile()
    ticks_per_timestep = int(mido_obj.ticks_per_beat/beatDiv)
    track = ct.Instrument(program=0, is_drum=False, name='example track')
    mido_obj.instruments = [track]
    proll = proll.T
    for pitch in range(proll.shape[0]):
        on = False
        for timestep in range(proll.shape[1]):
            if (proll[pitch, timestep] == 1) and ((not on) or (timestep == proll.shape[1] - 1)):
                start = timestep*ticks_per_timestep
                on = True
            if proll[pitch, timestep] == 0 and on:
                end = timestep*ticks_per_timestep
                on = False
                note = ct.Note(start=start, end=end, pitch=pitch+21, velocity=100)
                mido_obj.instruments[0].notes.append(note)
    mido_obj.dump(filename)

#reads the .mat file created with Matlab and converts to a 3D list, and returns the list.
#if save is true, saves it on disk as pickle.
def mat2Py(save):
    sourcePath = easygui.fileopenbox()
    mat = loadmat(sourcePath)
    what = mat[list(mat.keys())[3]]
    what = [i.tolist() for i in what.squeeze().tolist()]

    if save:
        destPath = easygui.filesavebox()
        with open(destPath, "wb") as fp:   #Pickling
            pickle.dump(what, fp)

    return what

#gets a 3D list of onehot encodngs and keeps only the highest pitch at each timestep.
#might be better to use the sparse encoding instead (keep only the first active pitch at each timestep)
def toMonophonic(midis, save):
    maxlen = max([len(i[0]) for i in midis])
    pitchrange = len(midis[0])
    for i in range(len(midis)):
        for timestep in range(maxlen):
            for pitch in range(pitchrange-1, 0, -1):
                if midis[i][pitch][timestep] == 1:
                    for j in range(pitch - 1, 0, -1):
                        midis[i][j][timestep] = 0
                    break
    if save:
        scipy.io.savemat(easygui.filesavebox(), mdict={'proll': midis[0]})
    return midis

#reads a pickle file and returns the value
def readpkl():
    sourcePath = easygui.fileopenbox()
    with open(sourcePath, 'rb') as f:
        x = pickle.load(f)
    return x

#saves a onehot encoded midi as .mat file
def py2Mat(midi, save2Desktop):
    if save2Desktop: path = 'c:\\users\\hooman\\desktop\\pak.mat'
    else: path = easygui.filesavebox()
    scipy.io.savemat(path, mdict={'proll': midi})


#gets a onehot pianoroll and converts to sparse encoding. each row will be the nonzero indices of a timestep.
def proll2Sparse(midi):
    return np.array([np.nonzero(ts) for ts in np.transpose(midi)])

#gets a sparse encoded 3D list and converts it to a piano roll onehot. ignores the '0's in the sparse values.
def sparse2Proll(sp):
    proll = np.zeros((len(sp), 88), dtype='uint8')
    for i in range(len(sp)):
        proll[i][[n for n in sp[i] if n > 0]] = 1
    return proll.T.tolist()

#assuming max leap is 11 (less than octave), we will have 23 nodes. 11 upward motion, 11 downward, and 1 for no movement
#the origin is the 12th node (no movement)
def sparse2HDroll(sp):
    droll = np.zeros((len(sp), 23), dtype='uint8')
    for i in range(len(sp)):
        droll[i][[n+11 for n in sp[i]]] = 1
    return droll.T.tolist()

def sparse2VDroll(sp, maxSpace):
    droll = np.zeros((len(sp), maxSpace), dtype='uint8')
    for i in range(len(sp)):
        if sp[i][0] > 0:
            droll[i][np.array(sp[i])-1] = 1
    return droll.tolist()

#gets a list of onehot pianorolls (will be a 3D list) and converts to a list of their sparse encodings.
def prolls2Sparses(midis):
    allSparses = []
    for midi in midis:
        allSparses.append(proll2Sparse(midi))
    return allSparses

#gets a list of sparse-encoded pianorolls and saves them on disk. you specify the name, and it will add the indices
#to the file names. Saves separate files for each spsarse-encoded pianoroll.
def saveSparses(sparses):
    destPath = easygui.filesavebox()
    for i in range(len(sparses)):
        np.save(destPath+str(i), sparses[i])

#loads the sparse-encoded pianoeolls from disk (the ones saved by saveSparses function)
#AND converts them to 3D lists. result is a set of sparses, each of which is a set oftimesteps,
#each of which is a set of active pitches
def loadSparses():
    sparses = []
    paths = easygui.fileopenbox(multiple=True)
    for path in paths:
        sparses.append(np.load(path, allow_pickle=True))
    sparses = [sparse.tolist() for sparse in sparses]
    return [[ts[0].tolist() for ts in sp] for sp in sparses]

#gets the list of sparse encoded midis and slices up each midi to chunks of length l (disposing the last chunk
#in each midi because it is a remainder and most likely has smaller size than l)
#and combines these chunks into a single long list X. X is still a 3D list like "sparses" but with
#small midi chunks instead of full midis. for each sample in X, the corresponding target in y is the immediate
#vector of pitches after the end of the sample. chunks can overlap. d determines the distance between chunks
def getSamples(sparses, l, d, dataType):
    X=[]; y=[]
    for sparse in sparses:
        i = 0
        while i+l < len(sparse):
            X.append(sparse[i:i+l])
            y.append(sparse[i+l])
            i += d
    return np.array(X, dtype=dataType), np.array(y, dtype=dataType)

#gets the list of sparse-encoded midis and zeropads/truncates the pitches in each timestep so that the size
#of the list of pitches in each timestep is n. we can't have variable length vectors for different timesteps of LSTM.
#before truncating/padding, the timesteps are sorted so that the highest pitches are prioritized for keeping.
def trp(sparses, n):
    for sparse in sparses:
        for ts in sparse:
            ts.sort(reverse=True)
    sparses = [[p[:n] + [0]*(n-len(p)) for p in sparse] for sparse in sparses]
    return sparses

#samples from a vector of values based on a temperature and returns the index. temp.=inf is equivalent to uniform distribution
#temp.=1 is equivalent to exact distribution as determined by the vector element values
#temp. = small makes it more likely that the element with the maximum value is picked
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


#lenghths = [len(n) for sparse in sparses for n in sparse]
#max([len(n) for n in sparses[i]])
#max([max([len(n) for n in sparse]) for sparse in sparses])
#sparses = loadSparses()
#mat2Py(False)