import os
import random
from itertools import count
import numpy as np
import librosa
import librosa.display
import pywt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib
from tqdm import tnrange, tqdm_notebook, tqdm
import matplotlib.pyplot as plt
import soundfile as sf

def plot_mel_spectrogram(S_db):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='LogMelSpectogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

def get_wav_files(dict='K:/LT/recordings/'):
    original_wavfiles = []

    for path, dirs, files in os.walk(dict, topdown=True):
        for file in files:
            if file.endswith('.wav'):
                original_wavfiles.append(os.path.join(path, file))
    print("Number of wavfiles: ", len(original_wavfiles))

    return original_wavfiles

def create_augmented_data(augmented_dir='K:/LT/augmented_recordings/',default_sample_rate = 8000,original_wavfiles=None):
    os.makedirs(augmented_dir, exist_ok=True)
    id = 1000
    for i in range(16):
        for file_path in tqdm(original_wavfiles):
            y, Fs = librosa.load(file_path)
            y_new = y
            k = random.randint(0, 1)
            if k==1:
                y_new = librosa.effects.pitch_shift(y_new, sr=Fs, n_steps=random.uniform(-2, 2))
            k = random.randint(0, 1)
            if k==1:
                speed_change = random.uniform(0.8, 1.2)
                tmp = librosa.effects.time_stretch(y_new, rate=speed_change)
                minlen = min(y_new.shape[0], tmp.shape[0])
                y_new *= 0
                y_new[0:minlen] = tmp[0:minlen]
            k = random.randint(0, 1)
            if k==1:
                noise_amp = 0.005 * np.random.uniform() * np.amax(y_new)
                y_new = y_new + noise_amp * np.random.normal(size=y_new.shape[0])
            #k = random.randint(0, 1)
            #if k==1:
                #y_hpss = librosa.effects.hpss(y_new)
               # y_new = y_hpss[1]

            basename = '_'.join([os.path.splitext(os.path.basename(file_path))[0],str(id+1),str(id+2)])+'.wav'
            output_file = os.path.join(augmented_dir, basename)
            sf.write(output_file, y_new, default_sample_rate)
            id += 1

def plot_duration_distribution(dataset):
    duration = []
    i = 0
    for wav_file in dataset:
        duration.append(librosa.get_duration(path=wav_file))
    # matplotlib histogram
    plt.hist(duration, color='blue', edgecolor='black',bins=100)
    plt.show()

def trim_wav_files(dataset,max_db=10,trimmed_dir='K:/LT/trimmed_recordings/',default_sample_rate=8000):
    for wav_file in dataset:
        wav, sr = librosa.load(wav_file)
        untrimmed_duration = librosa.get_duration(path=wav_file)
        wav, index = librosa.effects.trim(wav, top_db=max_db)
        trimmed_duration = librosa.get_duration(y=wav)
        print(str(wav_file)+" Untrimmed: "+str(untrimmed_duration)+" Trimmed: "+str(trimmed_duration))
        basename = '_'.join([os.path.splitext(os.path.basename(wav_file))[0], str('trimmed')]) + '.wav'
        output_file = os.path.join(trimmed_dir, basename)
        sf.write(output_file, wav, default_sample_rate)

def pad_wav_files(dataset, output_dir, max_length,default_sample_rate=8000):
    for wav_file in dataset:
        samples, Fs = librosa.load(wav_file)
        #librosa.display.waveshow(samples, sr=default_sample_rate,color='blue', alpha=0.25)
        #plt.title('Original Audio File')
        #plt.show()
        short_samples = librosa.util.fix_length(samples, size=int(max_length * Fs))
        #librosa.display.waveshow(short_samples, sr=default_sample_rate,color='blue', alpha=0.25)
        #plt.title('Padded Audio File')
        #plt.show()
        basename = '_'.join([os.path.splitext(os.path.basename(wav_file))[0], str('padded')]) + '.wav'
        output_file = os.path.join(output_dir, basename)
        sf.write(output_file, short_samples, default_sample_rate)


def cache_spectrogram_features(dataset, output_dir):
    for wav_file in dataset:
        samples, Fs = librosa.load(wav_file)

        melSpectrum = librosa.feature.melspectrogram(y=samples.astype(np.float16), sr=Fs, n_mels=20)

        logMelSpectrogram = librosa.power_to_db(melSpectrum, ref=np.max)
        plot_mel_spectrogram(logMelSpectrogram)
        basename = os.path.splitext(os.path.basename(wav_file))[0] + '.npy'
        output_file = os.path.join(output_dir, basename)
        np.save(output_file, logMelSpectrogram)

def create_dataset(root='K:/LT/mel_spectrograms/'):
    npyfiles_data = []
    labels = []
    for path, dirs, files in os.walk(root, topdown=True):
        for file in files:
            if file.endswith('.npy'):
                labels.append(int(file[0]))
                data = np.load(root+file)
                scaler = StandardScaler()
                npyfiles_data.append(scaler.fit_transform(data))
    np.save('labels_single.npy',labels)
    np.save('data_single.npy',npyfiles_data)

#todo
#shuffle and save the data
#fix split test,val and train sets
#augment the train set
#trimm and pad files
#save result as numpy array
#mel and normalization move to DataPreprocessing file
from random import shuffle
Resampling = 16000
original = get_wav_files()
original = random.sample(original, len(original))
test = np.array(original[0:600])
val = np.array(original[600:1200])
train = np.array(original[1200:3000])
np.save('test', test)
np.save('train', train)
np.save('val', val)
#exit(1)

#original = np.load("train.npy").tolist()
#print(len(original))


#create_augmented_data(original_wavfiles=original,default_sample_rate=Resampling)
#augmented = get_wav_files("K:/LT/augmented_recordings/")
#dataset = original + augmented

#plot_duration_distribution(original)
#trim_wav_files(original,default_sample_rate=Resampling)

#trimmed_dataset = get_wav_files('K:/LT/trimmed_recordings/')
#plot_duration_distribution(trimmed_dataset)

#pad_wav_files(trimmed_dataset,"K:/LT/padded_recordings/",0.8,default_sample_rate=Resampling)
padded_dataset = get_wav_files("K:/LT/padded_recordings/")
#plot_duration_distribution(padded_dataset)

cache_spectrogram_features(padded_dataset,'K:/LT/mel_spectrograms/')

create_dataset()