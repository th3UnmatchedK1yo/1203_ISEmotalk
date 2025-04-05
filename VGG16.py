# %% [markdown]
# ## import lib ##

# %%
# Tensorflow Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence

# Plotting and Model Evaluation Libraries
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Utility Libraries
import numpy as np
import os
import librosa
import librosa.display

# Setting seed for reproducibility
keras.utils.set_random_seed(42)

# %% [markdown]
# ## define spectogram gen func ##

# %%
def wav_to_spectrogram(wav_file, xdim=180, ydim=128):
    # Check if the file is a .wav file
    if not wav_file.endswith('.wav'):
        raise ValueError(f"Expected .wav file, but got: {wav_file}")
    
    # Load audio file
    audio, sr = librosa.load(wav_file, sr=None)  # sr=None keeps the original sample rate
    # Get audio duration in seconds
    duration = librosa.get_duration(y=audio, sr=sr)
    
    # Generate a mel spectrogram (time vs frequency)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=ydim, fmax=8000)
    
    # Convert to decibel scale
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Ensure spectrogram is a numpy array
    spectrogram = np.array(spectrogram)

    # Handling short and long files intelligently
    # Pad shorter files (less than 6 seconds)
    if duration < 6:
        time_frames_needed = xdim  # We want to pad up to xdim time steps
        if spectrogram.shape[1] < time_frames_needed:
            # Pad the time axis to match xdim
            spectrogram = np.pad(spectrogram, ((0, 0), (0, time_frames_needed - spectrogram.shape[1])), mode='constant', constant_values=0)
        elif spectrogram.shape[1] > time_frames_needed:
            # Slice the spectrogram if it exceeds the desired time dimension
            spectrogram = spectrogram[:, :time_frames_needed]
    
    # Slice the spectrogram for longer files (more than 12 seconds)
    elif duration > 12:
        # Choose 10 seconds of audio (as a fixed duration) for consistency
        time_frames_needed = int(10 * sr / 1024)  # 10 seconds, considering hop length
        if spectrogram.shape[1] > time_frames_needed:
            # Slice the spectrogram to match the desired time frames
            spectrogram = spectrogram[:, :time_frames_needed]

    # Ensure spectrogram has the correct size (xdim, ydim)
    # Pad or slice along the frequency axis (ydim)
    if spectrogram.shape[0] < ydim:
        # Pad with zeros if the frequency axis is too short
        spectrogram = np.pad(spectrogram, ((0, ydim - spectrogram.shape[0]), (0, 0)), mode='constant', constant_values=0)
    elif spectrogram.shape[0] > ydim:
        # Slice if the spectrogram exceeds the desired frequency bins
        spectrogram = spectrogram[:ydim, :]
    
    # Pad or slice the time axis (xdim)
    if spectrogram.shape[1] < xdim:
        # Pad with zeros if the time axis is too short
        spectrogram = np.pad(spectrogram, ((0, 0), (0, xdim - spectrogram.shape[1])), mode='constant', constant_values=0)
    elif spectrogram.shape[1] > xdim:
        # Slice if the spectrogram exceeds the desired time steps
        spectrogram = spectrogram[:, :xdim]
    
    # Add a channel dimension by repeating the single channel 3 times (grayscale to RGB)
    spectrogram = np.repeat(spectrogram[..., np.newaxis], 3, axis=-1)
    
    return spectrogram


# %% [markdown]
# ## define dataset class ##

# %%
class SpectrogramDataset(Sequence):
    def __init__(self, dataset_path, category_labels, xdim=180, ydim=128, batch_size=32):
        self.dataset_path = dataset_path
        self.category_labels = category_labels
        self.xdim = xdim
        self.ydim = ydim
        self.batch_size = batch_size
        self.files = []
        self.labels = []
        
        # Gather all files and labels
        for idx, category in enumerate(category_labels):
            category_folder = os.path.join(dataset_path, category)
            for file_name in os.listdir(category_folder):
                if file_name.endswith('.wav'):
                    self.files.append(os.path.join(category_folder, file_name))
                    self.labels.append(idx)
        
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        batch_files = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Load the batch of files and convert them to spectrograms
        batch_spectrograms = np.array([wav_to_spectrogram(file, self.xdim, self.ydim) for file in batch_files])
        return batch_spectrograms, np.array(batch_labels)


# %% [markdown]
# ## data loading ##

# %%
# Path to dataset
dataset_path = "data/VNEMOS"

# Define the 6 emotion categories
category_labels = os.listdir(dataset_path)

# Data Loading
train_dataset = SpectrogramDataset(dataset_path, category_labels, xdim=180, ydim=180, batch_size=32)
validation_dataset = SpectrogramDataset(dataset_path, category_labels, xdim=180, ydim=180, batch_size=32)


# %% [markdown]
# ## define model architecture ## 

# %%
# Model Fine-tuning
conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(180, 180, 3))  # Make sure input_shape matches spectrogram size
conv_base.summary()

# Freeze the layers of VGG16 base
conv_base.trainable = False

# Define the new head for emotion recognition
inputs = keras.Input(shape=(180, 180, 3))
x = keras.applications.vgg16.preprocess_input(inputs)
x = conv_base(inputs)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(len(category_labels), activation="softmax")(x)  # 6 labels
model = keras.Model(inputs, outputs)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])


# %% [markdown]
# ## train the model ## 

# %%
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    verbose=0)

# Plotting training history
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training Set", "Validation Set"]);


# %% [markdown]
# ## fine tune model ##

# %%
# Fine-tuning the model by unfreezing the last 4 layers of VGG16
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=["accuracy"])

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    verbose=0)

# Plotting fine-tuning history
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training Set", "Validation Set"]);


# %% [markdown]
# ## evaluate model ##

# %%
# Model Evaluation
test_dataset = SpectrogramDataset(dataset_path, category_labels, xdim=180, ydim=180, batch_size=32)

# Initialize lists to collect predictions and ground truth
all_predictions = []
all_ground_truth = []

# Loop over the test dataset in batches
for batch_data, batch_labels in test_dataset:
    # Predict for the batch
    batch_predictions = model.predict(batch_data)
    
    # Convert predictions to class indices
    batch_predictions = np.argmax(batch_predictions, axis=1)
    
    # Append the batch predictions and labels to the lists
    all_predictions.extend(batch_predictions)
    all_ground_truth.extend(batch_labels)

# Convert the lists to numpy arrays
all_predictions = np.array(all_predictions)
all_ground_truth = np.array(all_ground_truth)

# Now calculate accuracy
accuracy = accuracy_score(all_ground_truth, all_predictions)
print("Accuracy of the model:", accuracy)

# Confusion Matrix
fig, ax = plt.subplots(figsize=(12,8))
conf_matrix = confusion_matrix(all_ground_truth, all_predictions)
ConfusionMatrixDisplay(conf_matrix, display_labels=category_labels).plot(ax=ax)

# Classification Report
report = classification_report(all_ground_truth, all_predictions)
print('\nClassification Report:\n', report)



