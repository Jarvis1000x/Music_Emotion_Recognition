import librosa
import numpy as np
import os


class Loader:
    # loader is responsible for loading the audio file

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal


class Padder:
    # Padder is responsible to apply padding to an array

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    # LogSpectrogramExtractor extracts log spectrogram (in dB) from a time series signal

    def __init__(self, frame_size, hop_length):
        self.frame_rate = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_rate,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class Saver:
    # Saver is responsible to save features, and the min max values

    def __init__(self, feature_save_dir):
        self.feature_save_dir = feature_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        with open(save_path, 'wb') as f:
            np.save(f, feature, allow_pickle=True)
        return save_path

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


class PreprocessingPipeline:
    """
    PreprocessingPipeline processes audio files in a directory,
    applying the following to each file
        1 - load a file
        2 - pad the signal (if necessary)
        3 - extracting log spectrogram from signal
        4 - normalise spectrogram
        5 - save the normalised signal
    storing all the min max values for all the log spectrogram
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(self.loader.sample_rate * self.loader.duration)

    def process(self, audio_files_directory):
        for root, _, files in os.walk(audio_files_directory):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        save_path = self.saver.save_feature(feature, file_path)

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 15 # In seconds
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAM_SAVE_DIR = "data/spectrograms/"
    FILES_DIR = "data/audio/"

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    saver = Saver(SPECTROGRAM_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)