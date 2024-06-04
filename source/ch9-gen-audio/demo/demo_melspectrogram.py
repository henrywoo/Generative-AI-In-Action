import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

wav_file_path = '/home/wukong/Music/GAN_final.wav'


def show_signals(f):
    y, sr = librosa.load(f)
    with plt.style.context('ggplot'):
        plt.plot(y)
        plt.title('Signal')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig('signals.png')
        plt.show()
        return y, sr

def show_spectrum(y):
    n_fft = 2048
    ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft + 1))
    with plt.style.context('ggplot'):
        plt.plot(ft)
        plt.title('Spectrum')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig('spectra.png')
        plt.show()

def show_spectrogram(y, sr):
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    with plt.style.context('ggplot'):
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Hz');
        plt.tight_layout()
        plt.savefig('spectrogram.png')
        plt.show()
    return spec


def show_mel_spectrogram(y, sr, spect):
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

    with plt.style.context('ggplot'):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spect_db, y_axis='mel', fmax=8000, x_axis='time')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    y, sr = show_signals(wav_file_path)
    show_spectrum(y)
    spect = show_spectrogram(y, sr)
    show_mel_spectrogram(y, sr, spect)

