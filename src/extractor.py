import librosa
import numpy as np

def extract_features(file_path):
    """
    Loads an audio file and returns/extracts a mean MFCC vector.
    """
    try:
        # Load the audio file (only first 30 seconds for consistency)
        y, sr = librosa.load(file_path, duration=30)
        
        # Extract 13 MFCCs (a standard number for audio representation) timbre
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1) #Chroma (Harmony)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)#Spectral Contrast (Dynamics)

    
        return np.concatenate([mfcc, chroma, contrast]) #Concatenate them into one long "Fingerprint"

        # #for mfcc by itself
        # # Take the mean across the time axis. 
        # # This gives us a fixed-length vector regardless of the song's length.
        # mfcc_processed = np.mean(mfcc, axis=1)
        # return mfcc_processed

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None