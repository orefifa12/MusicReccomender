import librosa
import numpy as np

def extract_features(file_path):
    """
    Loads an audio file and extracts a mean MFCC vector.
    """
    try:
        # Load the audio file (only first 30 seconds for consistency)
        y, sr = librosa.load(file_path, duration=30)
        
        # Extract 13 MFCCs (a standard number for audio representation)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Take the mean across the time axis. 
        # This gives us a fixed-length vector regardless of the song's length.
        mfcc_processed = np.mean(mfcc, axis=1)

        # if mfcc_processed is not None:
        #     print(f"Success! Vector shape: {mfcc_processed.shape}")
        #     print(f"Vector preview: {mfcc_processed}")

        return mfcc_processed
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None