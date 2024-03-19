import torchaudio
import pathlib

def _wav2fbank(filename, filename2=None):
    # mixup
    if filename2 == None:
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
    # mixup
    else:
        waveform1, sr = torchaudio.load(filename)
        waveform2, _ = torchaudio.load(filename2)

        waveform1 = waveform1 - waveform1.mean()
        waveform2 = waveform2 - waveform2.mean()

        if waveform1.shape[1] != waveform2.shape[1]:
            if waveform1.shape[1] > waveform2.shape[1]:
                # padding
                temp_wav = torch.zeros(1, waveform1.shape[1])
                temp_wav[0, 0:waveform2.shape[1]] = waveform2
                waveform2 = temp_wav
            else:
                # cutting
                waveform2 = waveform2[0, 0:waveform1.shape[1]]

        # sample lambda from uniform distribution
        #mix_lambda = random.random()
        # sample lambda from beta distribtion
        mix_lambda = np.random.beta(10, 10)

        mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
        waveform = mix_waveform - mix_waveform.mean()
    if sr != len(waveform[0,:]):
        print(sr,len(waveform[0,:]),filename)
    # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                #   window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)


chunk_dataset_dir = "data/shipsEar_AUDIOS_chunk"
raw_audio_files = pathlib.Path(chunk_dataset_dir).glob('*.wav')

for file in raw_audio_files:
    _wav2fbank(file)