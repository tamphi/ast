from pydub import AudioSegment
from pydub.utils import make_chunks
import pathlib
import csv
import numpy as np
import librosa
import os

dataset_dir = "data/shipsEar_AUDIOS/"
chunk_dataset_dir = "data/shipsEar_AUDIOS_chunk/"
raw_audio_files = pathlib.Path(dataset_dir).glob('*.wav')
chunk_length_ms = 1000 # millisecond 
chunk_sample_rate=44100 #Set to 0 to disable resampling 
chunk_csv_file = "data/meta/ship_{chunk_length}ms_chunks_{sample_rate}k.csv".format(chunk_length=chunk_length_ms,sample_rate=chunk_sample_rate)
# chunk_csv_file = "data/meta/ship_{chunk_length}ms_chunks.csv".format(chunk_length=chunk_length_ms)

ship_chunk_dict = {}

type_dict ={}
total_chunks = 0
j = 0
index = 0
base_csv = np.loadtxt('data/shipsEar_AUDIOS/shipsEar.csv', delimiter=',', dtype='str',skiprows=1)
for file in raw_audio_files:
    #Get raw audio for segmentation
    segment_audio = AudioSegment.from_file(file,"wav")
    if chunk_sample_rate > 0: 
        segment_audio = segment_audio.set_frame_rate(chunk_sample_rate)

    #Make chunks
    chunks = make_chunks(segment_audio,chunk_length_ms)
    
    #split file path by "/" and ".wav" to get file name
    file_split_path = str(file).split("/")
    ship_name = file_split_path[-1].split(".wav")[0]

    #Calculate total number of audio chunks
    total_chunks += len(chunks)

    #Get index and labels of the audio
    cur_id = base_csv[j][0]
    #replace all white space with _ and lowercase
    cur_type = str(base_csv[j][3]).replace(" ", "_").lower()
    
    for i,chunk in enumerate(chunks):
        chunk_name = "{ship}_chunk_{chunk_number}.wav".format(ship=ship_name,chunk_number=i)
        ship_chunk_dict[index] = [chunk_name,cur_id,cur_type]
        print("Exporting {} to {}".format(chunk_name,chunk_dataset_dir))
        chunk.export(chunk_dataset_dir + chunk_name, format="wav")
        index+=1
    j += 1


field = ["Index","File","Id","Label"]
with open(chunk_csv_file,"w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(field)
    for key,value in ship_chunk_dict.items():
        writer.writerow([key,*value])

print(total_chunks)



