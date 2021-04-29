import json
import os
import numpy as np
import random
import librosa

micro1_wav = []
micro2_wav = []
refer1_wav = []
refer2_wav = []
clean1_wav = []
clean2_wav = []

wav_file_path = "/media/chenglinjuan/PythonProject/SAEC/Experiments6/PrepareData/wav_timit_train_SAEC_32h_20200905_test/seenspeaker_UnseenSpeakerPosition1_05m_far"
JSON_file_path = "JSON/test/seenspeaker_UnseenSpeakerPosition1_05m_far"
if not os.path.exists(JSON_file_path):
    os.makedirs(JSON_file_path)

micro1_dataset = os.path.join(wav_file_path, "micro1")
micro1_wav = librosa.util.find_files(micro1_dataset, ext="wav", limit=None, offset=0)
refer1_dataset = os.path.join(wav_file_path, "refer1")
refer1_wav = librosa.util.find_files(refer1_dataset, ext="wav", limit=None, offset=0)
refer2_dataset = os.path.join(wav_file_path, "refer2")
refer2_wav = librosa.util.find_files(refer2_dataset, ext="wav", limit=None, offset=0)
#noise_dataset = os.path.join(wav_file_path, "noise")
#noise_wav = librosa.util.find_files(noise_dataset, ext="wav", limit=None, offset=0)
#echo_micro1_dataset = os.path.join(wav_file_path, "echo_micro1")
#echo_micro1_wav = librosa.util.find_files(echo_micro1_dataset, ext="wav", limit=None, offset=0)
#clean1_dataset = os.path.join(wav_file_path, "near_end_micro1")
#clean1_wav = librosa.util.find_files(clean1_dataset, ext="wav", limit=None, offset=0)

total_sent = len(micro1_wav)
print("The total sentences is: %d" % total_sent)

micro_dataset_write = []
refer1_dataset_write = []
refer2_dataset_write = []
#noise_dataset_write = []
#echo_micro1_dataset_write = []
#clean_dataset_write = []
for idx in range(total_sent):
    tmp_path, tmp_wav = os.path.split(micro1_wav[idx])
    micro_dataset_write.append(tmp_wav)
    tmp_path, tmp_wav = os.path.split(refer1_wav[idx])
    refer1_dataset_write.append(tmp_wav)
    tmp_path, tmp_wav = os.path.split(refer2_wav[idx])
    refer2_dataset_write.append(tmp_wav)
#    tmp_path, tmp_wav = os.path.split(noise_wav[idx])
#    noise_dataset_write.append(tmp_wav)
#    tmp_path, tmp_wav = os.path.split(echo_micro1_wav[idx])
#    echo_micro1_dataset_write.append(tmp_wav)
#    tmp_path, tmp_wav = os.path.split(clean1_wav[idx])
#    clean_dataset_write.append(tmp_wav)


## write file
with open(os.path.join(JSON_file_path, "test_micro_wav.json"), "w") as f:
    json.dump(micro_dataset_write, f, indent=1)

with open(os.path.join(JSON_file_path, "test_refer1_wav.json"), "w") as f:
    json.dump(refer1_dataset_write, f, indent=1)

with open(os.path.join(JSON_file_path, "test_refer2_wav.json"), "w") as f:
    json.dump(refer2_dataset_write, f, indent=1)

#with open(os.path.join(JSON_file_path, "test_noise_wav.json"), "w") as f:
#    json.dump(noise_dataset_write, f, indent=1)

#with open(os.path.join(JSON_file_path, "test_echo_micro1_wav.json"), "w") as f:
#    json.dump(echo_micro1_dataset_write, f, indent=1)

#with open(os.path.join(JSON_file_path, "test_clean_wav.json"), "w") as f:
#    json.dump(clean_dataset_write, f, indent=1)
