import json
import os
import numpy as np
import random
import librosa

micro_wav = []
refer1_wav = []
refer2_wav = []
clean_wav = []

wav_file_path = "D:\Study\SAEC\Experiments6\PrepareData\wav_timit_train_SAEC_32h_20200905_temp" #"/media/chenglinjuan/PythonProject/SAEC/Experiments6/PrepareData/wav_timit_train_SAEC_32h_20200905"

micro1_dataset = os.path.join(wav_file_path, "micro1")
micro1_wav = librosa.util.find_files(micro1_dataset, ext="wav", limit=None, offset=0)
refer1_dataset = os.path.join(wav_file_path, "refer1")
refer1_wav = librosa.util.find_files(refer1_dataset, ext="wav", limit=None, offset=0)
refer2_dataset = os.path.join(wav_file_path, "refer2")
refer2_wav = librosa.util.find_files(refer2_dataset, ext="wav", limit=None, offset=0)
noise_dataset = os.path.join(wav_file_path, "noise")
noise_wav = librosa.util.find_files(noise_dataset, ext="wav", limit=None, offset=0)
echo_micro1_dataset = os.path.join(wav_file_path, "echo_micro1")
echo_micro1_wav = librosa.util.find_files(echo_micro1_dataset, ext="wav", limit=None, offset=0)
clean1_dataset = os.path.join(wav_file_path, "near_end_micro1")
clean1_wav = librosa.util.find_files(clean1_dataset, ext="wav", limit=None, offset=0)

total_sent = len(clean1_wav)
print("The total sentences is: %d" % total_sent)

micro_dataset_write = []
refer1_dataset_write = []
refer2_dataset_write = []
noise_dataset_write = []
echo_micro1_dataset_write = []
clean_dataset_write = []
for idx in range(total_sent):
    tmp_path, tmp_wav = os.path.split(micro1_wav[idx])
    micro_dataset_write.append(tmp_wav)
    tmp_path, tmp_wav = os.path.split(refer1_wav[idx])
    refer1_dataset_write.append(tmp_wav)
    tmp_path, tmp_wav = os.path.split(refer2_wav[idx])
    refer2_dataset_write.append(tmp_wav)
    tmp_path, tmp_wav = os.path.split(noise_wav[idx])
    noise_dataset_write.append(tmp_wav)
    tmp_path, tmp_wav = os.path.split(echo_micro1_wav[idx])
    echo_micro1_dataset_write.append(tmp_wav)
    tmp_path, tmp_wav = os.path.split(clean1_wav[idx])
    clean_dataset_write.append(tmp_wav)

## split train and cv
split_rate = 0.8
num_train = np.int(split_rate * total_sent)
num_test = total_sent - num_train
idx = np.array(random.sample(range(0, total_sent), total_sent))
print("The total sentences of training set is: %d, cv set isL %d" % (num_train, num_test))

train_micro_wav = []
cv_micro_wav = []
train_refer1_wav = []
cv_refer1_wav = []
train_refer2_wav = []
cv_refer2_wav = []
train_noise_wav = []
cv_noise_wav = []
train_echo_micro1_wav = []
cv_echo_micro1_wav = []
train_clean_wav = []
cv_clean_wav = []
n = 0
for i in idx:
    if n < num_train:
        train_micro_wav.append(micro_dataset_write[i])
        train_refer1_wav.append(refer1_dataset_write[i])
        train_refer2_wav.append(refer2_dataset_write[i])
        train_noise_wav.append(noise_dataset_write[i])
        train_echo_micro1_wav.append(echo_micro1_dataset_write[i])
        train_clean_wav.append(clean_dataset_write[i])
    else:
        cv_micro_wav.append(micro_dataset_write[i])
        cv_refer1_wav.append(refer1_dataset_write[i])
        cv_refer2_wav.append(refer2_dataset_write[i])
        cv_noise_wav.append(noise_dataset_write[i])
        cv_echo_micro1_wav.append(echo_micro1_dataset_write[i])
        cv_clean_wav.append(clean_dataset_write[i])
    n = n + 1

## write file
if not os.path.exists("JSON/train"):
    os.mkdir("JSON/train")

if not os.path.exists("JSON/cv"):
    os.mkdir("JSON/cv")

# micro_wav = json.dumps(micro_wav,indent=1)
with open("JSON/train/train_micro_wav.json", "w") as f:
    json.dump(train_micro_wav, f, indent=1)

with open("JSON/cv/cv_micro_wav.json", "w") as f:
    json.dump(cv_micro_wav, f, indent=1)

with open("JSON/train/train_refer1_wav.json", "w") as f:
    json.dump(train_refer1_wav, f, indent=1)

with open("JSON/cv/cv_refer1_wav.json", "w") as f:
    json.dump(cv_refer1_wav, f, indent=1)

with open("JSON/train/train_refer2_wav.json", "w") as f:
    json.dump(train_refer2_wav, f, indent=1)

with open("JSON/cv/cv_refer2_wav.json", "w") as f:
    json.dump(cv_refer2_wav, f, indent=1)

with open("JSON/train/train_noise_wav.json", "w") as f:
    json.dump(train_noise_wav, f, indent=1)

with open("JSON/cv/cv_noise_wav.json", "w") as f:
    json.dump(cv_noise_wav, f, indent=1)

with open("JSON/train/train_echo_micro1_wav.json", "w") as f:
    json.dump(train_echo_micro1_wav, f, indent=1)

with open("JSON/cv/cv_echo_micro1_wav.json", "w") as f:
    json.dump(cv_echo_micro1_wav, f, indent=1)

with open("JSON/train/train_clean_wav.json", "w") as f:
    json.dump(train_clean_wav, f, indent=1)

with open("JSON/cv/cv_clean_wav.json", "w") as f:
    json.dump(cv_clean_wav, f, indent=1)
