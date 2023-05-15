#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import re

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

files = "../../Datasets/spokencoco/SpokenCOCO"
train_fn = Path(files) / 'SpokenCOCO_train.json'
with open(train_fn, 'r') as f:
    data = json.load(f)

data = data['data']
# neg_imgs = set()
# neg_wavs = set()

# for entry in tqdm(data):
#     image = entry['image']
#     for caption in entry['captions']:
#         # for word in base:
#         #     if re.search(word, caption['text'].lower()) is not None:
#         c = False
#         for v in vocab:
#             if re.search(v, caption['text'].lower()) is not None:
#                 c = True
#         if c is False: 
#             neg_imgs.add(image)
#             neg_wavs.add(caption['wav'])

# val_neg_imgs = np.random.choice(list(neg_imgs), 7000, replace=False)
# train_neg_imgs = [entry for entry in list(neg_imgs) if entry not in val_neg_imgs]
# val_neg_wavs = np.random.choice(list(neg_wavs), 7000, replace=False)
# train_neg_wavs = [entry for entry in list(neg_wavs) if entry not in val_neg_wavs]

key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['key'].item()
id_to_word_key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for l in key:
    print(f'{key[l]}: {l}')

# audio_samples = np.load(Path("../data/sampled_audio_data.npz"), allow_pickle=True)['data'].item()
# image_samples = np.load(Path("../data/sampled_img_data.npz"), allow_pickle=True)['data'].item()

ss_save_fn = '../support_set/support_set_100.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()

s = {}
for wav_name in support_set:
    _, _, _, _, _, word = support_set[wav_name]
    if key[word] not in s: s[key[word]] = []
    s[key[word]].append(support_set[wav_name])
print(len(s))

# Training 

pos_lookup = {}

for id in s:
    if id not in pos_lookup: pos_lookup[id] = {"audio": {}, "images": {}}
    for wav, image, spkr, start, stop, word in s[id]:
        
        wav_name = Path(wav).stem
        if wav_name not in pos_lookup[id]['audio']: pos_lookup[id]['audio'][wav_name] = []
        pos_lookup[id]['audio'][wav_name].append(wav)

        image_name = Path(image).stem
        if image_name not in pos_lookup[id]['images']: pos_lookup[id]['images'][image_name] = []
        pos_lookup[id]['images'][image_name].append(str(image))

neg_lookup = {}
for id in tqdm(sorted(pos_lookup), desc='Sampling audio training negatives'):
    wavs_with_id = list(pos_lookup[id]['audio'].keys())
    imgs_with_id = list(pos_lookup[id]['images'].keys())
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
    
    for neg_id in all_ids: #tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['audio'] for i in pos_lookup[neg_id]['audio'][name] if name not in wavs_with_id]
        if len(temp) > 0:
            neg_lookup[id]['audio'][neg_id] = []
            # for w in list(set(train_neg_wavs) - set(temp)):
            for w in list(set(temp)):
                neg_lookup[id]['audio'][neg_id].append(w)

        temp = [i for name in pos_lookup[neg_id]['images'] for i in pos_lookup[neg_id]['images'][name] if name not in imgs_with_id]
        if len(temp) > 0:
            neg_lookup[id]['images'][neg_id] = []
            # for w in list(set(train_neg_wavs) - set(temp)):
            for im in list(set(temp)):
                neg_lookup[id]['images'][neg_id].append(im)

for id in pos_lookup:
    print(id, " audio ", len(pos_lookup[id]['audio']), " images ", len(pos_lookup[id]['images']))

for id in neg_lookup:
    print(id, " audio ", len(neg_lookup[id]['audio']), " images ", len(neg_lookup[id]['images']))

np.savez_compressed(
    Path("../data/train_lookup"), 
    lookup=pos_lookup,
    neg_lookup=neg_lookup
    )

# Validation

pos_lookup = {}

for id in s:
    if id not in pos_lookup: pos_lookup[id] = {"audio": {}, "images": {}}
    for wav, image, spkr, start, stop, word in s[id]:
        
        wav_name = Path(wav).stem
        if wav_name not in pos_lookup[id]['audio']: pos_lookup[id]['audio'][wav_name] = []
        pos_lookup[id]['audio'][wav_name].append(wav)

        image_name = Path(image).stem
        if image_name not in pos_lookup[id]['images']: pos_lookup[id]['images'][image_name] = []
        pos_lookup[id]['images'][image_name].append(str(image))


neg_lookup = {}
for id in tqdm(sorted(pos_lookup), desc='Sampling audio training negatives'):
    wavs_with_id = list(pos_lookup[id]['audio'].keys())
    imgs_with_id = list(pos_lookup[id]['images'].keys())
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
    
    for neg_id in all_ids: #tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['audio'] for i in pos_lookup[neg_id]['audio'][name] if name not in wavs_with_id]
        if len(temp) > 0:
            neg_lookup[id]['audio'][neg_id] = []
            # for w in list(set(train_neg_wavs) - set(temp)):
            for w in list(set(temp)):
                neg_lookup[id]['audio'][neg_id].append(w)

        temp = [i for name in pos_lookup[neg_id]['images'] for i in pos_lookup[neg_id]['images'][name] if name not in imgs_with_id]
        if len(temp) > 0:
            neg_lookup[id]['images'][neg_id] = []
            # for w in list(set(train_neg_wavs) - set(temp)):
            for im in list(set(temp)):
                neg_lookup[id]['images'][neg_id].append(im)


for id in pos_lookup:
    print(id, " audio ", len(pos_lookup[id]['audio']), " images ", len(pos_lookup[id]['images']))

for id in neg_lookup:
    print(id, " audio ", len(neg_lookup[id]['audio']), " images ", len(neg_lookup[id]['images']))

np.savez_compressed(
    Path("../data/val_lookup"), 
    lookup=pos_lookup,
    neg_lookup=neg_lookup
)