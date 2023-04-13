import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
import audio_utils
import librosa
import params
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
import numpy as np


gt_paths = '/nas/home/lcomanducci/music_txt/dataset/music_star_test'
normalize_gt_tracks = False
if normalize_gt_tracks:
    gt_paths_tracks = tf.io.gfile.glob(gt_paths+'/*.wav')
    for i in tqdm.tqdm(range(len(gt_paths_tracks))):
        gt_track, sr = librosa.load(path=gt_paths_tracks[i],sr=params.SAMPLE_RATE)
        gt_track = audio_utils.normalize_audio(gt_track)
        audio_encoded = tf.audio.encode_wav(
            tf.expand_dims(gt_track,axis=1), params.SAMPLE_RATE, name=None
        )
        write_file_op = tf.io.write_file(gt_paths_tracks[i], audio_encoded)

model_type = ['separate_model_individual_tracks', 'separate_model','mixture_model']
model_selected_idx = 0
diff_paths = os.path.join('/nas/home/lcomanducci/music_txt/eval/est_diffusion',model_type[model_selected_idx])
music_star_paths = os.path.join('/nas/home/lcomanducci/music_txt/eval/est_music_net',model_type[model_selected_idx])


normalize_music_star_tracks = False
if normalize_gt_tracks:
    for m in range(len(model_type)):
        music_star_paths_tracks = tf.io.gfile.glob(os.path.join('/nas/home/lcomanducci/music_txt/eval/est_music_net',model_type[m])+'/*.wav')
        for i in tqdm.tqdm(range(len(music_star_paths_tracks))):
            music_star_track, sr = librosa.load(path=music_star_paths_tracks[i], sr=params.SAMPLE_RATE)
            music_star_track = audio_utils.normalize_audio(music_star_track)
            audio_encoded = tf.audio.encode_wav(
                tf.expand_dims(music_star_track, axis=1), params.SAMPLE_RATE, name=None
            )
            write_file_op = tf.io.write_file(music_star_paths_tracks[i], audio_encoded)



gt_track, sr = librosa.load(path=os.path.join(gt_paths,'010.1.wav'),sr=params.SAMPLE_RATE)
diff_track, sr  = librosa.load(path=os.path.join(diff_paths,'010.1.wav'),sr=params.SAMPLE_RATE)
music_star_track, sr  = librosa.load(path=os.path.join(music_star_paths,'010.1_0.wav'),sr=params.SAMPLE_RATE)


#gt_track = audio_utils.normalize_audio(gt_track)
#diff_track = audio_utils.normalize_audio(diff_track)
#music_star_track = audio_utils.normalize_audio(music_star_track)

print(str(gt_track.shape))
print(str(diff_track.shape))
print(str(music_star_track.shape))


plt.figure()
plt.subplot(311)
plt.plot(gt_track)
plt.subplot(312)
plt.plot(diff_track)
plt.subplot(313)
plt.plot(music_star_track)
plt.show()

import numpy as np
conditions = ['C1','C2']
n_conditions = 8
for n in range(n_conditions):
    idxs = np.random.choice(2, 2,replace=False)
    print(conditions[idxs[0]]+'  '+conditions[idxs[1]])



# Up-mix tracks
path = '/nas/home/lcomanducci/music_txt/009.0_music_net.wav'
track, sr = librosa.load(path=path, sr=params.SAMPLE_RATE)
track = audio_utils.normalize_audio(track)
track = np.concatenate([np.expand_dims(track,axis=-1),np.expand_dims(track,axis=-1)],axis=-1)
audio_encoded = tf.audio.encode_wav(
    track, params.SAMPLE_RATE, name=None
)
write_file_op = tf.io.write_file(path, audio_encoded)

folder_names = ['part_1','part_2']
for i in range(len(folder_names)):
    paths_tracks = tf.io.gfile.glob(os.path.join('/nas/home/lcomanducci/music_txt/resources/audio/timbre_transfer', folder_names[i]) + '/*.wav')
    for t in range(len(paths_tracks)):
        path = paths_tracks[t]
        track, sr = librosa.load(path=path, sr=params.SAMPLE_RATE)
        track = audio_utils.normalize_audio(track)
        plt.plot(track)
        plt.show()
        track = np.concatenate([np.expand_dims(track, axis=-1), np.expand_dims(track, axis=-1)], axis=-1)
        # music_star_track = audio_utils.normalize_audio(path)
        audio_encoded = tf.audio.encode_wav(
            track, params.SAMPLE_RATE, name=None
        )
        write_file_op = tf.io.write_file(path, audio_encoded)
print('done')

