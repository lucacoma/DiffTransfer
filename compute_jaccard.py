"""
Taken from https://github.com/mahshidaln/Music-STAR/blob/ad23e9f9f18e8872d5e94d71c58fb72ade95c9b1/evaluation/pitch.py
"""


import os
import sys
import librosa
import numpy as np
import pretty_midi
from tqdm import tqdm
SR = 16000
import essentia.standard as estd




MFCC_KWARGS = dict(
    n_mfcc=26,
    hop_length=500
)


def get_pitches(audio):
    pitches = estd.MultiPitchMelodia(sampleRate=SR)(audio)
    pitches = [[pretty_midi.utilities.hz_to_note_number(p) for p in pl if not np.isclose(0, p)]
               for pl in pitches]
    pitches = [[int(p + 0.5) for p in pl] for pl in pitches]
    return pitches


def pitch_jaccard(output, reference):
    pitches_output, pitches_reference = get_pitches(output), get_pitches(reference)
    assert len(pitches_output) == len(pitches_reference)
    jaccard = []
    for pl_output, pl_reference in zip(pitches_output, pitches_reference):
        matches = len(set(pl_output) & set(pl_reference))
        total = len(set(pl_output) | set(pl_reference))
        if total == 0:
            jaccard.append(0)
        else:
            jaccard.append(1 - matches / total)
    jaccard = np.mean(jaccard)
    return jaccard

def normalize_audio(audio):
    audio = audio - audio.min()
    audio = audio/audio.max()
    audio = (audio*2)-1
    return audio

def main():
    top = 'samples'
    ref_dir = '/nas/home/lcomanducci/music_txt/eval/music_star_test'
    diff_dir = '/nas/home/lcomanducci/music_txt/eval/est_diffusion'
    music_net_dir = '/nas/home/lcomanducci/music_txt/eval/est_music_net'

    model_types = ['separate_model_individual_tracks', 'separate_model', 'mixture_model']

    for i in range(len(model_types)):
        track_names = os.listdir(os.path.join(diff_dir,model_types[i]))
        N_tracks = len(track_names)
        jaccard_array_diff = np.zeros(N_tracks)
        jaccard_array_music_net = np.zeros(N_tracks)

        for n_t in tqdm(range(N_tracks)):
            #print(track_names[i])
            outa_diff, _ = librosa.load(os.path.join(diff_dir,model_types[i],track_names[n_t]), SR)
            #print(os.path.join(diff_dir,model_types[i],track_names[n_t]))
            outa_music_net, _ = librosa.load(os.path.join(music_net_dir,model_types[i],track_names[n_t]), SR)
            # print(os.path.join(music_net_dir,model_types[i],track_names[n_t]))

            outa_diff, outa_music_net = normalize_audio(outa_diff), normalize_audio(outa_music_net)
            refa, _ = librosa.load(os.path.join(ref_dir,track_names[n_t]), SR)
            refa = normalize_audio(refa)#-np.mean(refa)
            jaccard_array_diff[n_t] = pitch_jaccard(outa_diff, refa)
            jaccard_array_music_net[n_t] = pitch_jaccard(outa_music_net, refa)
        print('Model: ' + model_types[i] + ' Jaccard diffusion: ' + str(np.mean(jaccard_array_diff)) + ' Jaccard musicstar: ' + str(np.mean(jaccard_array_music_net)))

if __name__ == "__main__":
    main()