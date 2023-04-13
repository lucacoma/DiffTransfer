import os
os.environ['CUDA_VISIBLE_DEVICES']="5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import network_lib_attention as network_lib
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
import params
import audio_utils
import numpy as np
from IPython.display import Audio
module = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')

val_data = None # Param needed to load models
test_dataset_path = 'dataset/music_star_test'

test_dataset_dict = {"001" :"Pirates of the Caribbean Theme",
"002" :"My Heart Will Go on",
"003" :"Beethoven's String",
"004" :"Moonlight Sonata",
"005" :"Fur Elise",
"006" :"Brahms's Clarinet",
"007" :"Beethoven's Piano",
"008" :"Dvorak's String",
"009" :"Romeo and Juliet",
"010":"Nuvole Blanche",
}
N_tracks = len(test_dataset_dict)
est_folder = 'eval/est_diffusion'

def main():
    parser = argparse.ArgumentParser(description='Train log-mel-to-mask network')
    parser.add_argument('--conversion_type', type=int, help='clarinet/vibraphone_to_strings_piano or viceversa',
                        default=1)
    parser.add_argument('--diff_steps', type=int, help='Number of steps in diffusion',
                        default=20)

    args = parser.parse_args()
    conversion_type = args.conversion_type
    diff_steps = args.diff_steps

    if conversion_type == 0:
        print('Convert clarinet/vibraphone to violin/piano')
    if conversion_type == 1:
        print('Convert strings/piano to clarinet/vibraphone ')

    # Select correct checkpoint
    if conversion_type == 0:
        print('Convert clarinet/vibraphone to violin/piano')
        checkpoint_path = "checkpoints/ATT_STARNET_NORM_diffusion_model_timbre_transfer_clarinet_vibraphone_to_strings_piano_20230327-103551"

    if conversion_type == 1:
        print('Convert strings/piano to clarinet/vibraphone ')
        checkpoint_path = "checkpoints/ATT_STARNET_NORM_diffusion_model_timbre_transfer_strings_piano_to_clarinet_vibraphone_20230327-103700"

    model = network_lib.DiffusionModel(params.mel_spec_size, params.widths, params.block_depth, val_data,
                                           params.has_attention, logdir='', batch_size=params.batch_size, )


    model.load_weights(checkpoint_path)


    for n_t in tqdm.tqdm(range(N_tracks)):
        # Load correct track
        idx_key = list(test_dataset_dict.keys())[n_t]
        final_folder = 'mixture_model'

        if conversion_type == 0:
            track_name = idx_key + '.3.wav'
        if conversion_type == 1:
            track_name = idx_key + '.0.wav'
        track_est_path = os.path.join(est_folder, final_folder, track_name)
        print(track_est_path)

        if conversion_type == 0:
            gt_track = os.path.join(test_dataset_path, idx_key + '.3.wav')
            cond_track = os.path.join(test_dataset_path, idx_key + '.0.wav')

        if conversion_type == 1:
            gt_track = os.path.join(test_dataset_path, idx_key + '.0.wav')
            cond_track = os.path.join(test_dataset_path, idx_key + '.3.wav')

        est_audio, _, _, _ = audio_utils.get_audio_track_diff_norm(
            cond_track_path=cond_track,
            checkpoint_path=checkpoint_path,
            model=model,
            diff_steps=diff_steps)

        # Write Audio
        audio_encoded = tf.audio.encode_wav(
            tf.transpose(est_audio), params.SAMPLE_RATE, name=None
        )
        write_file_op = tf.io.write_file(track_est_path, audio_encoded)


if __name__=='__main__':
    main()