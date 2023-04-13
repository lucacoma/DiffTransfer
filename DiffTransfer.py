import os
import tqdm
os.environ['CUDA_VISIBLE_DEVICES']="2"
import network_lib_attention as network_lib
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import datetime
import os
import numpy as np
import argparse
import params
import audio_utils
from tensorflow import keras
tf.config.list_physical_devices('GPU')
# SoundStream Spectrogram Inverter (Stuff stolen from https://storage.googleapis.com/music-synthesis-with-spectrogram-diffusion/index.html) and https://tfhub.dev/google/soundstream/mel/decoder/music/1
module = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')
# STARNET DATASET --> https://zenodo.org/record/6917099#.ZBiIEuzMI-Q
os.listdir('/nas/public/dataset/maestro/maestro-v3.0.0')
pre_load = False

do_norm = True

def preprocess_dataset(audio_paths):
    #for audio_path in val_tracks_paths:
    #    print(audio_path)

    audio_res1 = audio_utils.read_audio(audio_paths[0])
    audio_res2 = audio_utils.read_audio(audio_paths[1])
    if do_norm:
        audio_res1,_,_ = audio_utils.norm_tensor(audio_res1)
        audio_res2,_,_ = audio_utils.norm_tensor(audio_res2)
    #idx = tf.random.uniform(shape=(), minval=0, maxval=duration_track - duration_sample, dtype=tf.int32)
    idx = tf.random.uniform(shape=(), minval=0, maxval=tf.size(audio_res1) - params.duration_sample, dtype=tf.int32)
    audio_res1 = audio_res1[idx:idx+params.duration_sample]
    audio_res2 = audio_res2[idx:idx+params.duration_sample]


    audio_res1 = tf.expand_dims(tf.squeeze(audio_res1), axis=0)
    audio_res2 = tf.expand_dims(tf.squeeze(audio_res2), axis=0)

    spec1 = audio_utils.calculate_spectrogram(audio_res1)
    spec2 = audio_utils.calculate_spectrogram(audio_res2)

    spec1 = tf.expand_dims(tf.squeeze(spec1),axis=-1)
    spec1 = tf.reshape(spec1,shape=(params.mel_spec_size[0],params.mel_spec_size[1],1))

    spec2 = tf.expand_dims(tf.squeeze(spec2),axis=-1)
    spec2 = tf.reshape(spec2,shape=(params.mel_spec_size[0],params.mel_spec_size[1],1))


    if network_lib.do_norm_specs:
        spec1,_,_ = audio_utils.norm_tensor(spec1)
        spec2,_,_ = audio_utils.norm_tensor(spec2)

    return tf.concat([spec1,spec2],axis=-1)

def prepare_dataset(paths, training=True):
    files_ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        features_ds = files_ds.map(preprocess_dataset).repeat(2).batch(params.batch_size,drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        features_ds = files_ds.map(preprocess_dataset).cache().repeat(4).batch(params.batch_size,drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    return features_ds

def main():
    parser = argparse.ArgumentParser(description='Train log-mel-to-mask network')
    parser.add_argument('--dataset_train_path', type=str, help='Folder containing Train/val Dataset audio',
                        default='dataset/starnet/starnet_reduced')
    parser.add_argument('--desired_instrument', type=str, help='Desired Output Timbre',
                        default='strings')
    parser.add_argument('--conditioning_instrument', type=str, help='Desired Conditioning Timbre',
                        default='clarinet')
    parser.add_argument('--GPU', type=str, help='Select GPU number',
                        default='0')
    parser.add_argument('--train', type=str, help='Select GPU number',
                        default='True')
    dict_instruments = {"clarinet":"1","vibraphone":"2","strings":"4","piano":"5",'clarinet_vibraphone':"0",'strings_piano':"3"}
    args = parser.parse_args()
    desired_instrument = args.desired_instrument
    conditioning_instrument = args.conditioning_instrument
    dataset_train_path = args.dataset_train_path
    train = args.train
    print('Timbre transfering from '+conditioning_instrument+' to'+desired_instrument)

    # Handle Paths
    instruments_name = [dict_instruments[desired_instrument],dict_instruments[conditioning_instrument]]

    checkpoint_path = "checkpoints/ATT_STARNET_NORM_diffusion_model_timbre_transfer_"+conditioning_instrument+'_to_'+desired_instrument+'_'+ datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/ATT_STARNET_"+conditioning_instrument+'_to_'+desired_instrument+datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"))

    log_dir = 'logs/'
    logdir = log_dir + 'ATT_STARNET_NORM_diffusion_timbre_transfer_' + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + '5000__'+conditioning_instrument+'_to_'+desired_instrument

    # Each instrument is the same since track names are duplicatedk

    tracks_full = os.listdir(dataset_train_path)
    cond_tracks = [track for track in tracks_full if track.split('.')[-2]==instruments_name[1]]
    trgt_tracks = [track for track in tracks_full if track.split('.')[-2]==instruments_name[0]]
    cond_tracks.sort()
    trgt_tracks.sort()
    track_paths_trans = [[os.path.join(dataset_train_path,trgt_tracks[i]),os.path.join(dataset_train_path,cond_tracks[i])] for i in range(len(cond_tracks))]


    val_perc = 0.2
    n_tracks_train = len(track_paths_trans) - int(np.floor(val_perc * len(track_paths_trans)))
    rng = np.random.default_rng(12345)
    idxs = rng.choice(len(track_paths_trans), len(track_paths_trans), False)
    idxs_train = idxs[:n_tracks_train]
    idxs_val = idxs[n_tracks_train:]

    train_tracks_paths = np.array(track_paths_trans)[idxs_train].tolist()
    val_tracks_paths = np.array(track_paths_trans)[idxs_val].tolist()

    train_dataset = prepare_dataset(train_tracks_paths)
    val_dataset = prepare_dataset(val_tracks_paths, training=False)


    # create and compile the model
    first = True
    for a in val_dataset.take(2):
        if first:
            val_data = a
            first = False
        else:
            val_data = tf.concat([val_data, a],axis=0)
    val_data = a[:18]
    print(val_data.shape)
    model = network_lib.DiffusionModel(params.mel_spec_size, params.widths, params.block_depth, val_data, params.has_attention, logdir=logdir,batch_size=params.batch_size,)
    model.network.summary()

    if train:
        model.compile(
            optimizer=keras.optimizers.experimental.AdamW(
                learning_rate=params.learning_rate, weight_decay=params.weight_decay
            ),
            loss=keras.losses.mean_absolute_error,
        )

        # save the best model based on the validation KID metric

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor="val_n_loss",
            mode="min",
            save_best_only=True,
        )

        # calculate mean and variance of training dataset for normalization
        model.fit(
            train_dataset,
            epochs=5000,
            validation_data=val_dataset,
            callbacks=[
                keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
                checkpoint_callback,
                tensorboard_callback,
            ],
        )

if __name__=='__main__':
    main()
