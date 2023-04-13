import os
import tensorflow_io as tfio
#import network_lib
import network_lib_attention as network_lib
import params
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub

# Soundstream
module = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')

# Audio Stuff
#SAMPLE_RATE = 16000
#N_FFT = 1024
#HOP_LENGTH = 320
#WIN_LENGTH = 640 # (20 ms)
#N_MEL_CHANNELS = 128
#MEL_FMIN = 0.0
#MEL_FMAX = int(SAMPLE_RATE // 2)
#CLIP_VALUE_MIN = 1e-5
#CLIP_VALUE_MAX = 1e8
#N_IMG_CHANNELS = 1 #3

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=params.N_MEL_CHANNELS,
    num_spectrogram_bins=params.N_FFT // 2 + 1,
    sample_rate=params.SAMPLE_RATE,
    lower_edge_hertz=params.MEL_FMIN,
    upper_edge_hertz=params.MEL_FMAX)

def calculate_spectrogram(samples):
  """Calculate mel spectrogram using the parameters the model expects."""
  fft = tf.signal.stft(
      samples,
      frame_length=params.WIN_LENGTH,
      frame_step=params.HOP_LENGTH,
      fft_length=params.N_FFT,
      window_fn=tf.signal.hann_window,
      pad_end=True)
  fft_modulus = tf.abs(fft)

  output = tf.matmul(fft_modulus, MEL_BASIS)

  output = tf.clip_by_value(
      output,
      clip_value_min=params.CLIP_VALUE_MIN,
      clip_value_max=params.CLIP_VALUE_MAX)
  output = tf.math.log(output)
  return output

# data
dataset_repetitions = 5
num_epochs = 1  # train for at least 50 epochs for good results
image_size = (128,128)
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 10
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# optimization
batch_size = 64

widths = [64, 128, 256, 512]
block_depth = 4
batch_size=16

ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


# Handle Paths
dataset_train_path = 'dataset/dataset_timbre_transfer/train'
instruments_name = ['violin','saxophone']
checkpoint_path = "checkpoints/diffusion_model_timbre_transfer_saxophone_to_violin_20230305-150304"

checkpoint_path = "checkpoints/diffusion_model_timbre_transfer_saxophone_to_violin_20230309-170705"



# Each instrument is the same since track names are duplicated
track_names = os.listdir(os.path.join(dataset_train_path,instruments_name[0]))
track_names.sort()
# Let's do violin to saxophone
track_paths_trans = [[os.path.join(dataset_train_path,instruments_name[0],track_name),os.path.join(dataset_train_path,instruments_name[1],track_name)]  for track_name in track_names]


val_perc = 0.2
n_tracks_train = len(track_paths_trans) - int(np.floor(val_perc*len(track_paths_trans)))
train_tracks_paths = track_paths_trans[:n_tracks_train]
val_tracks_paths = track_paths_trans[n_tracks_train:]

#train_tracks_paths, val_tracks_paths = train_tracks_paths[:64], val_tracks_paths[:64],
#audio_path = train_tracks_paths[200]
duration_sample = 40960
duration_track = 480000

def normalize_audio(audio):
    audio = audio - audio.min()
    audio = audio/audio.max()
    audio = (audio*2)-1
    return audio


def read_audio(audio_path,resample=True, SAMPLE_RATE=params.SAMPLE_RATE):
    audio_bin = tf.io.read_file(audio_path)
    audio, sample_rate = tf.audio.decode_wav(audio_bin)
    if resample:
        audio = tfio.audio.resample(audio, rate_in=tf.cast(sample_rate,tf.int64), rate_out=SAMPLE_RATE, name=None)
    return audio


def norm_tensor(audio):
    min_val = tf.math.reduce_min(audio)
    audio = audio - min_val
    max_val = tf.math.reduce_max(audio)
    audio_norm = ((audio/max_val)*2)-1
    return audio_norm, max_val, min_val

def denorm_tensor(audio,max_val,min_val):
    return (((audio +1)/2)*max_val + min_val)

def get_audio_track_diff(cond_track_path, diff_steps=20):
    cond_track = read_audio(cond_track_path)
    # Remove audio Channel and add Batch Dimension
    cond_track = tf.expand_dims(cond_track[:,0],axis=0)
    do_norm = True
    if do_norm:
        cond_track,_,_ = norm_tensor(cond_track)

    # Compute mel spectrogram
    cond_track_spec = calculate_spectrogram(cond_track)

    # Frames given as input to diff model
    N_frames_diff = 128

    # Number of full frames contained in the considered track
    N_frames_full = cond_track_spec.shape[1]//N_frames_diff
    N_frames_gt = cond_track_spec.shape[1]
    # Split the cond track in frame sizes suitable to diff model
    if N_frames_full*N_frames_diff < cond_track_spec.shape[1]:
        cond_track_input_diff = np.zeros((N_frames_full+1,N_frames_diff,params.N_MEL_CHANNELS),dtype=np.float32)
        for i in range(N_frames_full):
            cond_track_input_diff[i] = cond_track_spec[0,(i*N_frames_diff):(i*N_frames_diff)+N_frames_diff]
        N_remaining_frames = len(cond_track_spec[0, (i * N_frames_diff) + N_frames_diff:])
        cond_track_input_diff[i+1,:N_remaining_frames] = cond_track_spec[0, (i * N_frames_diff) + N_frames_diff:]
    else:
        cond_track_input_diff = np.zeros(N_frames_full,N_frames_diff,N_MEL_CHANNELS,1)
        for i in range(N_frames_full):
            cond_track_input_diff[i] = cond_track_spec[0,(i*N_frames_diff):(i*N_frames_diff)+N_frames_diff]


    # Now let's apply the diffusion model
    model = network_lib.DiffusionModel(image_size, widths, block_depth,val_data=None,batch_size=batch_size)
    model.load_weights(checkpoint_path)
    N = cond_track_input_diff.shape[0]

    est_spec = model.generate(
        cond_images=tf.expand_dims(cond_track_input_diff, axis=-1),
        num_images=N,
        diffusion_steps=diff_steps,
    )

    est_spec_shift = tf.expand_dims(tf.zeros_like(cond_track_input_diff), axis=-1).numpy()
    N_slices = est_spec_shift.shape[0]
    for i in range(N_slices - 1):
        # Curr + shifted slice
        cond_shift = tf.expand_dims(
            tf.expand_dims(tf.concat([cond_track_input_diff[i][64:], cond_track_input_diff[i + 1][:64]], axis=0),
                           axis=0), axis=-1)
        est_spec_shift[i] = model.generate(cond_images=cond_shift, num_images=1, diffusion_steps=diff_steps)

    est_spec_smooth = est_spec.numpy()
    for i in range(est_spec.numpy().shape[0] - 1):
        est_spec_smooth[i, 96:] = est_spec_shift[i, 32:64]
        est_spec_smooth[i + 1, :32] = est_spec_shift[i, 64:96]

    est_spec = tf.reshape(est_spec_smooth,(N*128,128)).numpy()[:N_frames_gt]
    cond_track_input_diff = tf.reshape(cond_track_input_diff,(N*128,128)).numpy()[:N_frames_gt]

    est_audio = module(tf.expand_dims(est_spec,axis=0)).numpy()
    cond_audio =  module(tf.expand_dims(cond_track_input_diff,axis=0)).numpy()

    return est_audio, cond_audio



def get_audio_track_diff_norm(cond_track_path, checkpoint_path, model, diff_steps=20):
    cond_track = read_audio(cond_track_path)
    # Remove audio Channel and add Batch Dimension
    cond_track = tf.expand_dims(cond_track[:, 0], axis=0)
    do_norm = True
    if do_norm:
        cond_track, _, _ = norm_tensor(cond_track)

    # Compute mel spectrogram
    cond_track_spec = calculate_spectrogram(cond_track)

    # Frames given as input to diff model
    N_frames_diff = 128

    # Number of full frames contained in the considered track
    N_frames_full = cond_track_spec.shape[1] // N_frames_diff
    N_frames_gt = cond_track_spec.shape[1]
    # Split the cond track in frame sizes suitable to diff model
    if N_frames_full * N_frames_diff < cond_track_spec.shape[1]:
        cond_track_input_diff = np.zeros((N_frames_full + 1, N_frames_diff, params.N_MEL_CHANNELS), dtype=np.float32)
        for i in range(N_frames_full):
            cond_track_input_diff[i] = cond_track_spec[0, (i * N_frames_diff):(i * N_frames_diff) + N_frames_diff]
        N_remaining_frames = len(cond_track_spec[0, (i * N_frames_diff) + N_frames_diff:])
        cond_track_input_diff[i + 1, :N_remaining_frames] = cond_track_spec[0, (i * N_frames_diff) + N_frames_diff:]
    else:
        cond_track_input_diff = np.zeros(N_frames_full, N_frames_diff, params.N_MEL_CHANNELS, 1)
        for i in range(N_frames_full):
            cond_track_input_diff[i] = cond_track_spec[0, (i * N_frames_diff):(i * N_frames_diff) + N_frames_diff]

    # Now let's apply the diffusion model
    # model = network_lib.DiffusionModel(image_size, widths, block_depth,val_data=None,batch_size=batch_size)
    model.load_weights(checkpoint_path)
    N = cond_track_input_diff.shape[0]

    # Compute norm of cond ttrack
    cond_track_spec_reshaped_norm = np.zeros_like(cond_track_input_diff)
    max_val, min_val = np.zeros(N), np.zeros(N)
    for i in range(cond_track_input_diff.shape[0]):
        cond_track_spec_reshaped_norm[i], max_val[i], min_val[i] = norm_tensor(cond_track_input_diff[i])
        # print(str(max_val[i]) + ' ' + str(min_val[i]))

    cond_track_spec_reshaped_norm, max_val, min_val = norm_tensor(cond_track_input_diff)

    est_spec_norm = model.generate_fixed_noise(
        cond_images=tf.expand_dims(cond_track_spec_reshaped_norm, axis=-1),
        num_images=N,
        diffusion_steps=diff_steps,
    )
    # est_spec =  denorm_tensor(est_spec_norm,np.mean(max_val),np.mean(min_val))
    est_spec = denorm_tensor(est_spec_norm, max_val, min_val)

    est_spec = est_spec.numpy()

    est_spec = tf.reshape(est_spec, (N * 128, 128)).numpy()[:N_frames_gt]
    cond_track_input_diff = tf.reshape(cond_track_input_diff, (N * 128, 128)).numpy()[:N_frames_gt]

    est_audio = module(tf.expand_dims(est_spec, axis=0)).numpy()
    cond_audio = module(tf.expand_dims(cond_track_input_diff, axis=0)).numpy()

    return est_audio, cond_audio, cond_track_input_diff, est_spec
