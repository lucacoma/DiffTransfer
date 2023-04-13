import tensorflow as tf
# PARAMETERS
# Audio Stuff
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 640 # (20 ms)
N_MEL_CHANNELS = 128
MEL_FMIN = 0.0
MEL_FMAX = int(SAMPLE_RATE // 2)
CLIP_VALUE_MIN = 1e-5
CLIP_VALUE_MAX = 1e8
N_IMG_CHANNELS = 1 #3

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MEL_CHANNELS,
    num_spectrogram_bins=N_FFT // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=MEL_FMIN,
    upper_edge_hertz=MEL_FMAX)



# data
dataset_repetitions = 5
num_epochs = 1  # train for at least 50 epochs for good results
mel_spec_size = (128, 128)
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
ema = 0.999
learning_rate =  2e-5
weight_decay = 1e-4

# New
#widths = [32,  64, 96,  128]
#block_depth = 2

# optimization
batch_size = 64

widths = [64, 128, 256, 512]
has_attention = [False, False, True, True]
block_depth = 4
batch_size = 16

duration_sample = 40960 #*2 if 256
duration_track = 480000

