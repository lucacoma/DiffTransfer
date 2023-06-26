---
title: DiffTransfer
---


# Multi Instrument timbre transfer (mixture)

## Clarinet/Vibraphone to Piano/Strings

|Name| Input Mixture  |  DiffTransfer | Music-STAR| Ground Truth|
|---|---|---|---|---|
| Pirates of The Carribean | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.0.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/mixture_model/001.3.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/mixture_model/001.3.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.3.wav" type="audio/mpeg"></audio> |
| My Heart Will Go On | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.0.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/mixture_model/002.3.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/mixture_model/002.3.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.3.wav" type="audio/mpeg"></audio> |
| Beethoven's String | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.0.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/mixture_model/003.3.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/mixture_model/003.3.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.3.wav" type="audio/mpeg"></audio> |


## Piano/Strings to Clarinet/Vibraphone 

|Name| Input Mixture  |  DiffTransfer | Music-STAR| Ground Truth|
|---|---|---|---|---|
| Pirates of The Carribean | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.3.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/mixture_model/001.0.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/mixture_model/001.0.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.0.wav" type="audio/mpeg"></audio> |
| My Heart Will Go On | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.3.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/mixture_model/002.0.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/mixture_model/002.0.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.0.wav" type="audio/mpeg"></audio> |
| Beethoven's String | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.3.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/mixture_model/003.0.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/mixture_model/003.0.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.0.wav" type="audio/mpeg"></audio> |

# Multi Instrument timbre transfer (single and single/mix)

## Clarinet/Vibraphone to Strings/Piano

Clarinet to Strings

|Name| Input Clarinet  |  DiffTransfer | Universal Network| Ground Truth Strings|
|---|---|---|---|---|
| Pirates of The Carribean | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.1.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model_individual_tracks/001.4.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model_individual_tracks/001.4.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.4.wav" type="audio/mpeg"></audio>|
| My Heart Will Go On | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.1.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model_individual_tracks/002.4.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model_individual_tracks/002.4.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.4.wav" type="audio/mpeg"></audio>|
| Beethoven's String | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.1.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model_individual_tracks/003.4.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model_individual_tracks/003.4.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.4.wav" type="audio/mpeg"></audio>|

Vibraphone to Piano

|Name| Input Vibraphone  |  DiffTransfer | Universal Network| Ground Truth Piano|
|---|---|---|---|---|
| Pirates of The Carribean | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.2.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model_individual_tracks/001.5.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model_individual_tracks/001.5.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.5.wav" type="audio/mpeg"></audio> |
| My Heart Will Go On | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.2.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model_individual_tracks/002.5.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model_individual_tracks/002.5.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.5.wav" type="audio/mpeg"></audio> |
| Beethoven's String | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.2.wav" type="audio/mpeg"></audio>  | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model_individual_tracks/003.5.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model_individual_tracks/003.5.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.5.wav" type="audio/mpeg"></audio> |

Clarinet/Vibraphone to Piano/Strings (single/mix)

|Name| DiffTransfer | Universal Network| Ground Truth Piano/Strings|
|---|---|---|---|
| Pirates of The Carribean | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model/001.3.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model/001.3.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/001.3.wav" type="audio/mpeg"></audio> |
| My Heart Will Go On | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model/002.3.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model/002.3.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/002.3.wav" type="audio/mpeg"></audio> |
| Beethoven's String | <audio controls style="width: 100px;"><source src="audio/est_diffusion/separate_model/003.3.wav" type="audio/mpeg"></audio>  |  <audio controls style="width: 100px;"><source src="audio/est_music_net/separate_model/003.3.wav" type="audio/mpeg"></audio> | <audio controls style="width: 100px;"><source src="audio/music_star_test/003.3.wav" type="audio/mpeg"></audio> |

