from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(
    use_pca=False,
    use_activation=False,
    verbose=False
)

model_types = ['separate_model_individual_tracks','separate_model','mixture_model']
gt_dataset = "/nas/home/lcomanducci/music_txt/dataset/starnet/starnet_reduced"
for i in range(len(model_types)):
    fad_score_diff = frechet.score(gt_dataset, "/nas/home/lcomanducci/music_txt/eval/est_diffusion/"+model_types[i])
    fad_score_ms = frechet.score(gt_dataset, "/nas/home/lcomanducci/music_txt/eval/est_music_net/"+model_types[i])
    print('Model: '+model_types[i]+' FAD diffusion: '+str(fad_score_diff)+' FAD musicstar: '+str(fad_score_ms))


