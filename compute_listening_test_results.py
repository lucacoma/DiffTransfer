import pandas as pd
import numpy as np
#header = ['session_test_id,session_uuid,trial_id,rating_stimulus,rating_score,rating_time,rating_comment']
results = pd.read_csv('eval/listening_test_results.csv',delimiter = ',')
N_participants = len(set(results['session_uuid']))
Ids = list(set(results['session_uuid']))
print(str(N_participants)+' Participants')
print('')

# Single test
n_tracks = 4
results_c1 = np.zeros((N_participants,n_tracks))
results_c2 = np.zeros((N_participants,n_tracks))
for n_p in range(N_participants):
    result_participant = results[results['session_uuid']==Ids[n_p]]
    for i in range(n_tracks):
        result_trials = result_participant[result_participant['trial_id']=='track'+str(i+1)+'_single']
        results_c1[n_p,i] = result_trials[result_trials['rating_stimulus']=='C1']['rating_score']
        results_c2[n_p, i] = result_trials[result_trials['rating_stimulus'] == 'C2']['rating_score']
print('Single Instruments')
print('Diff rating: '+str(np.mean(results_c1)))
print('Universal Net rating: '+str(np.mean(results_c2)))


# Multi test
n_tracks = 4
results_c1_multi = np.zeros((N_participants,n_tracks))
results_c2_multi = np.zeros((N_participants,n_tracks))
results_c3_multi = np.zeros((N_participants,n_tracks))
results_c4_multi = np.zeros((N_participants,n_tracks))
for n_p in range(N_participants):
    result_participant = results[results['session_uuid']==Ids[n_p]]
    for i in range(n_tracks):
        result_trials = result_participant[result_participant['trial_id']=='track'+str(i+5)+'_multi']
        results_c1_multi[n_p,i] = result_trials[result_trials['rating_stimulus']=='C1']['rating_score']
        results_c2_multi[n_p, i] = result_trials[result_trials['rating_stimulus'] == 'C2']['rating_score']
        results_c3_multi[n_p,i] = result_trials[result_trials['rating_stimulus']=='C3']['rating_score']
        results_c4_multi[n_p, i] = result_trials[result_trials['rating_stimulus'] == 'C4']['rating_score']

print('')
print('Multi Instruments')
print('Diff (Separate) rating: '+str(np.mean(results_c1_multi)))
print('Universal Net rating: '+str(np.mean(results_c2_multi)))
print('Diff (Mix) rating: '+str(np.mean(results_c3_multi)))
print('MusicStar: '+str(np.mean(results_c4_multi)))


