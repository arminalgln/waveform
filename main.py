import numpy as np
import pandas as pd
import matplotlib
from importlib import reload
import matplotlib.pyplot as plt
import elements
elements = reload(elements)
from elements.event import Event
import os
from scipy.fft import fft, fftfreq, ifft

#%%
#meta data
meta_event = pd.read_csv('data/meta_data.csv')
#List of events
plt.ion()
whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]
#just testing some data
id = whole_events[0]
start= 0
end = -1
e = Event(id, start, end)
keys = ['Time (s)', ' Va', ' Vb', ' Vc', ' Ia', ' Ib', ' Ic', ' In']
yf, yf_mag_real, xf, start_index, N, T = e.fft_analyzer(keys[6])
fig = e.show_detail()
print(meta_event.loc[meta_event['EventId']==int(id)].values)
#%%
#get the fft for each event as the input features
whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]
number_of_freqs = 200
Nf = number_of_freqs * 7 # number of features
features = {}
max_voltage = 0
max_current = 0
bad_data_index= [122, 123, 124, 125, 127, 128]
unique_causes = meta_event['Cause'].unique()
bad_events_id = []
causes = pd.DataFrame(columns={'id','cause','label'})
for i, ev in enumerate(whole_events):

    print(i, ev)
    if i in bad_data_index:
        bad_events_id.append(ev)
    else:
        cause = meta_event.iloc[i]['Cause']
        label = [i for i, elem in enumerate(unique_causes) if elem == cause][0]
        new_row = {'id': ev, 'cause': cause, 'label': label}
        causes = causes.append(new_row, ignore_index=True)
        e = Event(ev, start, end)
        event_feature = np.array([])
        # for voltages and current append the fft features
        for idx, k in enumerate(keys[1:]):
            temp_feature = np.zeros(number_of_freqs, dtype=complex)
            yf, yf_mag_real, xf, start_index, N, T = e.fft_analyzer(k)
            #temp_feature[0:min(np.shape(yf)[0], 500)] = abs(yf[0:min(np.shape(yf)[0], 500)])
            temp_feature[0:min(np.shape(yf_mag_real)[0], number_of_freqs)] = yf_mag_real[0:min(np.shape(yf)[0], number_of_freqs)]
            event_feature = np.append(event_feature, temp_feature)

            #catch the max magnitude for currrent and voltage to normalize the features
            if idx <= 2:
                if max(yf_mag_real) > max_voltage:
                    max_voltage = max(yf_mag_real)
            else:
                if max(yf_mag_real) > max_current:
                    max_current = max(yf_mag_real)

        features[ev] = event_feature

for ev in features:
    features[ev][0:3*number_of_freqs] = features[ev][0:3*number_of_freqs]/max_voltage
    features[ev][3 * number_of_freqs:] = features[ev][3 * number_of_freqs] / max_current
#%%

causes.to_pickle('data/causes.pkl')

#%%
#save fft feature
features = pd.DataFrame(features)
features.to_pickle('data/fft_features_abs_clean_100.pkl')
#%%