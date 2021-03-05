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
whole_events = [i.split('.')[0] for i in os.listdir('data/csv')]
#just testing some data
id = whole_events[2]
start= 0
end = -1
e = Event(id, start, end)
keys = ['Time (s)', ' Va', ' Vb', ' Vc', ' Ia', ' Ib', ' Ic', ' In']
yf, yf_mag_real, xf, start_index, N, T = e.fft_analyzer(keys[6])
#fig = e.show_detail()
#print(meta_event.loc[meta_event['EventId']==int(id)].values)
e.data.plot()
e.res().plot()
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

#%%
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
max_size = 0
#kshape clustering test
from kshape.core import kshape, zscore
test_events = whole_events[0:50]
I_ns = []
for ev in test_events:
    temp_In = list(Event(ev, start, end).data[' In'].values)
    if max_size < len(temp_In):
        max_size = len(temp_In)
for ev in test_events:
    temp_In = list(Event(ev, start, end).data[' In'].values)
    zero_pad = [0] * max_size
    zero_pad[0:len(temp_In)] = temp_In
    I_ns.append(zero_pad)

#%%
causes = pd.read_pickle('data/causes.pkl')
def cluster_show(cluster_rep, cluster_id):
    #plt.plot(cluster_rep)
    for i in cluster_id:
        ev = causes.iloc[i]['id']
        temp_In = list(Event(ev, start, end).data[' In'].values)
        plt.plot(temp_In)
    plt.legend(list(causes.iloc[cluster_id]['cause']))
    plt.show()

cluster_num = 6
clusters = kshape(zscore(I_ns, axis=1), cluster_num)
for i in range(cluster_num):
    print(causes.iloc[clusters[i][1]],'\n','----------------------')
    cluster_show(clusters[i][0], clusters[i][1])


#%%
import statsmodels.api as sm

dta = sm.datasets.co2.load_pandas().data
# deal with missing values. see issue
dta.co2.interpolate(inplace=True)

id = whole_events[100]
start= 0
end = -1
e = Event(id, start, end)
res = sm.tsa.seasonal_decompose(e.data[' Ib'])
resplot = res.plot()
#%%
id = whole_events[2]

start= 0
end = -1
e = Event(id, start, end, 'downsampled')
from scipy.signal import hilbert
analytic_signal = hilbert(get_noisy(e.data[' Ib']))
amplitude_envelope = np.abs(analytic_signal)

plt.plot(get_noisy(e.data[' Ib']))
plt.plot(amplitude_envelope)
plt.show()
