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
#%%

#saving figures just for visual reference
causes = pd.read_pickle('data/causes.pkl')
for ev in whole_events[0:1]:
    path = "figures/eventshape/{}.png".format(ev)
    e = Event(ev, start, end)
    fig = e.show_detail()
    cause = causes.loc[causes['id'] == ev]['cause'].values[0]
    plt.title("{}".format(cause))
    plt.show()
    fig.savefig(path)
#%%
#save figures by their known group
causes = pd.read_pickle('data/causes.pkl')
causes = causes.groupby('cause')
all_causes = causes.groups.keys()
for c in all_causes:
    path = 'figures/knownclusters/{}'.format(c)
    os.mkdir(path)
    for ev in causes.get_group(c)['id']:
        e = Event(ev,0,-1)
        fig = e.show_detail()
        cause = c
        plt.title("{}".format(cause))
        plt.show()
        save_path = path + '/{}.png'.format(ev)
        fig.savefig(save_path)
#%%
#known event true figures
def show_event(ev, cl):
    causes = pd.read_pickle('data/causes.pkl')
    e = Event(ev, 0, -1, 'resampled')
    path = 'figures/eval_clusters/{}/{}.png'.format(cl, ev)
    selected_data = e.data.loc[:, e.data.columns != 'Time (s)']
    f, (ax1, ax2) = plt.subplots(2, 1)
    for i in selected_data.keys()[0:3]:
        ax1.plot(selected_data[i], label=i)
    for i in selected_data.keys()[3:]:
        ax2.plot(selected_data[i], label=i)

        ax1.set_title(ev)
    # plt.legend()
    f.savefig(path)
    plt.show()


    #%%
# read the main dataset
true_clusters_known = pd.read_pickle('data/known_true_clusters_ids.pkl')
all_clusters_data = []
cl = 'Equipment'
for cl in true_clusters_known.keys():
    # os.mkdir('figures/eval_clusters/{}'.format(cl))
    for ev in true_clusters_known[cl].dropna():
        show_event(ev, cl)


#%%
#known event true figures
def show_event_res(ev, cl):
    causes = pd.read_pickle('data/causes.pkl')
    e = Event(ev, 0, -1, 'resampled')
    path = 'figures/eval_res/{}/{}.png'.format(cl, ev)
    selected_data = e.res().loc[:, e.res().columns != 'Time (s)']
    f, (ax1, ax2) = plt.subplots(2, 1)
    for i in selected_data.keys()[0:3]:
        ax1.plot(selected_data[i], label=i)
    for i in selected_data.keys()[3:]:
        ax2.plot(selected_data[i], label=i)

        ax1.set_title(ev)
    # plt.legend()
    f.savefig(path)
    plt.show()
    #%%
# read the main dataset with res
true_clusters_known = pd.read_pickle('data/known_true_clusters_ids.pkl')
all_clusters_data = []
cl = 'Tree'
for cl in true_clusters_known.keys():
    # os.mkdir('figures/eval_res/{}'.format(cl))
    for ev in true_clusters_known[cl].dropna():
        show_event_res(ev, cl)

#%%
bad_events_horizon = {
    'Animal': ['21831','21832','21833','21834','21840','21842','21843'],
    'Equipment': ['21835','21836','21837','21841','21850','21862','21863','21865','21873'],
    'Lightning': ['21856','21857','21858','21859','21860','21861'],
    'Animal': ['2771','21838','21839','21844','21845','21846','21847','21848','21851','21852','21853','21854']
                      }

flat_list = [item for sublist in list(bad_events_horizon.values()) for item in sublist]
# matplotlib.use("Qt5Agg")
plt.ion()
# def onclick(event):
#     global pause
#     pause = not pause
#%%
#known event true figures
def show_bad_event(ev):
    causes = pd.read_pickle('data/causes.pkl')
    e = Event(ev, 0, -1, 'resampled')
    # path = 'figures/eval_clusters/{}/{}.png'.format(cl, ev)
    selected_data = e.data.loc[:, e.data.columns != 'Time (s)']
    f, (ax1, ax2) = plt.subplots(2, 1)
    # f.canvas.mpl_connect('button_press_event', onclick)
    for i in selected_data.keys()[0:3]:
        ax1.plot(selected_data[i], label=i)
    for i in selected_data.keys()[3:]:
        ax2.plot(selected_data[i], label=i)

        ax1.set_title(ev)
    # plt.legend()
    # f.savefig(path)
    plt.show()
    # plt.waitforbuttonpress()
i=0
#%%
animals = ['21831','21832','21833','21834','21840','21842','21843']
ev = animals[i]
show_bad_event(ev)
i+=1
#%%
