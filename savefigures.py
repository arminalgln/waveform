#%%
#saving figures just for visual reference
plt.show()
for ev in whole_events:
    e = Event(ev)
    path_to_save = "figures/"+ ev +"_current.jpg"
    current = e.keys[5:]
    fig = e.show_event(current)
    fig.savefig(path_to_save)
    plt.show()

    path_to_save = "figures/"+ ev +"_voltage.jpg"
    voltage = e.keys[2:5]
    fig = e.show_event(voltage)
    fig.savefig(path_to_save)
    plt.show()