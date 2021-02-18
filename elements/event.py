
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


class Event():
    def __init__(self, id, start, end):
        """
        :param id: is an string which shows the event id and file
        """
        self.id = id
        self.start = start
        self.end = end
        self.path = "data/csv/{}.csv".format(id)
        self.data = pd.read_csv(self.path)[start:end]
        self.data.rename(columns={'Unnamed: 0': 'id'})
        self.keys = self.data.keys()


    def show_event(self, selected_keys):
        """
        :param selected_keys: selected feature/s by user to show
        """
        for k in selected_keys:
            plt.plot(self.data[k])
        plt.legend(selected_keys)
        plt.show()

    def fft_analyzer(self, selected_key):
        """
        :param selected_key: main feature for fft analysis
        """
        # Number of sample points
        N = self.data.shape[0]
        # sample spacing
        T = self.data['Time (s)'][1] - self.data['Time (s)'][0]
        start_index = [i for i, c in enumerate(self.data['Time (s)']) if np.abs(c) == 0]
        x = np.linspace(0.0, N * T, N, endpoint=False)
        y = self.data[selected_key].values
        yf = fft(y)
        xf = fftfreq(N, T)[:N // 2]
        yf_mag_real = 2.0 / N * np.abs(yf[0:N // 2])

        return yf, yf_mag_real, xf, start_index, N, T

    def show_detail(self):
        current = self.keys[5:]
        voltage = self.keys[2:5]
        main_keys = self.keys[2:]
        # fft analyze for each event

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True)
        for k in main_keys:
            if k in voltage:
                ax0.plot(self.data[k])
                yf, yf_mag_real, xf, start_index, N, T = self.fft_analyzer(k)
                ax1.plot(xf, yf_mag_real)
            if k in current:
                ax2.plot(self.data[k])
                yf, yf_mag_real, xf, start_index, N, T = self.fft_analyzer(k)
                ax3.plot(xf, yf_mag_real)
        ax0.set_xlabel('time [s]')
        ax0.set_ylabel('voltage magnitude')
        ax0.legend(voltage)

        ax1.set_xlabel('freq')
        ax1.set_ylabel('magnitude (wrt voltage)')
        ax1.legend(voltage)

        ax2.set_xlabel('time [s]')
        ax2.set_ylabel('current magnitude')
        ax2.legend(current)

        ax3.set_xlabel('freq')
        ax3.set_ylabel('magnitude (wrt current)')
        ax3.legend(current)
        return fig