# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# dictionary to translate between benchmarks and plot labels
labels_translation = {
    'bm_num_parameters': 'lots of params',
    'bm_num_gates': 'deep circuit',
    'bm_num_qubits': 'wide circuit',
    'bm_small': 'small circuit',
    'bm_large': 'large circuit',
    'bm_iqp': 'iqp circuit'
}

# load the benchmark results
data = np.loadtxt('pl_xxx_device_comparison.csv', delimiter=',', dtype=str,
                  converters={0: lambda s: s.strip(),
                              1: lambda s: s.strip(),
                              2: lambda s: float(s.strip())})

# extract devices
devices = list(set(data[:, 0]))
# extract benchmark categories
categories = list(set(data[:, 1]))

# get a list of the slowest time per category
max_times = []
for category in categories:
    data_for_category = data[data[:, 1] == category]
    times_for_category = data_for_category[:, 2]
    slowest = np.max([float(t) for t in times_for_category])
    max_times.append(slowest+0.05)

for device in devices:
    # extract benchmark data for this device
    device_data = data[data[:, 0] == device]
    device_timings = []

    for max_time, category in zip(max_times, categories):
        # fill in -1 if category does not exist
        if category not in device_data[:, 1]:
            device_timings.append(-1)
        else:
            # find index of category
            timing = device_data[device_data[:, 1] == category][0]
            timing = float(timing[2])/max_time
            device_timings.append(1-timing)

    plt.cla()  # clear all
    ax = plt.subplot(111, polar=True)

    # repeat first value to close cycle
    device_timings += device_timings[:1]

    # angle of axes (divide plot by number of categories)
    angles = [n / len(categories) * 2 * pi + 0.5 for n in range(len(categories))]
    angles += angles[:1]

    # make labels
    labels = [labels_translation[category] if category in labels_translation else category for category in categories]

    plt.xticks(angles[:-1], labels, color='grey', size=15)
    ax.set_rlabel_position(0)
    plt.yticks([0.33, 0.66, 1], ["", "", ""], color="grey", size=10, style='italic')
    plt.ylim(0, 1)

    plt.title(device, fontdict={'size': 15, 'style': 'italic', 'color':'grey'},
              loc='center', pad=40)

    ax.plot(angles, device_timings, linewidth=1, linestyle='solid')
    ax.fill(angles, device_timings, 'b', alpha=0.1)

    plt.tight_layout()

    plt.savefig("spiderplot_{}.png".format(device.replace(".", "_")))
