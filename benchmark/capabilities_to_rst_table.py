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
"""Utility module to turn a list of device names into a table
comparing their capabilities.

The table is created by the tabular package. Supported table formats are:

    "plain"
    "simple"
    "github"
    "grid"
    "fancy_grid"
    "pipe"
    "orgtbl"
    "jira"
    "presto"
    "pretty"
    "psql"
    "rst"
    "mediawiki"
    "moinmoin"
    "youtrack"
    "html"
    "latex"
    "latex_raw"
    "latex_booktabs"
    "textile"

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow warnings

import pennylane as qml
try:
    import tabulate
except ModuleNotFoundError:
    raise ModuleNotFoundError("This module requires you to install the tabulate package, which is not part of the "
                              "PennyLane requirements.")

# Settings ====================================

DEVICES = ['default.qubit', 'default.qubit.tf']  # this may be imported from somewhere in future
TABLE_FORMAT = "rst"

# ==============================================

# translate capabilities to column headers in the table
# only capabilities listed will appear as columns
columns = {'model': 'Model',
           'tensor_observables': 'Tensor observables',
           'inverse_operations': 'Inverse operations',
           'reversible_diff': 'Reversible differentiation'
           }

data = []
for name in DEVICES:
    dev = qml.device(name, wires=1)
    dev_capabilities = dev.capabilities()

    dev_data = {'Device': name}
    for capability, new_key in columns.items():

        if capability in dev_capabilities:
            # rename the existing key
            dev_data[new_key] = dev_capabilities[capability]
        else:
            # add a new key
            dev_data[new_key] = 'n/a'

    data.append(dev_data)

header = data[0].keys()
rows = [x.values() for x in data]

print()
print(tabulate.tabulate(rows, header, tablefmt=TABLE_FORMAT))
print()
