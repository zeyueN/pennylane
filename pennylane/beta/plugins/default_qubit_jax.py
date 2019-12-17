# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Default qubit plugin with support for XLA and device jacobian
via JAX.
"""
from collections import OrderedDict
import itertools
import warnings

import numpy as onp
import numpy as np

from pennylane import Device, DeviceError
from pennylane.operation import Operation, Sample


# tolerance for numerical errors
tolerance = 1e-10

#========================================================
#  fixed gates
#========================================================

I = np.eye(2)
# Pauli matrices
X = np.array([[0, 1], [1, 0]]) #: Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]]) #: Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]]) #: Pauli-Z matrix

H = np.array([[1, 1], [1, -1]])/np.sqrt(2) #: Hadamard gate
# Two qubit gates
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) #: CNOT gate
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) #: SWAP gate
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]) #: CZ gate
S = np.array([[1, 0], [0, 1j]]) #: Phase Gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]) #: T Gate
# Three qubit gates
CSWAP = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]]) #: CSWAP gate

Toffoli = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1, 0]]) #: CSWAP gate

II = np.identity(4)
ZZ = np.kron(Z, Z)

IX = np.kron(I, X)
IY = np.kron(I, Y)
IZ = np.kron(I, Z)

ZI = np.kron(Z, I)
ZX = np.kron(Z, X)
ZY = np.kron(Z, Y)

#========================================================
#  parametrized gates
#========================================================

def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle
    Returns:
        array: unitary 2x2 phase shift matrix
    """
    return ((1 + np.exp(1j * phi)) * I + (1 - np.exp(1j * phi)) * Z) / 2


def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return np.cos(theta/2) * I + 1j * np.sin(-theta/2) * X


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return np.cos(theta/2) * I + 1j * np.sin(-theta/2) * Y


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return np.cos(theta/2) * I + 1j * np.sin(-theta/2) * Z


def Rot3(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return Rotz(c) @ (Roty(b) @ Rotz(a))


def CRotx(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    return (
        np.cos(theta / 4) ** 2 * II
        - 1j * np.sin(theta / 2) / 2 * IX
        + np.sin(theta / 4) ** 2 * ZI
        + 1j * np.sin(theta / 2) / 2 * ZX
    )


def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    return (
        np.cos(theta / 4) ** 2 * II
        - 1j * np.sin(theta / 2) / 2 * IY
        + np.sin(theta / 4) ** 2 * ZI
        + 1j * np.sin(theta / 2) / 2 * ZY
    )


def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    return (
        np.cos(theta / 4) ** 2 * II
        - 1j * np.sin(theta / 2) / 2 * IZ
        + np.sin(theta / 4) ** 2 * ZI
        + 1j * np.sin(theta / 2) / 2 * ZZ
    )


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return CRotz(c) @ (CRoty(b) @ CRotz(a))


#========================================================
#  Arbitrary states and operators
#========================================================

def unitary(*args):
    r"""Input validation for an arbitary unitary operation.

    Args:
        args (array): square unitary matrix

    Raises:
        ValueError: if the matrix is not unitary or square

    Returns:
        array: square unitary matrix
    """
    U = np.asarray(args[0])

    if U.shape[0] != U.shape[1]:
        raise ValueError("Operator must be a square matrix.")

    if not np.allclose(U @ U.conj().T, np.identity(U.shape[0])):
        raise ValueError("Operator must be unitary.")

    return U


def hermitian(*args):
    r"""Input validation for an arbitary Hermitian expectation.

    Args:
        args (array): square hermitian matrix

    Raises:
        ValueError: if the matrix is not Hermitian or square

    Returns:
        array: square hermitian matrix
    """
    A = np.asarray(args[0])

    if A.shape[0] != A.shape[1]:
        raise ValueError("Expectation must be a square matrix.")

    if not np.allclose(A, A.conj().T):
        raise ValueError("Expectation must be Hermitian.")

    return A


def identity(*_):
    """Identity matrix observable.

    Returns:
        array: 2x2 identity matrix
    """
    return np.identity(2)

#========================================================
#  device
#========================================================


def pauli_eigs(n):
    r"""Returns the eigenvalues for :math:`A^{\otimes n}`,
    where :math:`A` is any operator that shares eigenvalues
    with the Pauli matrices.

    Args:
        n (int): number of wires

    Returns:
        array[int]: eigenvalues of :math:`Z^{\otimes n}`
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([pauli_eigs(n - 1), -pauli_eigs(n - 1)])


def mat_vec_product(num_wires, mat, vec, wires):
    r"""Apply multiplication of a matrix to subsystems of the quantum state.

    Args:
        mat (array): matrix to multiply
        vec (array): state vector to multiply
        wires (Sequence[int]): target subsystems

    Returns:
        array: output vector after applying ``mat`` to input ``vec`` on specified subsystems
    """

    # TODO: use multi-index vectors/matrices to represent states/gates internally
    mat = np.reshape(mat, [2] * len(wires) * 2)
    vec = np.reshape(vec, [2] * num_wires)
    axes = (np.arange(len(wires), 2 * len(wires)), wires)
    tdot = np.tensordot(mat, vec, axes=axes)

    # tensordot causes the axes given in `wires` to end up in the first positions
    # of the resulting tensor. This corresponds to a (partial) transpose of
    # the correct output state
    # We'll need to invert this permutation to put the indices in the correct place
    unused_idxs = [idx for idx in range(num_wires) if idx not in wires]
    perm = wires + unused_idxs
    inv_perm = np.argsort(perm) # argsort gives inverse permutation
    state_multi_index = np.transpose(tdot, inv_perm)
    return np.reshape(state_multi_index, 2 ** num_wires)


class DefaultQubitJAX(Device):
    """Default qubit device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to 1000 if not specified.
            If ``analytic == True``, then the number of shots is ignored
            in the calculation of expectation values and variances, and only controls the number
            of samples returned by ``sample``.
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
    """
    name = 'Default qubit JAX PennyLane plugin'
    short_name = 'default.qubit.jax'
    pennylane_requires = '0.8'
    version = '0.8.0'
    author = 'Xanadu Inc.'
    _capabilities = {"model": "qubit", "tensor_observables": True, "inverse_operations": True}

    # Note: BasisState and QubitStateVector don't
    # map to any particular function, as they modify
    # the internal device state directly.
    _operation_map = {
        'BasisState': None,
        'QubitStateVector': None,
        'QubitUnitary': unitary,
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hadamard': H,
        'S': S,
        'T': T,
        'CNOT': CNOT,
        'SWAP': SWAP,
        'CSWAP': CSWAP,
        'Toffoli': Toffoli,
        'CZ': CZ,
        'PhaseShift': Rphi,
        'RX': Rotx,
        'RY': Roty,
        'RZ': Rotz,
        'Rot': Rot3,
        'CRX': CRotx,
        'CRY': CRoty,
        'CRZ': CRotz,
        'CRot': CRot3
    }

    _observable_map = {
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hadamard': H,
        'Hermitian': hermitian,
        'Identity': identity
    }

    _eigs = {}

    def __init__(self, wires, *, shots=1000, analytic=True):
        super().__init__(wires, shots)
        self.eng = None
        self.analytic = analytic

        self._state = None
        self._probability = None
        self._first_operation = True
        self._memory = False

    def pre_apply(self):
        self.reset()

    def apply(self, operation, wires, par):
        # number of wires on device
        n = self.num_wires

        if operation == 'QubitStateVector':
            input_state = np.asarray(par[0], dtype=np.complex128)

            if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
                raise ValueError("Sum of amplitudes-squared does not equal one.")
            n_state_vector = input_state.shape[0]
            if not self._first_operation:
                raise DeviceError("Operation {} cannot be used after other Operations have already been applied "
                                  "on a {} device.".format(operation, self.short_name))
            if input_state.ndim == 1 and n_state_vector == 2**len(wires):

                # generate basis states on subset of qubits via the cartesian product
                tuples = np.array(list(itertools.product([0, 1], repeat=len(wires))))

                # get basis states to alter on full set of qubits
                unravelled_nums = np.zeros((2 ** len(wires), n), dtype=int)
                unravelled_nums[:, wires] = tuples

                # get indices for which the state is changed to input state vector elements
                nums = np.ravel_multi_index(unravelled_nums.T, [2] * n)
                self._state = onp.zeros_like(self._state)
                self._state[nums] = input_state
            else:
                raise ValueError("State vector must be of length 2**wires.")
            self._first_operation = False
            return

        if operation == 'BasisState':
            # length of basis state parameter
            n_basis_state = len(par[0])

            if not set(par[0]).issubset({0, 1}):
                raise ValueError("BasisState parameter must consist of 0 or 1 integers.")
            if n_basis_state != len(wires):
                raise ValueError("BasisState parameter and wires must be of equal length.")
            if not self._first_operation:
                raise DeviceError("Operation {} cannot be used after other Operations have already been applied "
                                  "on a {} device.".format(operation, self.short_name))


            # get computational basis state number
            num = int(np.dot(par[0], 2**(n - 1 - np.array(wires))))

            self._state = onp.zeros_like(self._state)
            self._state[num] = 1.
            self._first_operation = False
            return

        A = self._get_operator_matrix(operation, par)
        self._state = mat_vec_product(self.num_wires, A, self._state, wires)
        self._first_operation = False

    def rotate_basis(self, obs, wires, par):
        """Rotates the specified wires such that they
        are in the eigenbasis of the provided observable.

        Args:
            observable (str): the name of an observable
            wires (List[int]): wires the observable is measured on
            par (List[Any]): parameters of the observable
        """
        if obs == "PauliX":
            # X = H.Z.H
            self.apply("Hadamard", wires=wires, par=[])

        elif obs == "PauliY":
            # Y = (HS^)^.Z.(HS^) and S^=SZ
            self.apply("PauliZ", wires=wires, par=[])
            self.apply("S", wires=wires, par=[])
            self.apply("Hadamard", wires=wires, par=[])

        elif obs == "Hadamard":
            # H = Ry(-pi/4)^.Z.Ry(-pi/4)
            self.apply("RY", wires, [-np.pi / 4])

        elif obs == "Hermitian":
            # For arbitrary Hermitian matrix H, let U be the unitary matrix
            # that diagonalises it, and w_i be the eigenvalues.
            Hmat = par[0]
            Hkey = tuple(Hmat.flatten().tolist())

            if Hkey in self._eigs:
                # retrieve eigenvectors
                U = self._eigs[Hkey]["eigvec"]
            else:
                # store the eigenvalues corresponding to H
                # in a dictionary, so that they do not need to
                # be calculated later
                w, U = np.linalg.eigh(Hmat)
                self._eigs[Hkey] = {"eigval": w, "eigvec": U}

            # Perform a change of basis before measuring by applying U^ to the circuit
            self.apply("QubitUnitary", wires, [U.conj().T])

    def pre_measure(self):
        self._probability = np.abs(self._state)**2

        for e in self.obs_queue:
            if hasattr(e, "return_type") and e.return_type == Sample:
                self._memory = True  # make sure to return samples

            # Add unitaries if a different expectation value is given
            if isinstance(e.name, list):
                # tensor product
                for n, w, p in zip(e.name, e.wires, e.parameters):
                    self.rotate_basis(n, w, p)
            else:
                # single wire observable
                self.rotate_basis(e.name, e.wires, e.parameters)

        self.rotated_probability = np.abs(self._state)**2

        # generate computational basis samples
        if self._memory:
            basis_states = np.arange(2**self.num_wires)
            self._samples = np.random.choice(basis_states, self.shots, p=self.rotated_probability)

    def marginal_prob(self, prob, wires=None):
        wires = wires or range(self.num_wires)
        wires = np.hstack(wires)
        inactive_wires = list(set(range(self.num_wires)) - set(wires))
        prob = prob.reshape([2] * self.num_wires)
        return np.apply_over_axes(np.sum, prob, inactive_wires).flatten()

    def expval(self, observable, wires, par):
        if self.analytic:
            # exact expectation value
            eigvals = self.eigvals(observable, wires, par)
            prob = self.marginal_prob(self.rotated_probability, wires=wires)
            return (eigvals @ prob).real

        # estimate the ev
        return np.mean(self.sample(observable, wires, par))

    def var(self, observable, wires, par):
        if self.analytic:
            # exact variance value
            eigvals = self.eigvals(observable, wires, par)
            prob = self.marginal_prob(self.rotated_probability, wires=wires)
            return (eigvals ** 2) @ prob - (eigvals @ prob).real ** 2

        return np.var(self.sample(observable, wires, par))

    def sample(self, observable, wires, par):
        comp_basis_samples = (((self._samples[:,None] & (1 << np.arange(2**self.num_wires)))) > 0).astype(int)

        if isinstance(observable, str) and observable in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            return 1 - 2 * comp_basis_samples[:, wires[0]]

        eigvals = self.eigvals(observable, wires, par)
        wires = np.hstack(wires)
        res = comp_basis_samples[:, np.array(wires)]
        comp_basis_samples = np.zeros([self.shots])

        for w, b in zip(eigvals, itertools.product([0, 1], repeat=len(wires))):
            comp_basis_samples = np.where(np.all(res == b, axis=1), w, comp_basis_samples)

        return comp_basis_samples

    def _get_operator_matrix(self, operation, par):
        """Get the operator matrix for a given operation or observable.

        If the inverse was defined for an operation, returns the
        conjugate transpose of the operator matrix.

        Args:
          operation    (str): name of the operation/observable
          par (tuple[float]): parameter values
        Returns:
          array: matrix representation.
        """

        operation_map = {**self._operation_map, **self._observable_map}

        if operation.endswith(Operation.string_for_inverse):
            A = operation_map[operation[:-len(Operation.string_for_inverse)]]
            return A.conj().T if not callable(A) else A(*par).conj().T

        A = operation_map[operation]
        return A if not callable(A) else A(*par)

    def ev(self, A, wires):
        r"""Expectation value of observable on specified wires.

         Args:
            A (array[float]): the observable matrix as array
            wires (Sequence[int]): target subsystems
         Returns:
            float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        As = mat_vec_product(self.num_wires, A, self._state, np.hstack(wires).tolist())
        expectation = np.vdot(self._state, As)
        return expectation.real

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        self._probability = None
        self._state = onp.zeros(2**self.num_wires, dtype=complex)
        self._state[0] = 1
        self._first_operation = True
        self._memory = False

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())

    def probability(self, wires=None, values_only=False):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            OrderedDict[tuple, float]: Dictionary mapping a tuple representing the state
            to the resulting probability. The dictionary should be sorted such that the
            state tuples are in lexicographical order.
        """
        if self._state is None:
            return None

        prob = self.marginal_prob(self.probability, wires)
        basis_states = itertools.product(range(2), repeat=len(wires))
        return OrderedDict(zip(basis_states, prob))

    def eigvals(self, observable, wires, par):
        """Determine the eigenvalues of observable(s).

        Args:
            observable (str, List[str]): the name of an observable,
                or a list of observables representing a tensor product
            wires (List[int]): wires the observable(s) is measured on
            par (List[Any]): parameters of the observable(s)

        Returns:
            array[float]: an array of size ``(len(wires),)`` containing the
            eigenvalues of the observable
        """
        # the standard observables all share a common eigenbasis {1, -1}
        # with the Pauli-Z gate/computational basis measurement
        standard_observables = {"PauliX", "PauliY", "PauliZ", "Hadamard"}

        # observable should be Z^{\otimes n}
        eigvals = pauli_eigs(len(wires))

        if isinstance(observable, list):
            # tensor product of observables

            # check if there are any non-standard observables (such as Identity, Hadamard)
            if set(observable) - standard_observables:
                # Tensor product of observables contains a mixture
                # of standard and non-standard observables
                eigvals = np.array([1])

                # group the observables into subgroups, depending on whether
                # they are in the standard observables or not.
                for k, g in itertools.groupby(
                    zip(observable, wires, par), lambda x: x[0] in standard_observables
                ):
                    if k:
                        # Subgroup g contains only standard observables.
                        # Determine the size of the subgroup, by transposing
                        # the list, flattening it, and determining the length.
                        n = len([w for sublist in list(zip(*g))[1] for w in sublist])
                        eigvals = np.kron(eigvals, pauli_eigs(n))
                    else:
                        # Subgroup g contains only non-standard observables.
                        for ns_obs in g:
                            # loop through all non-standard observables
                            if ns_obs[0] == "Hermitian":
                                # Hermitian observable has pre-computed eigenvalues
                                p = ns_obs[2]
                                Hkey = tuple(p[0].flatten().tolist())
                                eigvals = np.kron(eigvals, self._eigs[Hkey]["eigval"])

                            elif ns_obs[0] == "Identity":
                                # Identity observable has eigenvalues (1, 1)
                                eigvals = np.kron(eigvals, np.array([1, 1]))

        elif observable == "Hermitian":
            # single wire Hermitian observable
            Hkey = tuple(par[0].flatten().tolist())
            eigvals = self._eigs[Hkey]["eigval"]

        elif observable == "Identity":
            # single wire identity observable
            eigvals = np.ones(2 ** len(wires))

        return eigvals
