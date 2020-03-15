"""Split of data and structure to hold the graph dataset."""
import os
import numpy as np
from sklearn.model_selection import train_test_split


class ProtienGraphDataset:
    """Build protein graph dataset, reading IO at index time."""

    def __init__(
        self, data, nb_nodes=185, task_type="classification", nb_classes=2
    ):
        """Initialize object.

        Default values correspond to the KrasHras experiment

        Parameters
        ----------
        data: np.array of arrays
            each array is an instance: [0] is a path to and [1] is the label.
        nb_nodes: int
            max size of graph (input will be padded to that size). Default: 185
        task_type: str
            "classification" or "regression". Default: "classification"
        nb_classes: 2
            number of classes

        """
        self.data = data
        self.nb_nodes = nb_nodes
        self.nb_classes = nb_classes
        self.task_type = task_type

        self.ident = np.eye(nb_nodes)

    def __getitem__(self, index):
        """Return index operator.

        Return
        ------
        data: list of np.arrays
            v: np.array of 3 np.arrays
                one-hot encoding of aminoacid (length 23),
                sidechain info (residue_depth and residue_orientation)
                sinuisoidal tranformation about position
            c: np.array
                centered coordinates of aminoacid (x,y,z)
            m: np.array (matrix)
                mask
            y: list
                one-hot encoding of label ("classification") or [label]

        """
        # Parse Protein Graph
        data_ = []
        with open(self.data[index][0], "r") as f:
            for i, _ in enumerate(f):
                if i >= self.nb_nodes:
                    break
                data_.append(_)

        v = []
        v_ = []
        s = []
        c = []
        m = []
        for i, line in enumerate(data_):
            if i >= self.nb_nodes:
                break
            row = line[:-1].split()
            res = np.zeros(23)
            res[int(row[2])] = 1
            v.append(res)
            v_.append(row[3:5])
            c.append(row[-3:])
            s.append(int(row[1]))
            m.append(int(row[0]))
        del data_
        v = np.array(v, dtype=float)
        v_ = np.array(v_, dtype=float)
        c = np.array(c, dtype=float)
        c = c - c.mean(axis=0)  # Center on origin
        m = np.array(m, dtype=float)

        # Sequence Encoding
        # s = np.array(list(range(len(v))), dtype=int)
        p = self.sequence_encode(s, 4)
        v = np.concatenate([v, v_, p], axis=-1)

        # Zero Padding
        if v.shape[0] < self.nb_nodes:
            v_ = np.zeros((self.nb_nodes, v.shape[1]))
            v_[: v.shape[0], : v.shape[1]] = v
            c_ = np.zeros((self.nb_nodes, c.shape[1]))
            c_[: c.shape[0], : c.shape[1]] = c
            m_ = np.zeros((self.nb_nodes))
            m_[: m.shape[0]] = m
            v = v_
            c = c_
            m = m_

        # Set MasK
        m = np.repeat(np.expand_dims(m, axis=-1), len(m), axis=-1)
        m = (m * m.T) + self.ident
        m[m > 1] = 1

        if self.task_type == "classification":
            y = [0 for _ in range(self.nb_classes)]
            y[int(self.data[index][1])] = 1
        elif self.task_type == "regression":
            y = [float(self.data[index][1])]
        else:
            raise Exception("Task Type %s unknown" % self.task_type)

        data_ = [v, c, m, y]

        return data_

    def __len__(self):
        """Retrieve length of data."""
        return len(self.data)

    def sequence_encode(self, seq, nb_dims):
        """Transform position index.

        Feature vector of shape (seq_len, nb_dims) which encodes positional
        information of each index in a sequence using sinisodal functions.

        Paramseters
        -----------
            seq_len: int32
                Length of sequence
            nb_dims:int32
                Number of dimensions used to encode position

        Returns
        -------
            sequence_enc: np.array(seq_len, nb_dims)
                Sequential encoding

        """
        sequence_enc = np.array(
            [
                [
                    pos / np.power(10000, 2 * (j // 2) / nb_dims)
                    for j in range(nb_dims)
                ]
                if pos != 0
                else np.zeros(nb_dims)
                for pos in seq
            ]
        )
        sequence_enc[1:, 0::2] = np.sin(sequence_enc[1:, 0::2])  # dim 2i
        sequence_enc[1:, 1::2] = np.cos(sequence_enc[1:, 1::2])  # dim 2i+1

        return sequence_enc


def get_datasets(
    data_path,
    nb_nodes,
    task_type,
    nb_classes,
    split=None,
    k_fold=None,
    seed=1234,
):
    """Generate train/test/validation splits for proein graph data.

    Parameters
    ----------
    data_path: str
        path to data directory (generated by ./generate.py)
    nb_nodes:
        maximum length of aminoacids in protein
    task_type: str
        "classification" or "regression"
    nb_classes: 2
        number of classes
    split: list
        portion of train/test/validation. Default: [0.7, 0.1, 0.2]
    k_fold: None
    seed: int

    Returns
    -------
    (train_dataset, valid_dataset, test_dataset): ProtienGraphDataset

    """
    if split is None:
        split = [0.7, 0.1, 0.2]
    # Load examples
    X = []
    Y = []
    with open(data_path + "/data.csv", "r") as f:
        for line in f:
            row = line[:-1].split(",")
            pdb_id = row[0].lower()
            chain_id = row[1].lower()
            filename = pdb_id + "_" + chain_id + ".txt"
            if not os.path.exists(data_path + "/graph/" + filename):
                continue
            X.append(data_path + "/graph/" + filename)
            Y.append(row[2])
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)

    # Get Class Compossition
    unique, counts = np.unique(Y, return_counts=True)
    for i in range(len(unique)):
        print("CLASS[COUNTS]: ", unique[i], counts[i])

    if k_fold is not None:
        # Split into K Folds and return training, validation and test
        np.random.seed(seed)
        data = np.concatenate([X, Y], axis=-1)
        np.random.shuffle(data)
        fs = len(data) // int(k_fold[0])
        ind = [fs * (i + 1) for i in range(len(data) // fs)]
        remainder = len(data) % fs
        for i in range(remainder):
            for j in range(i % len(ind) + 1):
                ind[-(j + 1)] += 1
        folds = np.split(data.copy(), ind, axis=0)
        data_test = folds.pop(int(k_fold[1]))
        data_train = np.concatenate(folds, axis=0)
        x_train, x_valid, y_train, y_valid = train_test_split(
            data_train[:, 0:1],
            data_train[:, 1:],
            test_size=float(k_fold[-1]),
            random_state=seed,
        )
        data_train = np.concatenate([x_train, y_train], axis=-1)
        data_valid = np.concatenate([x_valid, y_valid], axis=-1)
    else:
        # Split Examples
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=split[2], random_state=seed
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train,
            y_train,
            test_size=split[1] / (split[0] + split[1]),
            random_state=seed,
        )
        data_train = np.concatenate([x_train, y_train], axis=-1)
        data_test = np.concatenate([x_test, y_test], axis=-1)
        data_valid = np.concatenate([x_valid, y_valid], axis=-1)

    # Initialize Dataset Iterators
    train_dataset = ProtienGraphDataset(
        data_train, nb_nodes, task_type, nb_classes,
    )
    valid_dataset = ProtienGraphDataset(
        data_valid, nb_nodes, task_type, nb_classes
    )
    test_dataset = ProtienGraphDataset(
        data_test, nb_nodes, task_type, nb_classes
    )

    return train_dataset, valid_dataset, test_dataset
