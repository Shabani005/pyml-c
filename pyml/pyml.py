import os
import ctypes
import numpy as np
from numpy._core.numeric import dtype
from numpy._core.numerictypes import float32, int32
import pandas as pd

lib_path = os.path.join(os.path.dirname(__file__), "libcml.so")
lib = ctypes.CDLL(lib_path)

# lib = ctypes.CDLL("./libcml.so")


# structures

# class Point(ctypes.Structure):
#     _fields_ = [
#             ("features", ctypes.POINTER(ctypes.c_float)),
#             ("label", ctypes.c_int)
#             ]
#




class Sample(ctypes.Structure):
    _fields_ = [
            ("x", ctypes.POINTER(ctypes.c_float)),
            ("y", ctypes.POINTER(ctypes.c_float)),
            ("size", ctypes.c_size_t)
            ]


class Point(ctypes.Structure):
    _fields_ = [
            ("features", ctypes.POINTER(ctypes.c_float)),
            ("label", ctypes.c_int32)
            ]

class Dataset(ctypes.Structure):
    _fields_ = [
            ("points", ctypes.POINTER(Point)),
            ("size", ctypes.c_size_t),
            ("dim", ctypes.c_size_t),
            ]

class Dataset_Split(ctypes.Structure):
    _fields_ = [
            ("train", Dataset),
            ("test", Dataset)
            ]


lib.cml_fit_linear_impl.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
lib.cml_fit_linear_impl.restype = Sample

lib.cml_train_linear.argtypes = [
        Sample,
        ctypes.c_int32
        ]

lib.cml_train_linear.restype = ctypes.c_float

lib.cml_knn_fit_impl.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_float
        ]

lib.cml_knn_fit_impl.restype = Dataset_Split

lib.cml_knn_train.argtypes = [
        Dataset_Split
        ]


def cml_knn_train(dataset: Dataset_Split):
    lib.cml_knn_train(dataset)

def cml_knn_fit(data: list, labels: list, train: float):
    data_np = np.array(data, dtype=float32)
    
    n_np = np.int32(data_np.shape[0])
    dim_np = np.int32(data_np.shape[1])
    labels_np = np.array(labels, dtype=int32)
    train_np = np.float32(train)
    return lib.cml_knn_fit_impl(
        n_np,
        dim_np,
        data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        labels_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        train_np
    )

def cml_train_linear(data: Sample, epochs: int):
    epochs_np = np.int32(epochs)
    return lib.cml_train_linear(data, epochs_np)
    


def cml_fit_linear(x: list, y: list):
    x_np = np.array(x, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)
    n = x_np.shape[0]

    return lib.cml_fit_linear_impl(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )
