# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod


class BasePPLImplementation(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize the model representation in the PPL.
        """
        pass
