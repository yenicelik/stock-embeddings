"""
    Parameter definitions
"""
from sys import platform

class Parameters:

    def __init__(self):
        self.embedding = True
        self.development = True
        self.is_leonhard = (platform == "linux" or platform == "linux2") # and False

        if self.is_leonhard:
            self.development = False

        if not self.development:
            print("Full mode!")
        else:
            print("DEV ON!!!")

    @property
    def embedding_dimension(self):
        return 5

params = Parameters()