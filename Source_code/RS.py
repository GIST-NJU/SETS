import random


# This function is reused from the official replication package of DeepGD:
# https://github.com/ZOE-CA/DeepGD

# size: test budget
# index: the indices of all the test inputs
def rs(size, index):
    hh = random.sample(index, size)
    return hh
