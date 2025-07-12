import random
import time


# This function is directly taken from the official reproduction package of DeepGD:
# https://github.com/ZOE-CA/DeepGD

# size: test budget
# index: the indices of all the test inputs
def rs(size, index):
    s_time = time.time()
    hh = random.sample(index, size)
    e_time = time.time()
    execution_time = e_time - s_time
    return hh, execution_time
