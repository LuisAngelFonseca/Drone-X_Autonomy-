import numpy as np
import pickle

with open('mask_values.pkl', 'rb') as f:
    lower_hsv, upper_hsv, erosion, dilation = pickle.load(f)

min_hue, min_sat, min_val = lower_hsv
print(min_hue)
print(erosion)
print(dilation)
# print(upper_hsv)
