import numpy as np

dataset = np.load('NatSGD_v0.3.npz',allow_pickle=True)['NatComm'].item()
# Get the keys present in the dataset
keys = dataset.keys()

# Print the keys
for key in keys:
    print(key)
# Pariticipants
#dict_keys([64, 70, 40, 45, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])
print(dataset.keys())

p64 = dataset[64]

# Show all keys
print(p64.keys())
# OUTPUT: dict_keys(['7EBkdm6zEmkUwJOyIwcIAMkMznMwGT2x','vWjFXNXuw8hkoA6iM6BvzzbmkkOrd7LU', ...]

# See a sample data
print(p64['7EBkdm6zEmkUwJOyIwcIAMkMznMwGT2x'])
# OUTPUT: [1129, 27, 416300, 418749, 'turn back', 1, 1, 'on', 'gas', {'12a12269d88648d992c480baf34c4d19': [416700, 417800, 'so turn back', [array([ ...], dtype=float32), array([ ... ], dtype=float32), array([ ...], dtype=float32)]]}, {'gesture_keypoints': array([[[..., [653, 245, 680, ..., 604, 671, 596]]]), 'gesture_info': [1.0, 1.0, 'RH|', 0.0, nan, 0.0, nan]}]
