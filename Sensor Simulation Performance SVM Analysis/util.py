import pickle

a = [15, 20, 25, 30]
b = [25, 30, 35, 40]

ab = [(i, j) for i in a for j in b if i < j]
print ab