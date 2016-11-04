import util
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracy(matrix, size_list):
	size_list = sorted(size_list, reverse=True)
	result = np.zeros(min(size_list))
	for size in size_list:
		factor = len(matrix) / size
		scaled = np.zeros((size, size))
		for i in range(0, len(matrix)):
			for j in range(0, len(matrix)):
				scaled[i / factor][j / factor] += matrix[i][j]

		factor_rem = size / min(size_list)

		for i in range(0, len(result)):
			if result[i] != 0:
				continue
			s = 0
			for j in range(0, factor_rem):
				index = factor_rem * i + j
				s += float(scaled[index][index]) / float(sum(scaled[index]))
			if s / factor_rem > 0.9:
				result[i] = size
	return result

if __name__ == '__main__':
	matpath = "../simulation_confusion_mats"
	size_list = [420, 210, 105, 70, 42, 35, 30, 21, 15, 14, 10]
	MIN_E = 0.089553894
	MAX_E = 107.8818103

	x = np.linspace(MIN_E, MAX_E, min(size_list))
	y = {}
	for f in os.listdir(matpath):
		if f.startswith("confusion_matrices_"):
			matrices = util.load_pickle(os.path.join(matpath, f))
			time = f[:-4].split('_', 2)[2]
			for k in matrices.keys():
				if not k in y:
					y[k] = {}
				matrix = matrices[k]

				result = plot_accuracy(matrix, size_list)
				for i in range(0, len(result)):
					result[i] = (MAX_E - MIN_E)/result[i]
				y[k][time] = result
	for config in y.keys():
		fig, ax = plt.subplots()
		for time in y[config].keys():
			ax.plot(x, y[config][time], color=np.random.rand(3,1), label = str(time) + 's')

		plt.title("t_amb = %s, t_sens_0 = %s, noise = %s" % (config[0], config[1], config[2]))
		legend = ax.legend(loc='upper right', shadow=True)
		plt.xlabel('e')
		plt.ylabel('delta_e')
		plt.show()