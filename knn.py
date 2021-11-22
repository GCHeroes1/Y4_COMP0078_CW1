import math
import numpy as np
import matplotlib.pyplot as plt
from q6 import data_generation


def knn(training_points: np.ndarray, test_point: tuple, k: int) -> int:
	distances = []
	for training_point in training_points:
		x_coor, y_coor = training_point[0], training_point[1]
		label = training_point[2]

		# get euclidean distance
		distance = math.sqrt((x_coor - test_point[0]) ** 2 + (y_coor - test_point[1]) ** 2)
		distances.append((distance, label))

	# only get the top k neighbours
	distances_sorted = sorted(distances)[:k]

	# counting the frequency of labels
	freq_0 = 0
	freq_1 = 0
	for d in distances_sorted:
		if d[1] == 0:
			freq_0 += 1
		elif d[1] == 1:
			freq_1 += 1
		else:
			raise Exception("label should be either 0 or 1")

	return 0 if freq_0 > freq_1 else 1


if __name__ == '__main__':
	S = data_generation(10)
	print(S)
	plt.figure(1)
	plt.scatter(S[:, 0], S[:, 1], c=S[:, 2])

	result = knn(S, (0.1, 0.2), 3)
	print(result)
	plt.show()
