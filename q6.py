import numpy as np
import matplotlib.pyplot as plt


def data_generation(size: int):
	S = np.zeros((size, 3))

	x_coordinates = np.random.uniform(0, 1, size)
	y_coordinates = np.random.uniform(0, 1, size)
	# np.random.randint generate labels from the discrete uniform distribution
	y_i = np.random.randint(2, size=size)

	S[:, 0] = x_coordinates
	S[:, 1] = y_coordinates
	S[:, 2] = y_i

	return S


if __name__ == '__main__':
	S = data_generation(100)
	plt.figure(1)
	plt.scatter(S[:, 0], S[:, 1], c=S[:, 2])
	plt.title("A hypothesis visualised with |S|=100")
	plt.savefig('./plots/q6.png')
