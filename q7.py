import numpy as np
import matplotlib.pyplot as plt
from knn import knn
from tqdm import tqdm


def sample_with_biased_coin(size: int, k: int):
	S = np.zeros((size, 3))

	x_coordinates = np.random.uniform(0, 1, size)
	y_coordinates = np.random.uniform(0, 1, size)

	S[:, 0] = x_coordinates
	S[:, 1] = y_coordinates

	# iterate through each datum and set y_i (label) by tossing biased coin
	for i in range(size):
		current_train_data = S[:i+1, :]
		current_x_coor = S[i][0]
		current_y_coor = S[i][1]

		# we get head side in 80% chance
		is_head = np.random.uniform(0, 1) < 0.8
		if is_head:
			# use knn to get y = h(x)
			y_i = knn(current_train_data, (current_x_coor, current_y_coor), k)
		else:
			y_i = np.random.randint(2)

		# set y_i
		S[i][2] = y_i

	return S


def protocol_a(by_k=49, runs=100):
	errors_with_k = np.zeros((runs, by_k))
	for k in tqdm(range(1, by_k+1)):
		for run in tqdm(range(runs)):
			training_points = sample_with_biased_coin(size=4000, k=k)
			testing_points = sample_with_biased_coin(size=1000, k=k)

			# run knn for each testing point
			testing_errors = np.zeros(len(testing_points))
			for i, testing_point in enumerate(testing_points):
				result = knn(training_points, (testing_point[0], testing_point[1]), k)
				error = 1 if result != testing_point[2] else 0
				# log testing error
				testing_errors[i] = error

			# log testing error with mean of each errors from testing points
			errors_with_k[run][k-1] = np.mean(testing_errors)

	return errors_with_k


if __name__ == '__main__':
	BY_K = 49
	RUNS = 100

	errors_with_k = protocol_a(by_k=BY_K, runs=RUNS)
	average_errors_by_k = np.average(errors_with_k, axis=0)

	# plot both average train and test errors in log scale
	plot1 = plt.figure(1)

	ks = [k for k in range(1, BY_K+1)]
	plt.plot(ks, average_errors_by_k, label='generalisation error', color='blue')
	plt.title("Q7: Average test errors")
	plt.legend()
	plt.xlabel("K")
	plt.ylabel("Estimated Generalised Error")
	plt.xticks(ks)
	plt.savefig('./plots/q7.png')
