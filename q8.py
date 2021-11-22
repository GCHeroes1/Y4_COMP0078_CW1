from q7 import sample_with_biased_coin
import numpy as np
import matplotlib.pyplot as plt
from knn import knn
from tqdm import tqdm


def protocol_b(m_list: list, by_k=49, runs=100):
	best_k_for_m = {}
	for m in tqdm(m_list):
		errors_with_k = np.zeros((runs, by_k))
		for run in tqdm(range(runs)):
			for k in range(1, by_k+1):
				training_points = sample_with_biased_coin(size=m, k=k)
				testing_points = sample_with_biased_coin(size=1000, k=k)

				# run knn for each testing point
				testing_errors = np.zeros(len(testing_points))
				for i, testing_point in enumerate(testing_points):
					result = knn(training_points, (testing_point[0], testing_point[1]), k)
					error = 1 if result != testing_point[2] else 0
					# log testing error
					testing_errors[i] = error

				errors_with_k[run][k-1] = np.mean(testing_errors)

		# take average of errors from 100 runs and pick best k
		average_errors_by_k = np.average(errors_with_k, axis=0)
		best_k = np.argmax(average_errors_by_k) + 1

		# update best k value for each m
		best_k_for_m.update({m: best_k})

	return best_k_for_m


if __name__ == '__main__':
	BY_K = 49
	RUNS = 100

	m_list = [100] + [x for x in range(500, 4001, 500)]
	best_k_for_m = protocol_b(m_list, by_k=BY_K, runs=RUNS)
	print(best_k_for_m)

	# plot m Vs. optimal K value
	optimal_k_values = []
	for m in m_list:
		optimal_k_values.append(best_k_for_m[m])

	plot1 = plt.figure(1)

	plt.plot(m_list, optimal_k_values, label='optimal k value', color='orange')
	plt.title("Q8: Optimal K for each training size")
	plt.legend()
	plt.xlabel("m")
	plt.ylabel("Optimal K")
	plt.xticks(m_list)
	plt.savefig('./plots/q8.png')
