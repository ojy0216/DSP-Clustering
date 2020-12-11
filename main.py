import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

MAG = 10
N_TOTAL = 600
N_CLUSTER = 6
MAX_CLUSTER = 10


def normalize(x):
    max_x = np.max(x)
    min_x = np.min(x)
    return 2 / (max_x - min_x) * (x - min_x) - 1


sample = [N_TOTAL // N_CLUSTER] * N_CLUSTER

X, _ = make_blobs(sample)

sse = np.zeros(MAX_CLUSTER)

for i in range(1, MAX_CLUSTER + 1):
    km = KMeans(n_clusters=i)
    pred = km.fit_predict(X)

    prediction = [0] * i

    for pred in pred:
        prediction[pred] += 1

    print(prediction, km.inertia_)

    sse[i - 1] = km.inertia_

sse /= N_TOTAL
sse /= sse[0]

grad = -np.gradient(np.log(sse))
log_grad = np.log10(grad)

optimal_k = 1
# for k, data in enumerate(log_grad):
#     if data < np.average(log_grad):
#         optimal_k = k
#         break
for k, data in enumerate(normalize(log_grad)):
    if data < 0:
        optimal_k = k
        break

print("Number of cluster: ", N_CLUSTER)
print('Optimal number of cluster: ', optimal_k)

optim_km = KMeans(n_clusters=optimal_k)
optim_pred = optim_km.fit_predict(X)

fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(4, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
ax4 = fig.add_subplot(gs[2, :])
ax5 = fig.add_subplot(gs[3, :])

ax1.scatter(X[:, 0], X[:, 1], marker='.')
ax1.set_title('Raw data (K = {})'.format(N_CLUSTER))

ax2.scatter(X[:, 0], X[:, 1], c=optim_pred, marker='.')
ax2.set_title('Clustered data')

ax3.plot(range(1, MAX_CLUSTER + 1), sse, marker='x')
ax3.set_title('Normalized Sum of Squared distances of Samples (SSE)')
ax3.grid(alpha=0.5)
ax3.set_xticks(range(1, MAX_CLUSTER + 1))

ax4.plot(range(1, MAX_CLUSTER + 1), np.log(sse), marker='x')
ax4.set_title('Log of Normalized Sum of Squared distances of Samples (Log(SSE))')
ax4.grid(alpha=0.5)
ax4.set_xticks(range(1, MAX_CLUSTER + 1))

ax5.plot(range(1, MAX_CLUSTER + 1), normalize(log_grad), marker='x')
ax5.set_title('Log of Negative Gradient of Log(SSE)\nNormalized to [-1, 1]')
ax5.set_xlabel("Optimal K: {}".format(optimal_k))
ax5.grid(alpha=0.5)
ax5.set_xticks(range(1, MAX_CLUSTER + 1))

plt.tight_layout()
plt.show()
