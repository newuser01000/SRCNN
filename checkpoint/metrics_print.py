import numpy as np
import matplotlib.pyplot as plt

metrics = np.load("./SRCNN915/x2/AdamW/Band234/metrics.npy")
plt.plot(metrics)
plt.xlabel("Iteration")
plt.ylabel("Metrics")
plt.title("Training Metrics Curve")
plt.show()