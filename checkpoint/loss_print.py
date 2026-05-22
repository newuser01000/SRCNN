import numpy as np
import matplotlib.pyplot as plt

loss = np.load("./SRCNN915/x2/AdamW/B8/val_losses.npy")
plt.plot(loss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Band8 Validation Loss Curve")
plt.show()