# plot_results.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(data, labels, ways):
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.style.use("ggplot")
    plt.figure()
    for i in range(4):
        plt.plot(data[i,:], label=labels[i])
    plt.xticks(range(len(ways)), ways)
    plt.xlabel("Way")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig("figures/results.png")
    plt.show()

if __name__ == "__main__":
    ways = [2,5,10,25]
    data = torch.Tensor([[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5],[0.3,0.4,0.5,0.6],[0.4,0.5,0.6,0.7]])
    labels = ["2-way", "5-way", "10-way", "25-way"]

    plot_results(data, labels, ways)