# plot_results.py

import torch
import matplotlib.pyplot as plt
import os



# data = []

# data.append({"protonet_64_5_5" : [0.9354838728904724, 0.7884615659713745, 0.7142857313156128, 0.5833333134651184, ],
#              "protonet_64_5_1" : [0.8790322542190552, 0.6875, 0.6071428656578064, 0.2708333432674408, ],
#              "protonet_25_5_1" : [0.8629032373428345, 0.5769230723381042, 0.4910714328289032,  0.25, ],
#              "clip_only" : [0.9213709831237793, 0.8653846383094788, 0.7767857313156128, 0.6666666865348816, ]})

# data.append({"protonet_64_5_5" : [0.95703125, 0.9553571343421936, 0.8333333134651184, 0.71875, ],   
#                 "protonet_64_5_1" : [0.90234375, 0.7946428656578064, 0.7083333134651184, 0.34375, ],
#                 "protonet_25_5_1" : [0.73046875, 0.5267857313156128, 0.1875, 0.0625, ],
#                 "clip_only" : [0.9765625, 0.8928571343421936,  0.8125, 0.65625, ]})

# data.append({"protonet_64_5_5" : [0.9821428656578064, 0.9583333134651184, 0.9375, 0.6875,],
#                 "protonet_64_5_1" : [0.9375, 0.7916666865348816, 0.71875, 0.4375, ],
#                 "protonet_25_5_1" : [0.6875, 0.1666666716337204, 0.40625, 0.0, ],
#                 "clip_only" : [0.9910714030265808, 0.9375,  0.9375, 0.9375]})

# shot = [1,2,5]

# def plot_results(data):
#     if not os.path.exists("figures"):
#         os.makedirs("figures")

#     for i, datum in enumerate(data):
#         plt.style.use("ggplot")
#         plt.figure()
#         for model in datum.keys():
#             plt.plot([2, 5, 10, 25], datum[model], label=model)  # Modify x-values here
#         plt.xlabel("Way")
#         plt.ylabel("Test Accuracy")
#         plt.legend()
#         plt.savefig(f'figures/{shot[i]}_shot.png')


dir = 'results/results.csv'

if not os.path.exists("figures"):
    os.makedirs("figures")


legend_dict = {
    'ProtoNetICL_64_0.0001_5_1_8_256_5_5_decathalon' : '1-layer, 5-way, 5-shot',
    'ProtoNetICL_64_0.0001_5_2_8_256_5_5_decathalon' : '2-layer, 5-way, 5-shot',
    'ProtoNetICL_64_0.0001_5_2_8_256_5_1_decathalon' : '2-layer, 5-way, 1-shot',
    'ProtoNetICL_64_0.0001_5_4_8_256_5_5_decathalon' : '4-layer, 5-way, 5-shot',
    'ProtoNetICL_64_0.0001_5_4_8_256_5_1_decathalon' : '4-layer, 5-way, 1-shot',
    'ProtoNetWithoutEncoder' : 'CLIP Only',
}
               


import pandas as pd
import matplotlib.pyplot as plt
def plot_results():
    df = pd.read_csv(dir)
    grouped_data = df.groupby('shot')
    leg = True
    for name, group in grouped_data:
        plt.style.use("ggplot")
        plt.figure()
        for model in group['model'].unique():
            plt.plot(group[group['model'] == model]['way'], group[group['model'] == model]['test_accuracy'], label=legend_dict[model] if leg else None)
        leg = False
        plt.xlabel("Way")
        plt.xticks([2, 5, 10])
        plt.ylabel("Test Accuracy")
        plt.legend(loc='lower left')
        plt.savefig(f'figures/{name}_shot.png', dpi=300)
    

if __name__ == "__main__":
    plot_results()