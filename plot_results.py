# plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt



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
               



def plot_results():
    df = pd.read_csv(dir)
    grouped_data = df.groupby('shot')
    leg = True
    for name, group in grouped_data:
        plt.style.use("ggplot")
        plt.figure()
        for model in group['model'].unique():
            plt.plot(group[group['model'] == model]['way'], group[group['model'] == model]['test_accuracy'], label=legend_dict[model] if leg else None,
                     linewidth=4.0)
        leg = False
        plt.ylim(0,1)
        plt.xlabel("Way")
        plt.xticks([2, 5, 10])
        plt.ylabel("Test Accuracy")
        plt.legend(loc='lower left')
        plt.savefig(f'figures/{name}_shot.png', dpi=300, transparent=True)
    

if __name__ == "__main__":
    plot_results()