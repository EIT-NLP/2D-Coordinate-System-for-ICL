import matplotlib.pyplot as plt
def plot_PIR(rrs, plot_file):
    plt.figure(figsize=(14, 9))
    layers = [i for i in range(len(rrs))]
   
    plt.plot(layers, rrs, linestyle='-', marker='*', markersize=25, linewidth=3, color='#ff7f0e')

    # Customize plot
    xticks = list(range(0, len(rrs), 5))
    plt.xticks(xticks, fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('Layer', fontsize=35)
    plt.ylabel('$PIR$', fontsize=35)

    # Remove grid
    plt.grid(False)
    plt.ylim(0, 1)

    # Show plot
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.show()


