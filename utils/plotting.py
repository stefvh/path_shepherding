from matplotlib import pyplot as plt


class Plotr(object):
    def __init__(self):
        # Set plotting font for TeX labels
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'text.latex.preamble': r'\usepackage{amsfonts}' r'\usepackage{amsmath}'
        })

        # Set markers & colors
        self.markers = ['s', 'o', 'D', 'p', '^', '*', '>', 'h', 'v', '<']
        self.colors = [
            'k', 'firebrick', # 'seagreen',
            'darkorange', 'indigo', 'peru', 'orchid', 'maroon', 'darkmagenta', 'darkgreen', 'navy', 
        ]
        self.lightcolors = [
            'lightgrey', 'steelblue', 'lightcoral', 'mediumseagreen', 'orchid',
            'sandybrown', 'mediumpurple'
        ]
        self.figlabels = [
            r'A', r'B', r'C', r'D',
            r'E', r'F', r'G', r'H'
        ]
        self.figbflabels = [
            r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}',
            r'\textbf{E}', r'\textbf{F}', r'\textbf{G}', r'\textbf{H}'
        ]
        self.linestyles = ['-', '--', ':', '-.']
        self.tabcolors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
            'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
            'tab:olive', 'tab:cyan'
        ]
