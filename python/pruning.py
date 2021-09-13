import monkdata as m
import dtree as d
import random
import numpy as np
import matplotlib.pyplot as plt

possible_frac = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
global num_of_fig
num_of_fig = 1 

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction) 
    return ldata[:breakPoint], ldata[breakPoint:]


def analyse_pruning(data, test, name):
    global num_of_fig
    results = dict()
    for frac in possible_frac:
        results[frac] = dict()
        performances = []
        for i in range(100):
            train, val = partition(data, frac)
            t=d.buildTree(train, m.attributes)
            while True:
                best_possible_t_so_far = None
                curr_best_performace = d.check(t, val)
                for possible_t in d.allPruned(t):
                    performance = d.check(possible_t, val)
                    if performance > curr_best_performace:
                        best_possible_t_so_far = possible_t
                        curr_best_performace = performance
                if not best_possible_t_so_far: break
                t = best_possible_t_so_far
            performances.append(d.check(t, test))
        results[frac]['mean'] = np.mean(performances)
        results[frac]['stddev'] = np.std(performances)


    y_pos = np.arange(len(possible_frac))
    means = [results[frac]['mean'] for frac in possible_frac]
    stddevs = [results[frac]['stddev'] for frac in possible_frac]


    plt.figure(num_of_fig)
    num_of_fig += 1
    plt.bar(y_pos, means, align='center', zorder=4)
    plt.grid(zorder=0, axis='y')
    plt.xticks(y_pos, possible_frac)
    plt.ylabel('Mean Accuracy (%)')
    plt.xlabel('Fraction of data used as training set')
    plt.title(f'Performance of pruned trees on {name}')


    plt.figure(num_of_fig)
    num_of_fig += 1
    plt.bar(y_pos, stddevs, align='center', zorder=4)
    plt.grid(zorder=0, axis='y')
    plt.xticks(y_pos, possible_frac)
    plt.ylabel('Std of Accuracy (%)')
    plt.xlabel('Fraction of data used as training set')
    plt.title(f'Performance of pruned trees on {name}')
    

analyse_pruning(m.monk1, m.monk1test, 'Monk1 dataset')
analyse_pruning(m.monk3, m.monk3test, 'Monk3 dataset')
plt.show()
    



