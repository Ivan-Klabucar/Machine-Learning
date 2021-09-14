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
    num_of_iter = 1000
    global num_of_fig
    results = dict()
    results['pruned'] = dict()
    results['unpruned'] = dict()
    for frac in possible_frac:
        results['pruned'][frac] = dict()
        results['unpruned'][frac] = dict()
        performances_pruned = []
        performances_unpruned = []
        for i in range(num_of_iter):
            train, val = partition(data, frac)
            t=d.buildTree(train, m.attributes)
            performances_unpruned.append(d.check(t, test))
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
            performances_pruned.append(d.check(t, test))
        results['pruned'][frac]['mean'] = np.mean(performances_pruned)
        results['pruned'][frac]['stddev'] = np.std(performances_pruned)
        results['unpruned'][frac]['mean'] = np.mean(performances_unpruned)
        results['unpruned'][frac]['stddev'] = np.std(performances_unpruned)


    x_pos = np.arange(len(possible_frac))
    means_p = [results['pruned'][frac]['mean'] for frac in possible_frac]
    stddevs_p = [results['pruned'][frac]['stddev'] for frac in possible_frac]
    means_up = [results['unpruned'][frac]['mean'] for frac in possible_frac]
    stddevs_up = [results['unpruned'][frac]['stddev'] for frac in possible_frac]
    labels = ['pruned', 'unpruned']


    plt.figure(num_of_fig)
    plt.subplots_adjust(right=0.80)
    num_of_fig += 1
    plt.bar(x_pos - 0.2, means_p, align='center', zorder=4, width=0.4, color='bisque')
    plt.bar(x_pos + 0.2, means_up, align='center', zorder=4, width=0.4, color='cornflowerblue')
    plt.grid(zorder=0, axis='y')
    plt.xticks(x_pos, possible_frac)
    plt.ylabel('Mean Accuracy (%)')
    plt.xlabel('Fraction of data used for training')
    plt.title(f'Performance of decision trees on {name}\nn={num_of_iter}')
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))


    plt.figure(num_of_fig)
    plt.subplots_adjust(right=0.80)
    num_of_fig += 1
    plt.bar(x_pos - 0.2, stddevs_p, align='center', zorder=4, width=0.4, color='bisque')
    plt.bar(x_pos + 0.2, stddevs_up, align='center', zorder=4, width=0.4, color='cornflowerblue')
    plt.grid(zorder=0, axis='y')
    plt.xticks(x_pos, possible_frac)
    plt.ylabel('Std of Accuracy')
    plt.xlabel('Fraction of data used for training')
    plt.title(f'Performance of decision trees on {name}\nn={num_of_iter}')
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    

analyse_pruning(m.monk1, m.monk1test, 'Monk1 dataset')
analyse_pruning(m.monk3, m.monk3test, 'Monk3 dataset')
plt.show()
    



