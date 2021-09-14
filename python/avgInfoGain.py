import monkdata as m
import dtree as d

def infogain_for_dataset(data, attr, name):
    print(f'infogain for {name}: ')
    for a in attr:
        print(f'  {a.name}: {d.averageGain(data, a)}')
    print()

infogain_for_dataset(m.monk1, [m.attributes[i] for i in range(6)], 'monk1')
infogain_for_dataset(m.monk2, [m.attributes[i] for i in range(6)], 'monk2')
infogain_for_dataset(m.monk3, [m.attributes[i] for i in range(6)], 'monk3')