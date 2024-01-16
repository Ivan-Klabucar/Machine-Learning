import monkdata as monk
import dtree as dtree
import drawtree_qt5 as dt

monk1_tree = dtree.buildTree(monk.monk1, monk.attributes)
#dt.drawTree(monk1_tree)
print(dtree.check(monk1_tree, monk.monk1))
print(dtree.check(monk1_tree, monk.monk1test))

monk2_tree = dtree.buildTree(monk.monk2, monk.attributes)
#dt.drawTree(monk3_tree)
print(dtree.check(monk2_tree, monk.monk2))
print(dtree.check(monk2_tree, monk.monk2test))

monk3_tree = dtree.buildTree(monk.monk3, monk.attributes)
#dt.drawTree(monk3_tree)
print(dtree.check(monk3_tree, monk.monk3))
print(dtree.check(monk3_tree, monk.monk3test))