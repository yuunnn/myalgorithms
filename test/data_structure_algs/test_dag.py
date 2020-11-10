from algs.data_structure_algorithms.dag import dag, sort_dag


data = {
        "input_data1": {},
        "input_data2": {},
        "input_data3": {},
        "input_data4": {},
        "f1": {"input_data1"},
        "f2": {"input_data2", "input_data3"},
        "f3": {"input_data1", 'input_data4'},
        "f4": {"f1"},
        "f5": {"f2", "f3", "f4"},
        "f6": {"f4", "f5"},
        "f7": {"f2", "f4", "f5"},
        "f8": {"f6", "f7"}}

res1 = dag(data)
# data其实已经描述了DAG的边了，因此sort_dag可以直接用data来做，Dag的顺序遍历就是找到没有出度的节点，然后依次pop出来
res2 = sort_dag(data)
print("###################################################")
print("dag长相：")
print(res1)
print("###################################################")
print("dag的拓扑排序：")
print(res2)
