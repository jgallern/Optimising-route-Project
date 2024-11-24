from test import *

test1 = instances("./solomon_instances/c101.txt")
test2 = instances("./solomon_instances/r102.txt")
test3 = instances("./solomon_instances/rc103.txt")
test1.plot()
test2.plot()
test3.plot()
plt.show()