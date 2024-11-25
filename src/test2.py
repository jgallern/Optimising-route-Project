from instances import *
import os

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

test1 = instances(os.path.join(projectRoot, "solomon_instances", "c101.txt"))
test2 = instances(os.path.join(projectRoot, "solomon_instances", "r102.txt"))
test3 = instances(os.path.join(projectRoot, "solomon_instances", "rc103.txt"))
test1.plot()
test2.plot()
test3.plot()
plt.show()