import numpy as np
import timeit

setup = """
import numpy as np
a = np.random.rand(1000,1000)
b = np.random.rand(1000,1000)
"""

# Write statement to be timed
stmt = """np.dot(a,b)"""

print(np.mean(timeit.Timer(stmt = stmt, setup = setup).repeat(10, 1)))
