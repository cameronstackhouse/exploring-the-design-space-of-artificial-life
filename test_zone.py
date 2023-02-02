from tools.speciate import distance
from networks import CPPN

if __name__ == "__main__":
    a = CPPN([8,8,7])
    b = CPPN([8,8,7])
    print(distance(a, b))