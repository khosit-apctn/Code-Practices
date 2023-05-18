import math
import os
import random
import re
import sys

# def compareTriplets(a, b):
    # Alice = 0 
    # Bob = 0
    # print(a,b)
    # for i in range(len(a)):
    #     if a[i] > b[i]:
    #         Alice +=1
    #     elif a[i] < b[i]:
    #         Bob +=1
    # ans = [Alice,Bob]
    # # print(ans)
    # return ans


# if __name__ == '__main__':


    # a = list(map(int, input().rstrip().split()))

    # b = list(map(int, input().rstrip().split()))

    # result = compareTriplets(a, b)

# def aVeryBigSum(ar):
    # ans = 0
    # for i in ar:
    #     ans += i
    # print(ans)

# if __name__ == '__main__':
    # ar = list(map(int, input().rstrip().split()))
    # result = aVeryBigSum(ar)

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'diagonalDifference' function below.
#
# The function is expected to return an INTEGER.
# The function accepts 2D_INTEGER_ARRAY arr as parameter.
#

def diagonalDifference(arr):
    print(arr)
    for i in range(3):
        print(i)


if __name__ == '__main__':

    n = int(input().strip())

    arr = []

    for _ in range(n):
        arr.append(list(
                    map(int, input().rstrip().split())
                    ))

    result = diagonalDifference(arr)