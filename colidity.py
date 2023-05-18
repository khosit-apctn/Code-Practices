# def solution(N):
#     bi_ = bin(N)[2:]
#     binary = [int(i) for i in str(bi_)]
#     temp= []
#     ans = []
#     print(binary)
#     for i in range(0,len(binary)) :
#         if binary[i] == 1:
#             temp.append(i)
#             print(temp)
#     if len(temp) >= 2:
#         for j in range(0,len(temp)):
#             if j+1 != len(temp):
#                 # print(temp[j+1]-temp[j]-1)
#                 ans.append(temp[j+1]-temp[j]-1)
#         print(max(ans))
#         return max(ans)
#     else : return 0
        
# solution(N = 32)


# def solution(A, K):
#     ans = []
#     n = K % len(A)
#     print(n)
#     for i in range(n):
#         ans.append(A[len(A)-n+i])
#         A.remove(A[len(A)-n+i])
#     # print(ans)
#     # print(A)
#     ans.extend(A)
#     print(ans)
#     return ans
# solution( A = [3, 8, 9, 7, -6] , K = 15)

# def solution(A):
#     # initialize the result to zero
#     # result = 0
#     # for element in A:
#     #     result ^= element
#     # print(result)
        
#     # # return the unpaired element
#     # return result
#     ans = []
#     # print(ans)
#     for val in A:

#         if val in ans:
#             ans.remove(val)

#         elif val not in ans:
#             ans.append(val)
#     print(ans[-1])
#     # return A.index(ans[-1])

# def solution(X, Y, D):
#     n=0
#     n = (Y-X)/D
#     print(int(n+(n%1>0)))
#     return int(n+(n%1>0))

# solution(  X = 10,Y = 85,D = 30)  # Output: 7

# def solution(A):
#     A.sort()
#     print(A)
#     i = 1
#     for idx in range(len(A)):
#         if i in A:
#             i+=1
#         elif i not in A :
#             print(i)
#             return(i)
#     else:
#         print(i)
#         return i

# solution(A=[1, 2, 3])

# def smallest_missing_positive_integer(A):
#     n = len(A)
#     seen = [False] * n  # mark all seen values
#     print(seen)
#     for i in range(n):
#         if 1 <= A[i] <= n:
#             seen[A[i]-1] = True  # mark as seen
#     for i in range(n):
#         if not seen[i]:
#             return i+1  # smallest missing positive integer
#     return n+1  # all values from 1 to n are present

# # Example usage:
# A = A=[1, 2, 3]
# print(smallest_missing_positive_integer(A))  