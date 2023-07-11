class Solution(object):
    def twoSum(nums, target):
        prevMap = {} # val : index
        # print(prevMap)
        for i, n in enumerate(nums):
            diff = target - n
            if diff in prevMap:
                print(prevMap[diff], i)
                return [prevMap[diff], i]
            prevMap[n] = i

        # return[]

class Solution(object):
    def longestCommonPrefix(strs):
        strs.sort()
        print(strs)
        i=0
        k=""
        for i in range(len(strs[len(strs)-1])):

            if i == len(strs[0]) or i == len(strs[len(strs)-1]):

                return k
            elif strs[0][i]==strs[len(strs)-1][i]:

                k += strs[0][i]

            else: 

                return k 
            # return k 

# car = ["flower","flow","flight"]

# print(Solution.longestCommonPrefix(car))

class Solution():
    def isValid(s):

        stack0=[]

  
        for i in  range(0,len(s)):
            if s[i]=='('or s[i]=='['or s[i]=='{' :
                stack0.append(s[i])
                # print(stack0)
            elif not stack0 :
                return False
            elif s[i]==']' and stack0[-1]== '[':
                stack0.pop()
                # print(stack0)
            elif s[i]==')' and stack0[-1]== '(':
                stack0.pop()
                # print(stack0)
            elif s[i]=='}' and stack0[-1]== '{':
                stack0.pop()
                # print(stack0)
            else:
                stack0.append(s[i])

        if stack0==[]:
            print("T")
            return True
        else:
            print("F")
            return False

# Solution.isValid("(])")


class Solution(object):
    def plusOne(digits):
        # a = ""
        # l = []
        # for i in range(0,len(digits)):
        #     a += str(digits[i])
            
        # ans = str(int(a)+1)
        # for i in ans:
        #     l.append(int(i))
        # print(l)
        # return l
        digits1=[str(i) for i in digits]
        s=''.join(digits1)
        s = int(s)+1
        l=list(str(s))
        l = [int(i) for i in l]
        print(s)
        return l

# Solution.plusOne([5,3,2,9])

class Solution(object):
    def mergeTwoLists( list1, list2):
        # ans 1
        for i in range(0,len(list2)):
            list1.append(list2[i])

        a = sorted(list1)
        print(a)
        return a
        
        # ans2
        list1.extend(list2)
        a = sorted(list1)
        print(list1)
        print(a)
        return a


# Solution.mergeTwoLists(list1 = [1,2,4], list2 = [1,3,4])

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        cur = dummy = ListNode()
        while list1 and list2:               
            if list1.val < list2.val:
                cur.next = list1
                list1, cur = list1.next, list1
            else:
                cur.next = list2
                list2, cur = list2.next, list2
                
        if list1 or list2:
            cur.next = list1 if list1 else list2
        print(dummy)    
        return dummy.next

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = new_node
    
    def print_list(self):
        current = self.head
        while current is not None:
            print(current.data)
            current = current.next

# my_list = LinkedList()
# my_list.append(1)

class Solution:
    def generate(numRows: int):
        dp=[]
        for i in range(1,numRows+1):
            dp.append([0]*i)
        for i in range(0,numRows):
            for j in range(0,i+1):
                
                if(j==0 or j==i):
                    #For leading and trailing of the row the 1 should be appended....
                    dp[i][j]=1
                    print(dp)
                else:
                    #The previous values both are added together
                    dp[i][j]=dp[i-1][j-1]+dp[i-1][j]
      
# Solution.generate(numRows = 5)

class Solution(object):
    def generate( numRows):
        # Create an array list to store the output result...
        res = [[1]]
        for i in range(numRows-1):
            temp = [0]+res[-1]+[0]
            row = []
            for j in range(len(res[-1])+1):
                row.append(temp[j]+temp[j+1])
                # print(row)
            res.append(row)
            print(res)

# Solution.generate(numRows = 5)


# class Solution:
#     def removeDuplicates(nums:list[int]):
#         if not nums:
#             return 0

#         i = 0
#         for j in range(1, len(nums)):
#             if nums[j] != nums[i]:
#                 i += 1
#                 nums[i] = nums[j]
        
#         return i + 1


#     nums = [1,1,2]# Input array
#     expectedNums = [1, 2] # The expected answer with correct length

#     k = removeDuplicates(nums) # Calls your implementation
#     # print(k)
#     k == len(expectedNums)
#     for i in range(k):
#         nums[i] == expectedNums[i]
#         print(f"nums[{i}] = {nums[i]}, expectedNums[{i}] = {expectedNums[i]}") # Debugging statement

# Solution.removeDuplicates(nums = [0,0,1,1,1,2,2,3,3,4])
    
def containsDuplicate(nums) -> bool:
        return len(set(nums))!=len(nums)
# Solution.containsDuplicate(nums=[1,1,1,3,3,4,3,2,4,2])

# Time Limit Exceeded{Too slow} 
class Solution: 
    def containsDuplicate(nums) -> bool:
        nums.sort()
        for i,j in enumerate(nums):
            for k in range(i+1,len(nums)):
                if j == nums[k]:
                    print(j,nums[k])
                    return True
            

# Solution.containsDuplicate(nums=[1,2,3,1])
class Solution:
    def containsDuplicate( nums) -> bool:
        unique_nums = set()
        for num in nums:
            if num in unique_nums:
                print(unique_nums)
                return True
            unique_nums.add(num)
        return False
# Solution.containsDuplicate(nums=[1,2,3,1])

class Solution:
    def isAnagram(s: str, t: str) -> bool:
        S = list(s)
        T = list(t)
        S.sort()
        T.sort()
        print(S== T)



# Solution.isAnagram(s = "rat", t = "car")

class Solution:
    def diagonalSum(mat) -> int:
        temp = 0
        for i in range(len(mat)):
            # print(i)
            for j in range(len(mat[i])):
                # print(j)
                if i == j or i+j==len(mat)-1:
                    temp += mat[i][j]
                    # print("i = "+str(i))
                    # print("j = "+str(j))
                    # print(mat[i][j])
                    print(temp)
        return temp
                
            
# Solution.diagonalSum(mat = [[1,2,3],[4,5,6],[7,8,9]])

# Definition for singly-linked list.

class Solution:
    def runningSum(nums) :
        # ans 1 
        # ans = []
        # temps = 0
        # for i in range(len(nums)):
        #     # print("i="+str(i))
        #     for j in range(i) :
        #         temps +=nums[j]
        #         # print("temps="+str(temps))
        #     ans.append(nums[i]+temps)
        #     temps = 0
        # print(ans)
        # return ans

        # ans 2
        l = len(nums)
        for i in range(1, l):
            nums[i] += nums[i-1]
        print(nums)
        return nums
# Solution.runningSum(nums = [1,2,3,4])
class Solution:
    def pivotIndex(nums) -> int:
        r_sum,l_sum = 0,sum(nums)
        for idx , v in enumerate(nums):
            l_sum -= v

            print(idx)
            if r_sum == l_sum:
                
                return idx
            r_sum += v
        return -1
# Solution.pivotIndex(nums = [1,7,3,6,5,6])

class Solution:
    def isIsomorphic(self, s, t):
        map1 = []
        map2 = []
        for idx in s:
            map1.append(s.index(idx))
        for idx in t:
            map2.append(t.index(idx))
        if map1 == map2:
            return True
        return False

# Solution.isIsomorphic(s = "egg", t = "add")
# # Time Complexity : O(n)


class Solution:
    def isSubsequence(s: str, t: str) -> bool:
        T = list(t)
        S = list(s)
        i  = 0
        if S!=[] : 
            for idx in range(len(T)):
                # print(idx)
                if i<len(S) and S[i] == T[idx]:
                    i+=1
            print(i == len(s))
            return i == len(s)

        return True


# Solution.isSubsequence(s = "axc", t ="ahbgdc")

class Solution:
    def maxProfit(prices):
    #     diff = []
    #     for i in range(0,len(prices)):
    #         # print(i)
    #         for j in range(i) :
    #             print("j ="+str(j)+"รอบที่ i =" + str(i))
    #             diff.append(prices[i]-prices[j])
    #             # print("pricesi-j == "+str(prices[i]-prices[j]))
    #     if diff != [] and max(diff) > 0:
    #         return max(diff)
    #     return 0

        if not prices:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        for price in prices:
            max_profit = max(max_profit, price - min_price)
            print(max_profit)
            min_price = min(min_price, price)
        # print(max_profit)
        return max_profit
# Solution.maxProfit(prices = [7,1,5,3,6,4])

class Solution(object):
    def twoSum(nums, target):
        pm = {}
        for i , j in enumerate(nums):
            diff = target-j
            if diff in pm:
                print(pm[diff],i)
            pm[j]=i
# Solution.twoSum(nums = [2,7,11,15], target = 9)


def solution(A, B):
    start=int(A[:2])*60+int(A[3:])
    end = int(B[:2])*60+int(B[3:])

    # print(start,end)
    print(start%60,end%60)
    if end< start:
        end+=1440
    if end< start:
        end+=1440
    T = end-start
    if start % 60 == 0 or start % 60 == 15 or start % 60 == 30 or start % 60 == 45 or end % 60 == 0 or end % 60 == 15 or end % 60 == 30 or end % 60 == 45:
        print((T//15)-1)

    print(T//15)

# solution(A="12:03",B="12:03")

def solution(S):
    r = [0]*len(S)
    l = [0]*len(S)

    r_c =0
    l_c =0
    
    for i in range(len(S)):
        if S[i] == '.':
            r[i]= r_c
        elif S[i] == '>':
            r_c += 1
    for i in range(len(S)-1,-1,-1):
        if S[i] == '.':
            l[i]= l_c
        elif S[i] == '<':
            l_c += 1
    ans = 0
    for i in range(0,len(S)):
        ans = ans+ l[i]+r[i]
    print(ans)

# solution(S=">>>.<<<")


def solution(N):
    st_n = str(N)
    idx = st_n.index('5')

    maxv = int(st_n[:idx]+st_n[idx+1:])

    if '5'in st_n[idx+1:]:
        print("555")
        tem = [int(st_n[:i]+st_n[i+1:]) for i in range(idx+1,len(st_n)) if st_n[i] == '5']

    R_max=max(maxv,*tem)
    print(R_max)
    

# solution(-5859)
def solution(N):
    st_n = str(N)
    idx = st_n.index('5')

    maxv = int(st_n[:idx]+st_n[idx+1:])

    for i in range(idx+1,len(st_n)):
        if st_n[i] == '5':
            tem = int(st_n[:i]+st_n[i+1:])
            print(tem)
    R_max  =max(maxv,tem)
    print( R_max)
    return R_max
   
# solution(157060)



def solution(S):
    word = "BANANA"
    word_counts = {}
    for char in word:
        if char not in word_counts:
            word_counts[char] = 0
        word_counts[char] += 1
    deletions = float("inf")
    for char in word:
        char_count = S.count(char)
        if char_count < word_counts[char]:
            return 0
        deletions = min(deletions, char_count // word_counts[char])
    print(deletions)
    return deletions

# solution(S = "NAANAAXNABABYNNBZ")


class Solution:
    def maxProfit(prices) -> int:
        diff = []

        for i in range(0,len(prices)):
            diff.append(max(int(prices[x]) for x in range(i,len(prices)))-prices[i])
        print(diff)
        print(max(diff))


# Solution.maxProfit(prices = [7,1,5,3,6,4])



# scb
def countDuplicate(numbers) :
    # print(numbers)
    # nums = set(numbers)
    # print(nums)
    # a = tem = 0 
    # for i in nums:
    #     tem = numbers.count(i)
    #     if tem >1:
    #         a+=1
    # print(a)
    print(sum(numbers.count(i) > 1 for i in set(numbers)))
    # return a

# countDuplicate(numbers= [1,1,1,1,1,2,2,5,4])


class Solution:
    def removeDuplicates( nums) -> int:
        nums[:] = sorted(set(nums))
        print(nums)
        return len(nums)

        
# Solution.removeDuplicates(nums = [1,1,2])


class Solution:
    def removeElement( nums, val: int) -> int:
        while val in nums:
            nums.remove(val)
  
# Solution.removeElement(nums = [3,2,2,3], val = 3) 


class Solution:
    def addBinary( a, b) -> str:
        return bin(int (a,2) + int(b,2))[2:]
# Solution.addBinary(a = "11", b = "1")


class Solution:
    def mySqrt(x: int) -> int:
        # return int(x**0.5)
        a = 0
        while a * a <= x:
            a += 1
        print(a - 1)
        return a - 1
# Solution.mySqrt(x = 4)

# scb
def fibonacci (n):

    # 1
    # if n <= 0:
    #     return []
    # elif n == 1:
    #     return [0]
    # elif n == 2:
    #     return [0, 1]
    # ans = [0,1]
    # for i in range(n-2):
    #     num = ans[-1] + ans[-2]
    #     ans.append(num)
    #     print(ans)
    #     return ans

    # 2
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    print(fib)
    return fib[:n]

    # 3
    # fib = [0, 1]
    # [fib.append(fib[-1] + fib[-2]) for _ in range(2, n)]
    # print(fib[:n])
fibonacci(n=4)



def carParkingRoof(cars, k):
    cars.sort()
    b = []
    b.append(9999999999999999)
    for i in range(len(cars)-k-1):
        b.append(cars[i+k-1]-cars[k-1]+1)
    print(min(b))

# carParkingRoof(cars=[2, 10, 8, 17 ,3,13] , k=4)


# LooLooTech

def count_pairs (number_list, power, distance):
    count = 0  
    l_resulte = []
    squared = [num**power for num in number_list]
    # print(squared)

    for i in range(len(squared)):
        # print(squared[i] % power)
        for j in range(i+1, len(squared)):
            # print(squared[i],squared[j])
            if squared[i] % power == 0 or squared[j] % power ==0:
                if abs(squared[i] - squared[j]) <= distance:
                    count += 1
                    l_resulte.append((number_list[i],number_list[j]))
    
    print(l_resulte)
#     print(count)
	
# count_pairs([1,2,3,4,6,10], 3, 200)


def find_max_ngram (text, length):
    T_ngram = {}
    count = 0
    ans_ngram = ""
    # print("abcx"  < "bcax")
    for i in range(len(text) - length + 1):

        ngram = text[i:i+length]
        # print(ngram)
        if ngram in T_ngram:
            T_ngram [ngram] += 1
        else:
            T_ngram[ngram]=1
        
        if T_ngram[ngram] > count:
            count = T_ngram[ngram]
            ans_ngram = ngram
        elif T_ngram[ngram] == count and ngram < ans_ngram:
            ans_ngram = ngram
    print(ans_ngram)

# find_max_ngram ("xbcxbcaxbcaxb", 4)

def check_perfect_list(numbers, distance):
    if len(numbers) < distance:
        return False

    for i in range(len(numbers) - distance +1):

        next = numbers[i+1:i+3]
        print(next)
        for num in next:
            print(num-numbers[i])
            if num - numbers[i] > distance:
                return False

    return True

# check_perfect_list([1, 4, 3, 2, 3], 3)



class Solution:
    def shortestPathBinaryMatrix(grid) -> int:
        edge = -1
        count = 0
        margin = []
        m=k=-1
        for i ,value in enumerate(grid):
            for j , border in enumerate(value):
                if border==0:
                    margin.append([i,j])
                    
        print(margin)
        for i , value in enumerate(margin):
            print(value[0][0])
            
        # edge = count
        # print(edge) 


# Solution.shortestPathBinaryMatrix(grid =[[0,0,0],[1,1,0],[1,1,0]])

class Solution:
    def fizzBuzz(n):
        for i in range(1, n+1):
            if i % 3 == 0 and i % 5 == 0:
                print("FizzBuzz")
            elif i % 3 == 0:
                print("Fizz")
            elif i % 5 == 0:
                print("Buzz")
            else:
                print(i)
# Solution.fizzBuzz(n=15)

def minTime(processorTime, taskTime):
    processorTime.sort()  
    taskTime.sort()
    taskTime.reverse()  
    # print(processorTime,taskTime)

    finish_time = 0
    max_f = [0]
    for i in range(len(processorTime)):
        if i == 0 :
            finish_time = processorTime[i] + taskTime[i]
            max_f.append(finish_time)
        else:
            finish_time = processorTime[i] + taskTime[int(len(taskTime)/2)]
            max_f.append(finish_time)
        print(max(max_f))
        
    # print(max(finish_time))
        # finish_time.append(processorTime[i] + taskTime[i])
        # print(finish_time)
        # max_time =max(finish_time)
        # print(max_time)
    

# def minTime(processorTime, taskTime):
#     n = len(processorTime)
#     m = len(taskTime)
#     taskTime.sort(reverse=True)  # Sort task times in descending order

#     # Assign tasks to processors
#     processors = [[] for _ in range(n)]
#     for i in range(m):
#         processors[i % n].append(taskTime[i])

#     # Calculate the finish time for each processor
#     finish_times = [0] * n
#     for i in range(n):
#         processors[i].sort(reverse=True)  # Sort task times for each processor
#         finish_times[i] = processorTime[i] + sum(processors[i][:4])

#     print(max(finish_times))
#     return max(finish_times)  # Return the maximum finish time




# minTime(processorTime = [10,20], taskTime=[2, 3, 1,2,5, 8, 4, 3])


def minimizeBias(ratings):
    ratings.sort() 
    total_bias = 0

    for i in range(0, len(ratings), 2):
        bias = ratings[i+1] - ratings[i]  
        total_bias += bias
    print(total_bias)
    return total_bias
# minimizeBias(ratings=[2,4,5,3,7,8])

