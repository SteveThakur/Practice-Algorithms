# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import sys
import numpy as np
from collections import defaultdict 
import collections
import bisect

sys.setrecursionlimit(10000)



def sqt(N):
    return math.floor(math.sqrt(N))
        
def denominations(A, N):
    
    D = [[0 for _ in range(N+1)] for _ in range(len(A))]
    
    for i in range(len(A)):
        D[i][0] = 1
    for j in range(N+1):
        if j % A[0] == 0:
            D[0][j] = 1
        else:
            D[0][j] = 0
    
    for i in range(1, len(A)):
        for j in range(1, N+1):
            if j>= A[i]:
                D[i][j] = D[i-1][j] + D[i][j-A[i]]
            else:
                D[i][j] = D[i-1][j]

    return D[len(A)-1][N]


A = [i+1 for i in range(12)]
N = len(A)

print(denominations(A, N))


def Kadane(A):
    max_so_far = 0
    max_ending = 0
    
    for i in range(len(A)):
        max_ending = max(max_ending+A[i], A[i], 0)
        max_so_far = max(max_so_far, max_ending)
        
    return max_so_far

print(Kadane([2, 3, -1, 4, 5,-6, -7]))



def base(N, n):
    
    digits = 0
    while n**(digits) <= N:
        digits +=1
    
    r = N
    
    A = []
    
    for i in range(digits):
        A.append(math.floor(r/(n**(digits-i-1))))
        r = r - (math.floor(r/(n**(digits-i-1))))* (n**(digits-i-1))
    
    return A


def longestPalindrome(A):
    L = [[0 for _ in range(len(A))] for _ in range(len(A))]
    
    for i in range(len(A)):
        L[i][i] = 1
    
        
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if j == i+1 and A[i] == A[j]:
                L[i][j] = 2
            elif A[i] == A[j]:
                L[i][j] = L[i+1][j-1]+2
            else:
                L[i][j] = max(L[i+1][j], L[i][j-1])
                
    return L

def findWater(A):
    N = len(A)
    right = [0]*N    
    left = [0]*N
    
    left[0] = A[0]
    for i in range(1, N):
        left[i] = max(left[i-1], A[i])
        
    right[N-1] = max(A)
    for i in range(N-2, -1, -1):
        right[i] = max(right[i+1], A[i])
     
    water = 0    
    for i in range(N):
        water += min(left[i], right[i])-A[i]
        
    return water
 

def column(A, i):
    return [A[j][i] for j in range(len(A))]

def Represent(T):
    for r in T:
        for c in r:
            print(c, end = "    ")
        print("")



 
def Transpose(A):
    
    B = [[0 for _ in range(len(A))] for _ in range(len(A[0]))]
    for i in range(len(A[0])):
        for j in range(len(A)):
            B[i][j] = A[j][i]
    return B


A = [[1,2,3], [4, 5, 6], [7, 8, 9]]
B =[[7, 4, 1], [8, 5, 2], [9, 6, 3]]

print(Represent(A))

print(Represent(B))

                    
def Rotate(A):
    N = len(A)
    
    B = [[0 for _ in range(N)] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):   
            B[i][j] = A[j][N-1-i]
    for i in range(N):
        for j in range(N):           
            B[i][j] = B[N-1-i][N-1-j]
    return B

print(Represent(Rotate(A)))

def Rotator(A, k):
    B = A
    for i in range(1, k+1):
        B= Rotate(B)
    return B






def Product(A, B):
    if len(A[0]) != len(B):
        return("Error")
    else:
        C = [[0 for _  in range(len(B[0]))] for _ in range(len(A))]

        for i in range(len(A)):
            for j in range(len(B[0])):
                C[i][j] = sum([A[i][k]* B[k][j] for k in range(len(B))])
                
        return C
    

Bases =[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def ProbablyPrime(N):
    for x in Bases:
        if (N%x != 0) and (x**(N-1)) % N != 1:
            return False
    return True


def Maxes(A):
    M = []
    index = 0
     
    while index <= len(A)-2:
         m = max(A[index:-1])
         index = A.index(m)
         M.append([m])
         index = index + 1
    
    
    return M

def Stirling(N, k):
    S = [[0 for _ in range(k+1)] for _ in range(N)]
    
    for i in range(N):
        S[i][1] = 1
    
    for j in range(1, k+1):
        S[j-1][j] = 1
     
    for i in range(N):
        for j in range(k+1):
            S[i][j] = S[i-1][j-1] + S[i][j-1] * i

     
    return S[N-1][k]

# print(Stirling(10, 6))

def Bell(N):
    return sum([Stirling(N, k) for k in range(1, N+1)])




# Minimum node in a BST

class Node:
    def _init_(self, key):
        self.data = key
        self.left = None
        self.right = None
        

def insert(node, data):
    if node is None:
        return (Node(data))
    else:
        if data <= node.data:
            node.left = insert(node.left, data)
        else:
            node.right = insert(node.right, data)

        return node
    
def minValue(node):
    current = node
    
    while(current.left is not None):
        current = current.left

    return current.data


def Partitions(N):
    M = [[0 for _ in range(N+1)] for _ in range(N+1)]
    
    for i in range(1, N+1):
        M[i][0] = 1
        M[i][1] = 1
    
    for j in range(N+1):
        if j == 1:
            M[1][j] = 1
        
    for i in range(1, N+1):
        for j in range(1, N+1):
            if j >= i:
                M[i][j] = M[i-1][j] + M[i][j-i]
            else:
                M[i][j] = M[i-1][j]
                
    return M[N][N]


    

def Sums(A, N):
    M = [[False for _ in range(N+1)] for _ in range(len(A))]
    
    for i in range(len(A)):
        M[i][0] = True
    for j in range(N+1):
        M[0][j] = (A[0] == j)
    
    for i in range(len(A)): 
         for j in range(N+1):
             M[i][j] = M[i-1][j] or ((j>=A[i])  and (M[i-1][j-A[i]])) 
    
    
    return M[len(A)-1][N]

def buckets(A, n):
    N = sum(A)//n
    for i in range(n+1):
        if Sums(A, i*N) == False:
            return False
    return True

def gcd(a,b):
    a, b = min(a,b), max(a,b)
    
    if a == 0:
        return b
    else:
        return gcd(b%a, a)

def PossiblyPrime(p):
    for a in range(2, 20):
        if gcd(a, p) ==1 and (a **(p-1)) % p != 1:
            return False
    return True


def lcm(a, b):
    return a*b//gcd(a, b)


def max_difference(A):
    min_visited = A[0]
    max_diff = A[1] - A[0]
    
    for i in range(1, len(A)):
        
        max_diff = max(max_diff, A[i] - min_visited)
        min_visited = min(min_visited, A[i])
        
    return min_visited, max_diff

A = [5, 7, 2, 10, 1, 6, 40 ,100]

def findIndex(A, a):
    low = 0
    hi = len(A)-1
    counter = 0
    
    while low <= hi:
        mid = (low + hi)//2 + 1
        if A[mid] == a:
            return mid
        elif A[mid] < a:
            return mid + findIndex(A[mid:], a)
        else:
            return findIndex(A[:mid], a)

def GCD(A):
    if len(A) == 2:
        return gcd(A[0], A[1])
    else:
        return gcd(A[0], GCD(A[1:]))



B = [12, 22, 32, 44, 74]

print(GCD(B))    

def base(N, n):
    A = []
    R = N
    m = int(math.log(N, n))
    
    while m >= 0:
        Q = R//(n**m)
        R = R - Q*(n**m)
        A.append(Q)
        m = m-1
    return A


def Expo2(a, d, n):
    Pow = a
    for i in range(d):
        Pow = (Pow ** 2) % n
    return Pow

def Expo(a, e, n):
    A = list(reversed(base(n, 2)))
    Pow = 1
    
    for i in range(len(A)):
        if A[i] == 1:
            Pow = (Pow * Expo2(a, i, n)) % n 
    return Pow




    
    

def ProbablyPrime(p):
    for a in range(2, 20):
        if a**(p-1) % p != 1:
            return False
    return True


def longestPalSub(A):
    N = len(A)
    L = [[1 for _ in range(N)] for _ in range(N)]
    
    for i in range(N):
        L[i][i] = 1
    for i in range(N-1):
        if A[i] == A[i+1]:
            L[i][i+1] = 2
    
    for i in range(N-2):
        for j in range(i+2, N):
            if A[i] == A[j]:
                L[i][j] = 2 + L[i+1][j-1]
            else:
                L[i][j] = max(L[i+1][j], L[i][j-1])
                
    return Represent(L)


A = [1,1, 1, 1, 1 ,1 ]

print(longestPalSub(A))
    
    
    





def distances_sort(A):
    B = sorted(A, key = lambda  a: sum([x**2 for x in a]))
    return B

def sort_by_diff(A, n):
    B = sorted(A, key = lambda a: abs(n-a))
    return B

def Wes(x, e, y, l):
    
    r = e%l
    q = (e-r)//l
    
    Q = x**q
    
    return (Q**l) * (x**l) == y
            


def Index(A, a):
    low = 0
    hi = len(A)-1
    
    if a > A[-1]:
        return len(A)
    if a < A[0]:
        return 0
    else:
  
        while low <= hi:
            mid = (low+hi)//2 + 1
            if a in range(A[mid], A[mid+1]):
                return mid + 1
            elif a < A[mid]:
                hi = mid-1
            else:
                low = mid + 1

def Palindrome(A):
    for i in range(len(A)//2 + 1):
        if A[i] != A[len(A)- 1 -i]:
            return False
    return True
    
class Node: 
  
    # Constructor to initialize the node object 
    def __init__(self, data): 
        self.data = data 
        self.next = None
  
class LinkedList: 
  
    # Function to initialize head 
    def __init__(self): 
        self.head = None
  
    # Function to reverse the linked list 
    def reverse(self): 
        prev = None
        current = self.head 
        while(current is not None): 
            next = current.next
            current.next = prev 
            prev = current 
            current = next
        self.head = prev 
          
    # Function to insert a new node at the beginning 
    def push(self, new_data): 
        new_node = Node(new_data) 
        new_node.next = self.head 
        self.head = new_node 
  
    # Utility function to print the linked LinkedList 
    def printList(self): 
        temp = self.head 
        while(temp): 
            print (temp.data) 
            temp = temp.next
    

    
def Maxes(A):
    M = []
    index = 0
    
    while index <= len(A)-2:
        m = max(A[index: ])
        M.append([m, A.index(m)])
        index = A.index(m) + 1
    
    return M

def Sells(A):
    M = Maxes(A)
    Sell = M[0][0] * M[0][1] + sum(M[i][0] *(M[i][1] - M[i-1][1]-1) for i in range(1,len(M)) )
    
    return Sell

def Buys(A):
    M = Maxes(A)
    Buy = sum(a for a in A if [a, A.index(a)] not in M)
    return Buy

def Profit(A):
    return Sells(A) - Buys(A)


        
A = [2, 8, 0, 4, 2, 3]

def max_diff(A):
    min_visited = A[0]
    max_diff = A[1]-A[0]
    
    for i in range(len(A)):
        max_diff = max(max_diff, A[i]-min_visited)
        min_visited = min(min_visited, A[i])
        
    return max_diff, min_visited


def Kadane(A): 
    max_yet = 0
    max_ending = 0
    
    for i in range(len(A)):
        max_ending = max(max_ending, max_ending+ A[i])
        max_yet = max(max_yet, max_ending)

    return max_yet

print(Kadane([- 1, -4, -2, -3, -6, -3]))


def PalindromicSubarray(A):
    N = len(A)
    max_yet = 1
    for i in range(len(A)-1):
        new = 1
        for j in range(1, min(i, len(A)-i)):
            if A[i-j]== A[i+j]:
                new += 2
        max_yet = max(max_yet, new)
    
    return max_yet
    

   


def getPosition(A, x):
    
    lo = 0
    hi = len(A)-1
    
    if lo <= hi:
        mid = (lo+hi)//2
        if A[mid] == x:
            return mid
        elif A[mid] < x:
            return mid + getPosition(A[mid: hi+1], x)
        elif A[mid] > x:
            return getPosition(A[lo:mid], x)
    else:
        return -1




def binarySearch(A, x):
    lo = 0
    hi = len(A)-1
    mid = 0
    
    while lo <= hi:
        if A[mid] == x:
            return mid
        elif A[mid] <= x:
            hi = mid-1
        else:
            lo = mid+1
            
    return -1

A = [1, 3, 5, 7, 10, 34, 56, 76, 900]

x = 76

print(getPosition(A, x))


def GCD(a,b):    
    a, b = abs(a), abs(b)
    a, b = min(a,b), max(a,b)
    if a == 0:
        return b
    else:
        return GCD(b%a, a)
    
    
    

print(GCD(36, -57))
    

from collections import defaultdict 
  
# This class represents a directed graph using 
# adjacency list representation 
class Graph: 
  
    # Constructor 
    def __init__(self): 
  
        # default dictionary to store graph 
        self.graph = defaultdict(list) 
  
    # function to add an edge to graph 
    def addEdge(self, u, v): 
        self.graph[u].append(v) 
  
    # A function used by DFS 
    def DFSUtil(self, v, visited): 
  
        # Mark the current node as visited  
        # and print it 
        visited[v] = True
        print(v, end = ' ') 
  
        # Recur for all the vertices  
        # adjacent to this vertex 
        for i in self.graph[v]: 
            if visited[i] == False: 
                self.DFSUtil(i, visited) 
  
    # The function to do DFS traversal. It uses 
    # recursive DFSUtil() 
    def DFS(self, v): 
  
        # Mark all the vertices as not visited 
        visited = [False] * (max(self.graph)+1) 
  
        # Call the recursive helper function  
        # to print DFS traversal 
        self.DFSUtil(v, visited)  
        
        
        
def partition(arr,low,high): 
    i = ( low-1 )         # index of smaller element 
    pivot = arr[high]     # pivot 
  
    for j in range(low , high): 
  
        # If current element is smaller than the pivot 
        if   arr[j] < pivot: 
          
            # increment index of smaller element 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 
  
# The main function that implements QuickSort 
# arr[] --> Array to be sorted, 
# low  --> Starting index, 
# high  --> Ending index 
  
# Function to do Quick sort 
def quickSort(arr,low,high): 
    if low < high: 
  
        # pi is partitioning index, arr[p] is now 
        # at right place 
        pi = partition(arr,low,high) 
  
        # Separately sort elements before 
        # partition and after partition 
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 
        
        
        
        
        
        
def Represent(T):
    for r in T:
        for c in r:
            print(c, end = "  ")
        print(" ")
        
        
A = [[1,2,3], [4,5,6], [7,8,9]]

print(Represent(A))        
        
        
def nextWord(S):
    N = len(S)
    first_discrep = N
    for i in range(2, N):
        if S[N-i] < S[N-i+1]:
            last_discrep = N-i
            break            
    
    if last_discrep <= N-2:
        S[last_discrep], S[last_discrep+1] = S[last_discrep+1], S[last_discrep]
        return S
    else:
        return ("This is the end.")
    

S = ['z','y','h','g','b','c','a']

print(nextWord(S))    





def lastOdd(A):
    N = len(A)
    last = 0
    for i in range(1, N+1):
        if A[N-i]% 2 == 1:
            return A[N-i]


B = [1,4,6, 10,11, 2, 21, 24, 37]

print(lastOdd(B))







def maxContainer(A):
    l = 0
    r = len(A) - 1
    area = 0
    
    while l < r:
        area = max(area, min(A[l],A[r]), (r-l))
        if A[l]< A[r]:
            l +=1
        else:
            r -= 1
    return area


def mostFrequent(A):
    n = len(A)
    Hash = dict()
    for i in range(n):
        if A[i] in Hash.keys():
            Hash[A[i]] += 1
        else:
            Hash[A[i]] = 1
    
    max_count = 0
    ele = -1
    for i in Hash:
        if max_count < Hash[i]:
            ele = i
            max_count = Hash[i]
    return ele


#A = [1,2,4,5,2,3,4,5,5, 7]

#print(mostFrequent(A))

def mostFreq(A, n):
    for i in range(n-1):
        ele = mostFrequent(A)
        A = [x for x in A if x!= ele]
    return mostFrequent(A)        




def line(A):
    a = A[0]
    b = A[1]
    c = A[2]
    d = A[3]
    
    slope = (d-b)/(c-a)
    con = b-(slope*a)
    return [con,slope]

#print(line([1,2, 3,4]))










def Search(M, a):
    n = len(M[0])
    i = 0
    j = len(M[0])-1
    while i < n and j>0:
        if M[i][j] == a:
            return (i,j)
        elif M[i][j] < a:
            i += 1
        else:
            j -= 1
    return -1

M = [[1,2,3,4], [4,5,6,7], [7,8, 9, 10], [10, 11, 12, 13]]

print(Represent(M))

print(Search(M, 8))


def Anagram(S1, S2):
    L1 = sorted(list(S1))
    L2 = sorted(list(S2))
    return L1 == L2
    

S1 = 'abbacdd'
S2 = 'ababdcc'

#print(Anagram(S1, S2))






def Peaks(A):
    L = []
    for i in range(1, len(A)-1):
        if A[i]> max(A[i-1], A[i+1]):
            L.append([i, A[i]])
    return L


A = [0, 3, 1, 4, 1, 0, 5, 3, 7, 9, 11, 4]

print(Peaks(A))

def Valleys(A):
    L = []
    for i in range(1, len(A)-1):
        if A[i]< min(A[i-1], A[i+1]):
            L.append([i, A[i]])
    return L 




def intlog(N, n):
    k = 0
    while n**k <= N:
        k += 1
    return k-1

def Base(N, n):
    L = []
    k = intlog(N, n)
        
    for i in range(k+1):
        q = N//(n**(k-i))
        L.append(q)
        N = N - q* n**(k-i)
    return L
    

def findPivot(A, lo, hi):
    if lo == hi:
        return lo
    if lo > hi:
        return -1
  
    
    mid = (lo+hi)//2
  
    
    if mid < hi and A[mid] > A[mid+1]:
        return mid
    elif mid > lo and A[mid] < A[mid-1]:
        return mid-1
    elif A[lo] >= A[mid]:
        return findPivot(A, lo, mid-1)
    else:
        return findPivot(A, mid+1, hi)
    
L = [6,7,8, 1,2,3,4,5]

print(findPivot(L, 0, len(L)-1))

            
            
          



def findIndex(A, x):
    lo = 0
    hi = len(A)-1
    mid = (lo + hi)//2
    
    while lo <= hi:
            mid = (lo + hi)//2
            if (mid < len(A)-1 and A[mid] <= x and A[mid+1] > x) or (mid == len(A)-1 and A[mid] <= x):
                return mid+1
            elif A[mid+1] <= x:
                lo= mid + 1
            else:
                hi = mid 

    

# L = [1,3, -2, 5, -7, 9, -11]

# print(sorted(L, key = lambda x: abs(x-3)))

def CoinDenominations(A, n):
    N = len(A)
    M = [[0 for _ in range(n+1)] for _ in range(N)]
    M[0][0] = 1
    for i in range(N):
        M[i][0] = 1
    for j in range(n+1):
        if j % M[0] == 0:
            M[0][j] = 1
        else:
            M[0][j] = 0
    
    for i in range(1, N):
        for j in range(1, n+1):
            if j-A[i] >= 0:
                M[i][j] = M[i][j-A[i]] + M[i-1][j]
            else:
                M[i][j] = M[i-1][j]
    
    return M[N-1][n]


def longestPalindrome(A):
    N = len(A)
    L = [[0 for _ in range(N)] for _ in range(N)]
    #L[i][j] = length of longest palindromic subsequence of A[i:j+1]
    
    for i in range(N):
        L[i][i] = 1
    for i in range(N-1):
        if L[i] == L[i+1]:
            L[i][i+1] = 2
        else:
            L[i][i+1] = 1
    
    for i in range(N-2):
        for j in range(i+2, N):
            if L[i] == L[j]:
                L[i][j] = max(L[i+1][j-1]+2, L[i][j-1], L[i+1][j])
            else:
                L[i][j] = max(L[i][j-1], L[i+1][j])
    
    return L[0][N-1]

def findNext(A, j):
    if A[-1] <= A[j]:
        return None
    else:
        lo = j+1
        hi = len(A)-1
        
        while lo <= hi:
            mid = (lo+hi)//2
            if A[mid] > A[j] and A[mid-1] == A[j]:
                return mid
            elif A[mid] == A[j]:
                lo = mid
            else:
                hi = mid
    
def Breaks(A):
    B = [[0, A[0]]]
    nextIndex = 0
    while B[-1][1] < A[-1]:
        nextIndex = findNext(A, nextIndex)
        B.append([nextIndex, A[nextIndex]])
    return B


class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def originDist(self):
        return math.sqrt(self.x**2 + self.y**2)


def Slope(P, Q):
    return (Q.y - P.y)/(Q.x-P.x)

def line(P, Q):
    slope = (Q.y - P.y)/(Q.x-P.x)
    con = P.y - slope*(P.x)
    return [slope, con]
    

P = Point(1,2)
Q = Point(5, 11)

print(line(P, Q))
    
class Line:
    
    def __init__(self, slope, con):
        self.slope = slope
        self.con = con

    def xAxisIntersect(self):
        if self.slope == 0:
            return None
        else:
            return [-(self.con)/(self.slope) , 0]



def Intersection(L1, L2):
    if L1.slope == L2.slope:
        return  None
    else:
        x = (L2.con - L1.con)/(L1.slope - L2.slope)
        y = x*L1.slope + L1.con
        return Point(x,y)

def maxDiff(A):
    lo = A[0]
    max_diff = A[1] - A[0]
    for i in range(1, len(A)):
        if A[i] - lo > max_diff:
            max_diff = A[i] - lo
        if A[i] < lo:
            lo = A[i]
    return lo, max_diff



def lastHigh(A):
    end = len(A)-1
    while A[end-1] > A[end]:
        end -= 1
    return end

def Platforms(Arr, Dep):
    Arr.sort()
    Dep.sort()
    
    plat = 1
    max_so_far = 1
    i = 1
    j = 0
    
    while (max(i, j) < len(Arr)):
        if Arr[i] <= Dep[j]:
            plat += 1
            i +=1
        elif Arr[i] > Dep[j]:
            plat -= 1
            j +=1
        max_so_far = max(plat, result)
    return max_so_far




class Graph:
    
    def __init__(self):
        self.graph = defaultdict(list)
        
    def addEdge(self, u, v):
        self.graph[u].append(v)
        
    def DFSUtil(self, v, visited):
        visited[v] = True
        
        for i in self.graph[v]:
            if visited[i] == False:
                self.DFSUtil(i, visited)
                
    def DFS(self, v):
        visited = [False] * (max(self.graph)+1)
        self.DFSUtil(v, visited)
           



def smallest(A, k):
    L = A[0:k]
    L.sort()
    for i in range(k, len(A)):
        bisect.insort(L, A[i])
        L.pop()
    return L



def Rooms(Starts, Ends):
    i = 1
    j = 0
    rooms = 1
    max_so_far = 1
    
    Starts.sort()
    Ends.sort()
    
    while (max(i,j) < len(Starts)):
        if Starts[i] <= Ends[j]:
            i += 1
            rooms = +1
        else:
            j = +1
            rooms = -1
        max_so_far = max(max_so_far, rooms)
        
    return max_so_far
    
    
class HeapNode:
    
    def __init__(self, val, row, col):
        self.val = val
        self.row = row
        self.col = col
    
def minHeapify(harr, i, heap_size):
    l = 2*i+1
    r = 2*r+2
    if l < heap_size and harr[l].val < harr[i].val:
        smallest = l
    if r < heap_size and harr_[r].val < harr[i].val:
        smallest = r
    if smallest != i:
        harr[i], harr[smallest] = harr[smallest], harr[i]
        minHeapify(harr, smallest, heap_size)
            
def buildHeap(harr, n):
    i = (n-1)//2
    while i >= 0:
        minHeapify(harr, i, n)
        i -= 1

def Smallest(Mat, k):
    n = len(Mat)
    harr = [0]*n
    for i in range(n):
        harr[i] = HeapNode(Mat[0][i],0,i)
    buildHeap(harr, n)
    
    hr = HeapNode(0,0,0)
    
    for i in range(k):
        hr = harr[0]
        nextval = Mat[hr.row + 1][hr.col + 1]
        harr[0] = HeapNode(nextval, hr.row + 1, hr.col)
        minHeapify(harr, 0, n)
    return hr.val






class Node:
    
    def __init__(self, key):
        self.data = key
        self.left = left
        self.right = right
        
def insert(node, data):
    if node is None:
        return Node.data

    else:
        if data <= node.data:
            node.left = insert(node.left, data)
        else:
            node.right = insert(node.right.data)
        
        return node
    
    
def minValue(node):
    current = node
    
    while (current.left is not None):
        current = current.left
    return current.data
    
    
    

def Equilibrium(A):
    current = 0
    left = 0
    right = sum(A) - A[0]
    
    for i in range(len(A)-1):
        if left == right:
            return current
        else:
            current +=1
            left = left + A[current-1]
            right = right - A[current]
    if current <= len(A)-2 or (current == len(A)-1 and sum(A) == A[-1]):
        return current
    else:
        return("No equilibrium")
        



class newNode:
    
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
    

def inroder(root):
    if root != None:
        inorder(root.left)
        inorder(root.right)
        
def insert(node, key):
    if node == None:
        return newNode(key)
    
    if key < node.key:
        node.left = insert(node.left, key)
    else:
        node.right = insert(node.right, key)
    return node

def oddNode(root):
    if root != None:
        oddNode(root.left)
        
        if root.key % 2 == 1:
            print(root.key, end = "")
        oddNode(root.right)



def depth(node):
    if node is None:
        return 0
    else:
        ldepth = depth(node.left)
        rdepth = depth.node
        return max(ldepth+1, rdepth+1)



class Node():
    def __init__(self, data):
        self.data = data
        self.next = None
        
class LinkedList:
     def __init__(self):
         self.head = None
    
    
     def reverse(self):
         prev = None
         current = self.head
         while current is not None:
             next = current.next
             current.next = prev
             prev = current
             current = next
         self.head = prev
         
        
             
     def reverse1(self):
         prev = None
         current = self.head
         while current is not None:
             next = current.next
             current.next = prev
             prev = current
             current = next
         self.head = prev
        
    








class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        
    def addEdge(self, u, v):
        self.graph[u].append[v]
        

    def BFS(self, s):
        visited = [False]*len(self.graph)
        
        Queue = []
        
        Queue.append(s)
        visited[s] = True
        
        while Queue != []:
            s = Queue.pop()
            
            for i in self.graph[s]:
                if visited[i] == False:
                    Queue.append(i)
                    visited[i] = True




class Graph1:
    def __init__(self):
        self.graph = defaultdict(list)
        
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def BFS(self, s):
        visited = [False] * len(self.graph)
        Q = []
        Q.append(s)
        visited[s] = True
        
        while Q:
            s = Q.pop(0)
            
            for i in self.graph[s]:
                if visited[i] == False:
                    Q.append(i)
                    visited[i] = True
                    
            
            


def flatten(iterable):
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in flatten(e):
                yield f
            else:
                yield e


def median(A):
    A.sort()
    if len(A) % 2 == 1:
        return A[int((len(A))/2 )]
    else:
        return (A[int((len(A)-1)/2)] + A[int((len(A)+1)/2)])/2
    
def med(A1, A2):

    n = len(A)
    if n == 0:
        return -1
    elif n == 1:
        return (A1[0]+A2[0])/2
    elif n == 2:
        return (max(A1[0], A2[0])+min(A1[1], A2[1]))/2
    
    else:
        m1 = median(A1)
        m2 = median(A2)
        
        if m1 == m2:
            return m1
        
        elif m1 > m2:
            if n % 2 == 0:
                return med(A1[:int(n/2)+1], A2[int(n/2)-1:])
            else:
                return med(A1[:int(n/2)+1], A2[int(n/2):])
        else:
            if n % 2 == 0:
                return med(A2[:int(n/2)+1], A1[int(n/2)-1:])
            else:
                return med(A2[:int(n/2)+1], A1[int(n/2):])


def Root(N, k):
    start = 1
    end = N
    
    while start <= end:
        mid = (start+end)//2
        if mid**k <= N <= (mid+1)**k:
            return mid
        elif (mid+1)**k <= N:
            end = mid+1
        else:
            start = mid-1
    



def Missing(A):
    visited = [0]*len(A)
    for i in range(len(A)):
        new = A[i]-1
        visited[new] = 1
    return [i+1 for i in range(len(A)) if visited[i] == 0]



class Node:
    def __init__(self, data):
        self.data = data
        self.left = left
        self.right = right
    
    def maxDepth(node):
        if node == None:
            return 0
        else:
            return max(maxDepth(node.left)+1, maxDepth(node.right)+1)

    def LCA(root, n1, n2):
        if root is None:
            return None
        if root.key == n1 or root.key == n2:
            return root
        if LCA(root.left,n1,n2) and LCA(root.right,n1,n2):
            return root
        
        if LCA(root.left,n1,n2) is not None:
            return LCA(root.left,n1,n2)
        else:
            return LCA(root.right,n1,n2)


# Detect cycles in digraph
            
        
def isCyclicUtil(self, v, visited, recStack):
    visited[v] = True
    recStack[v] = True
    
    for neighbor in self.graph[v]:
        if visited[neighbor] == False:
            if self.isCyclicUtil(neighbor, visited, recStack) == True:
                return True
            elif recStack[neighbor] == True:
                return True
    recStack[v] = False
    return False

def isCyclic(self):
    visited = [False] * self.V
    recStack = [False] * self.V
    for node in range(self.V):
        if visited[node] == False:
            if self.isCyclicUtil(node, visited, recStack) == True:
                return True
    return False




def Rooms(Starts, Ends):
    Starts.sort()
    Ends.sort()
    
    i = 1
    j = 0
    
    current = 1
    
    max_so_far = 1
    
    
    while max(i,j) < len(Starts):
        if Starts[i] <= Ends[j]:
            i+= 1
            current += 1
        else:
            j += 1
            current -= 1
        max_so_far = max(max_so_far, current)
        
    return max_so_far
    
    
def searchSortedGrid(Grid, x):
    
    i = 0
    j = len(G)-1
    
    while 0<= i <= len(Grid) and 0<= i <= len(Grid):
        if Grid[i][j] == x:
            return x
        elif Grid[i][j] < x:
            j -=1
        else:
            i += 1

def Rotateby90(Mat):
    N = len(Mat)    
    for i in range(N//2):
        for j in range(N-1-i):
            temp = Mat[i][j]
            Mat[i][j] = Mat[j][N-1-i]
            Mat[j][N-1-i] = Mat[N-1-i][N-1-j]
            Mat[N-1-i][N-1-j] = Mat[N-1-j][i]
            Mat[N-1-j][i] = temp
    return Represent(Mat)

M = [[1,2,3], [4,5,6], [7,8,9]]

print(Represent(M))

print(Rotateby90(M))


def isPalindrome(A):
    for i in range(len(A)//2):
        if A[i] != A[-i-1]:
            return False
    return True



# largest index j such that A[M[j]] <= A[i]

def largestIndex(A, M, i):
    
    lo = 0
    hi = len(M)
    
    while lo <= hi:
        mid = (lo+hi)//2
        if A[M[mid]] <= A[i] < A[M[mid+1]]:
            return mid
        elif A[M[mid]] > A[i]:
            hi = mid-1
        else:
            lo = mid+1
            
            
            
def longIncSub(A):

    P = [0]*len(A)
    M = [0]*(len(A)+1)

    L = 0

    for i in range(len(A)):
        j = largestIndex(A, M, i)
        newL = lo
        P[i] = M[newL-1]
        M[newL] = i
    
        if newL > L:
            L = newL

    S = [0]*L
    k = M[L]
    
    for i in range(L-1, -1, -1):
        S[i] = A[k]
        k = P[k]
        
    return S
            


# Moore's algorithm    

def findCandidate(A):    
    maj_ind = 0
    count = 1
    for i in range(len(A)):
        if A[maj_ind] == A[i]:
            count += 1
        else:
            count -= 1
        if count == 0:
            maj_ind = i
            count = 1
    return A[maj_ind]



def ignoreStartingZeros(A):
    start = 0
    while A[start] == 0:
        start += 1
    return A[start:]



        


def PartitionPalindromes(A):
    T = [[True for j in range(len(A))] for i in range(len(A))]
    C = [[len(A) for j in range(len(A))] for i in range(len(A))]
    
    for i in range(len(A)):
        C[i][i] = 0
        
    for i in range(len(A)-1):
        for j in range(i+1, len(A)):
            if j == i+1:
                T[i][j] = (A[i] == A[j])
            else:
                T[i][j] = T[i+1][j-1] and (A[i] == A[j])
        
    
    for i in range(len(A)-1):
        for j in range(i+1, len(A)):
            T[i][j] = T[i+1][j-1] and (A[i] == A[j])
            if T[i][j] == True:
                C[i][j] = 0
            else:
                C[i][j] = min([C[i][k] + C[k+1][j] + 1 for k in range(i+1)])
    

    return C[0][len(A)-1]    




        
    





def lastIndex(A, x):
    end = len(A)-1
    while A[end] !=x:
        end -= 1
    return end




def Maxes(A):
    M = []
    end = len(A)-1
    while end >= 1 and A[end-1] > A[end]:
        end -= 1
    A = A[:end+1]
    if len(A) == 0:
        return "Never buy"
    else:
        last = -1
        while last <= len(A)-2:
            m = max(A[last+1:])
            last = lastIndex(A, m)
            M.append([m, last])
            
    return M





def RussianDoll(A):
    A = sorted(A, key = lambda  x : x[0])
    L = longIncSub(A)
    return L



def pivotPlace(A):
    N = len(A)
    lo = 0
    hi = N-1
    while lo <= hi:
        mid = (lo+hi)//2
        if A[mid] < min(A[(mid-1)%N], A[(mid+1)%N]):
            return mid
        else:
            if A[0] < A[mid]:
                lo = mid+1
            else:
                hi = mid-1
                
                
                
L = [1,2,3,4,5,6,7,8,9,10]

L = [L[(i-4) % 10] for i in range(10)]

print(L)
print(pivotPlace(L))

                                


# Missing integer from [1,n] is a sorted array

def Missing(A):
    lo = 0
    hi = len(A)-1
    
    while lo <= hi:
        mid = (lo+hi)//2
        if (A[mid] > mid+1) and (mid == 0 or A[mid-1] == mid):
            return mid+1
        elif A[mid] == mid+1:
            lo = mid+1
        else:
            hi = mid-1
            
            
            
def Multiples(a,b,N):
    L = (a*b)//GCD(a,b)
    return (N//a) + (N//b) - (N//L)

def Magic(a,b,n):
    L = (a*b)//GCD(a,b)
    d = GCD(a,b)
    N = math.ceil(n*(1/(1/a + 1/b - 1/L)))
    next_a = N + a - N%a
    next_b = N + b - N%b
    return min(next_a, next_b)



# A[i] = candidate voted for at time T[i]. Two candidates only

def Leaders(A, T):
    counter = 0
    Leader = []
    for i in range(0, len(A)):
        if A[i] == 1:
            counter += 1
        else:
            counter -= 1
        if counter < 0:
            Leader.append(0)
        elif counter > 0:
            Leader.append(1)
        else:
            Leader.append('-')
    
    return Leader
            
            
def LeadersQuery(A, T, Q):
    L = Leaders(A, T)
    L1 = []
    for i in range(len(Q)):
        j = findIndex(T, Q[i])
        L1.append(L[j])
    return L1











def BuySell(A):
    buy = A[0]
    sell = A[1]
    profit = sell - buy
    
    for i in range(1, len(A)):
        if i< len(A) and A[i] < buy:
            buy = A[i]
            sell = A[i+1]
            profit = max(profit, sell-buy)
        elif A[i] > sell:
            sell = A[i]
            profit = max(profit, sell - buy)
    
    return profit
   



# largest index j such that A[I[j]] < x

def findIndex(A, I, x):
    lo = 0
    hi = len(I)-1
    
    while lo <= hi:
        mid = (lo+hi)//2
        if A[I[mid]] <= x <= A[I[mid]]:
            return mid
        elif A[I[mid]] > x:
            hi = mid-1
        else:
            lo = mid+1
    
    
def check(A, m, k):
    count = 0
    Sum = 0
    current = 0
    for i in range(1, len(A)):
        current += 1
        Sum += A[i]
        if Sum > m:
            count +=1
            Sum = A[i]
            
    return (count <= k)
    

def minMaxSumPart(A, k):
    lo = 0
    hi = sum(A)
    while lo <= hi:
        mid = (lo+hi)//2
        if check(A, mid, k) == True and check(A, mid-1, k) == False:
            return mid        
        elif check(A, mid, k) == False:
            lo = mid+1
        else:
            hi = mid - 1
    
    

def closer(a, b, x):
    if abs(x-a) == abs(x-b):
        return min(a,b)
    elif abs(x-a) <= abs(x-b):
        return a
    else:
        return b
    


def findPositionFrom(A, x, a):
    lo = 0
    hi = len(A)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if (mid< len(A)-1) and ((closer(A[mid], a, x) == A[mid]) and (closer(A[mid+1], a, x) == a)):
            return mid+1
        elif mid == len(A)-1 and (closer(A[mid], a, x) == a):
            return mid+1
        elif (closer(A[mid], a, x) == a):
            hi = mid-1
        else:
            lo = mid+1


def func(x):
    return None


def findIndexModified(A, a, func):
    lo = 0
    hi = len(A)-1
    
    while lo <= hi:
        mid = (lo+hi)//2
        if (mid<=len(A)-2) and func(A[mid])<= func(a) < func(A[mid+1]):
            return mid+1
        elif (mid == len(A)-1) and func(A[mid])<= func(a):
            return mid+1
        elif func(A[mid+1])<= func(a):
            lo = mid+1
        else:
            hi = mid-1
            
            
    
def firstMissing(A):
    lo = 0
    hi = len(A)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if (mid>=1) and (A[mid] > mid and A[mid-1] == mid-1):
            return mid
        elif mid == 0 and A[mid] == 0:
            return 0
        elif A[mid] == mid:
            lo = mid+1
        else:
            hi = mid-1
            
            

def Anagram(A, B):
    return (sorted(A) == sorted(B))           
    

def isSubsequence(S, T):
    
    A = [[True for _ in range(len(T)+1)] for _ in range(len(S)+1)]
    
    A[0][0] = (S[0] == T[0])
    
    
    for i in range(0, len(S)):
        for j in range(0, len(T)):
            if (i,j) != (0,0):
                A[i][j] = A[i-1][j] or (A[i-1][j-1] and (S[i]==T[j]) )
    
    return A[len(S)-1][len(T)-1]
    
    
def subsetSum(A, n):
    N = len(A)
    M = [[False for _ in range(n+1)] for _ in range(N)]
    
    for j in range(n+1):
        M[0][j] = (A[0] == j)
    for i in range(N):
        if i == 0:
            M[i][0] = A[0] == 0
        else:
            M[i][0] = M[i-1][0] or (A[i] == 0)
    
    
    for i in range(1, N):
        for j in range(n+1):
            M[i][j] = M[i-1][j] or (j>= A[i]  and  M[i-1][j-A[i]])
    
    return M[N-1][n]




def twoEqualParts(A):
    if sum(A)%2 == 1:
        return False
    else:
        return subsetSum(A, sum(A)//2)
    
    
def neighbors(Grid, i,j):
    return [(i-x, i-y) for x in range(0,2) for y in range(0,2) if (0<= i-x < len(Grid) and 0<= i-y < len(Grid[0]) )   ]

    


def Interleaving(A, B, C):
    T = [[True for _ in range(len(B))] for _ in range(len(A))]
    T[0][0] = ([C[0], C[1]] == [A[0], B[1]] or [C[0], C[1]] == [B[0], A[1]])
    
    for i in range(1, len(A)):
        T[i][0] =  (C[i+1]== A[i] and T[i-1][0]) or (C[i+1] == B[0] and C[:i+1] == A[:i+1])
    
    for j in range(len(B)):
        T[0][j] = (C[j+1]== B[j] and T[0][j-1]) or (C[j+1] == A[0] and C[:j+1] == B[:i+1])
    
    
    for i in range(1, len(A)):
        for j in range(1, len(B)):
            
                T[i][j] = (C[i+j+1] == A[i] and T[i-1][j]) or (C[i+j+1] == B[j] and T[i][j-1])
    


def maxOverlap(Starts, Ends):
    Starts.sort()
    Ends.sort()
    
    i = 1
    j = 0
    current = 1
    max_so_far = 1
    
    
    while max(i,j) < len(Starts):
        if Starts[i] >= Ends[j]:
            i += 1
            current += 1
            max_so_far = max(max_so_far, current)
        else:
            j += 1
            current -= 1
            max_so_far = max(max_so_far, current)
            
    return max_so_far 
        
print(" ")


def numDivOr(N, a, b):
    lcm = (a*b)//GCD(a,b)
    return (N//a) + (N//b) - (N//lcm)

def MultipleOr(n, a, b):
    lo = 1
    hi = a*b*n
    
    while lo <= hi:
        mid = (lo+hi)//2
        if numDivOr(mid, a, b) == n:
            return max(mid - (mid % a), mid - (mid % b))
        elif numDivOr(mid, a, b) < n:
            lo = mid+1
        else:
            hi = mid-1
    
    

def lastMultiples(A, n, k):

    ind = 0
    S = []
    
    while len(S) <= k and ind < len(A):
        if A[ind] % n == 0:
            S.append(ind)
        ind += 1
        
    for i in range(S[-1]+1, len(A)):
        if A[i] % n == 0:
            S.pop(0)
            S.append(i)
            
    return S
        
        


def leftmost(T, root):
    node = root
    while node:
        node = node.left
        
def rightmost(T, root):
    node = root 
    while node:
        node = node.right
        
def isBST(T, root):
    
    LeftMax = rightmost(T, root.left)
    RightMin = leftmost(T, root.right)
    
    if root is None:
        return True
    
    else:
        return (LeftMax.key < root.key < RightMin.key) and (isBST(root.left)) and (isBST(root.right))







def waterAccumulation(A):
    
    max_left = [0]*len(A)
    max_right = [0]*len(A)
    
    max_left[0] = 0
    max_right[len(A)-1] = 0
    for i in range(1, len(A)):
        max_left[i] = max(max_left[i], A[i-1])
    for i in range(len(A)-2, -1, -1):
        max_right[i] = max(max_right[i], A[i+1])
    Area = sum([min(max_left[i], max_right[i]) for i in range(len(A))])

    return Area


def Balanced(A):
    S = []
    for x in A:
        if x < 0:
            S.append(x)
        else:
            if x+S[-1]==0:
                S.pop()
            else:
                return False
    if len(S) != 0:
        return False
    return True

def findSortedMatrix(M, x):
        
        i = 0
        j = len(M)-1
        
        while i < len(M) and j >= 0:
            if M[i][j] == x:
                return (i,j)
            elif M[i][j] < x:
                j -= 1
            else:
                i += 1


def missingIntegers(A):
    N = len(A)
    for i in range(len(A)):
        new = A[i]
        A[new] = new
    Missing = []
    for i in range(len(A)):
        if A[i] != i:
            Missing.append(i)
    return Missing 



def Equilibrium(A):
    
    current = 0
    Sum = A[0]
    
    while current <= len(A)-1 and 2*Sum + A[current] != sum(A):
        current += 1
    if current == len(A):
        return -1
    else:
        return current
        
    
    
    
def Arrange(A):
    m = max([(math.ceil(math.log10(x))) for x in A])
    A = sorted(A, key  = lambda x: x*10**(m-math.ceil(math.log10(x))))
    A = list(reversed(A))
    return A


def sort012(A):
    Zeros = 0
    Ones = 0
    Twos = 0
    
    for x in A:
        if x==0:
            Zeros +=1
        elif x == 1:
            Ones +=1
        else:
            Two += 1
        
    return [0]*Zeros + [1]*Ones + [2]*Twos
        
    
    

    
def sumPairs(A, n):
    A.sort()
    current = 0
    counter = 0
    while A[current] <= (n//2):
        if binarySearch(A[current+1: ], n-A[current]) != -1:
            counter +=1
        current += 1
    return counter


def negativesFirst(A):
    pos = []
    neg = []
    
    for x in A:
        if x >= 0:
            pos.append(x)
        else:
            neg.append(x)
    return neg+pos



def logarithm(N, n):
    k = 0
    while n**k < N:
        k += 1
    return k

def closeToTarget(A, t):
    closest_so_far = A[0]*len(A)
    sum_so_far = 0

    for i in range(1, len(A)):
        sum_so_far = sum_so_far + A[i-1]
        closest_so_far = Closer(closest_so_far, sum_so_far + A[i]*(len(A)-i), n)
    
    return closest_so_far

def largerFraction(a, b):
    if a[0]*b[1] >= a[1]*b[0]:
        return a
    else:
        return b
    
    

    
def Leaders(A):
    L = []
    m = max(A)
    L.append(m)
    if A.index(m) == len(A)-1:
        return L
    else:
        L.extend(Leaders(A[A.index(m)+1:]))
        return L
    



def numberofMultiples(a,b,c, N):
    L1 = lcm(a,b)
    L2 = lcm(b,c)
    L3 = lcm(b,c)
    L4 = ((a*b*c)*GCD(GCD(a,b), c))/(GCD(a,b)*GCD(b,c)*GCD(c,a))
    
    return (N//a) + (N//b) + (N//c) - (N//L1) - (N//L2) - (N//L3) + (N//L123)

def multiple(a,b,c, n):
    lo = 0
    hi = min(a,b,c)*n
    
    while lo <= hi:
        mid = (lo+hi)//2
        if numberofMultiples(a,b,c, mid) == n:
            return max(n - n%a, n - n%b, n - n%c)
        elif numberofMultiples(a,b,c, mid) < n:
            lo = mid + 1
        else:
            hi = mid-1


    






    
















