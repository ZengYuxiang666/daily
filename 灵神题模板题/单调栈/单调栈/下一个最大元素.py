# 求最短
class Solution(object):
    def nextGreaterElements(self, nums):
        stack= []
        lst = [-1 for i in range(len(nums))]
        nums+=nums
        for right in range(len(nums)):
            while stack!=[] and nums[right] > nums[stack[-1]]:
                if stack[-1] < len(nums)//2:
                    lst[stack[-1]] = nums[right]
                stack.pop()
            stack.append(right)
        return  lst