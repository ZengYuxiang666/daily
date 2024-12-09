# 相向双指针就是找到一个合适的值，大了right就往左走，小了left就往右走
# 四指针就是固定住两个指针，让left和right去移动
# https://leetcode.cn/problems/4sum
class Solution(object):
    def fourSum(self, nums, target):  # 四指针
        nums.sort()
        n = len(nums)
        list1 = []
        for x in range(n - 3):
            if x > 0 and nums[x] == nums[x - 1]:
                continue
            for y in range(x + 1, n - 2):
                left = y + 1
                right = n - 1
                while left < right:
                    if left > y + 1 and nums[left] == nums[left - 1]:
                        left += 1
                        continue
                    elif right < n - 1 and nums[right] == nums[right + 1]:
                        right -= 1
                        continue
                    sum = nums[x] + nums[y] + nums[left] + nums[right]
                    if sum > target:
                        right -= 1
                    elif sum < target:
                        left += 1
                    else:
                        list1.append([nums[x], nums[y], nums[left], nums[right]])
                        left += 1
                        right -= 1
        unique_list = list(map(list, set(tuple(sorted(sublist)) for sublist in list1)))
        return unique_list
