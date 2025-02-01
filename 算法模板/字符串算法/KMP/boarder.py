s = input()
n = len(s)
Next = [0] * (n+1)
for i in range(1, n):
  j = Next[i]
  while j > 0 and s[i] != s[j]:
    j = Next[j]
  if s[i] == s[j]:
    Next[i + 1] = j + 1
  else:
    Next[i + 1] = 0
arr = []
for x in range(1,n+1):
  if 2*Next[x] <= x:
    arr.append(Next[x])
print(Next)
print(max(arr))
