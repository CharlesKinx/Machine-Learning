
## 聚合操作


```python
import numpy as np
```


```python
L  = np.random.random(100)
```


```python
L
```




    array([0.43491726, 0.66633001, 0.086582  , 0.54397958, 0.12086773,
           0.51026314, 0.50574478, 0.33817863, 0.00342827, 0.99729214,
           0.30902156, 0.07224728, 0.42056901, 0.62650721, 0.47589286,
           0.07737593, 0.89387912, 0.18926556, 0.18375608, 0.62600246,
           0.20388229, 0.80527943, 0.28390067, 0.55556364, 0.26676501,
           0.94482744, 0.64346107, 0.35936027, 0.40098094, 0.24336138,
           0.009993  , 0.12962035, 0.32680232, 0.71534336, 0.73847278,
           0.94117629, 0.96097282, 0.59123102, 0.28749634, 0.04961379,
           0.47029322, 0.74189512, 0.82488857, 0.70179623, 0.17836792,
           0.47188544, 0.47329884, 0.18073013, 0.72712879, 0.51366985,
           0.71954345, 0.93816003, 0.49447764, 0.17697059, 0.19001757,
           0.04471757, 0.47066029, 0.06984886, 0.29646332, 0.28234052,
           0.23842714, 0.50022   , 0.27626705, 0.98998247, 0.30208272,
           0.24629768, 0.47464441, 0.44413759, 0.06238919, 0.12060324,
           0.47711408, 0.33221397, 0.04480951, 0.04723282, 0.64620365,
           0.35631147, 0.66376741, 0.55264799, 0.56511135, 0.40853604,
           0.9909795 , 0.94437189, 0.64565178, 0.57304719, 0.61151188,
           0.14320454, 0.56179195, 0.60670603, 0.70314226, 0.95253464,
           0.04262608, 0.36173386, 0.47614896, 0.61983389, 0.16906909,
           0.65437659, 0.8225802 , 0.4562654 , 0.27664525, 0.5830861 ])




```python
sum(L)
```




    45.45166562388123




```python
np.sum(L)
```




    45.45166562388124




```python
%%time
sum(L)
```

    Wall time: 0 ns
    




    45.45166562388123




```python
L.max()
```




    0.997292137801767




```python
L.min()
```




    0.003428272919851927




```python
X = np.arange(16).reshape(4,-1)
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])




```python
np.sum(X)
```




    120




```python
np.sum(X,axis = 0) #沿着行进行计算
```




    array([24, 28, 32, 36])




```python
np.sum(X, axis = 1)# axis这参数沿着列这个方向进行计算
```




    array([ 6, 22, 38, 54])




```python
np.prod(X)#返回的是所有的数乘积
```




    0




```python
np.mean(X)#求平均数
```




    7.5




```python
np.median(X)#求中位数
```




    7.5




```python
np.percentile(X, q= 50)#百分位,q是百分位
```




    7.5




```python
np.var(X)#方差
```




    21.25




```python
np.std(X)#标准差
```




    4.6097722286464435




```python
x = np.random.normal(0, 1, size = 1000000)
```


```python
np.mean(x)
```




    -0.001020943108495056



## 索引


```python
np.min(x)
```




    -4.643041484802793




```python
np.argmin(x)#索引值，324090位最小值
```




    324090




```python
x[324090]
```




    -4.643041484802793




```python
np.argmax(x)
```




    62526




```python
x[62526]
```




    4.748071062112505




```python
np.max(x)
```




    4.748071062112505



### 排序和使用索引


```python
x = np.arange(16)
x
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
np.random.shuffle(x)#乱序
x
```




    array([ 3,  5,  2,  7,  1, 13, 15,  9,  4, 12, 10,  6,  0, 11, 14,  8])




```python
np.sort(x)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
x.sort()
```


```python
x
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
X = np.random.randint(10,size=(4,4))
X
```




    array([[6, 2, 6, 6],
           [3, 9, 1, 2],
           [2, 9, 0, 0],
           [1, 0, 4, 3]])




```python
np.sort(X)
```




    array([[2, 6, 6, 6],
           [1, 2, 3, 9],
           [0, 0, 2, 9],
           [0, 1, 3, 4]])




```python
np.sort(X, axis = 1)
```




    array([[2, 6, 6, 6],
           [1, 2, 3, 9],
           [0, 0, 2, 9],
           [0, 1, 3, 4]])




```python
np.sort(X, axis=0)
```




    array([[1, 0, 0, 0],
           [2, 2, 1, 2],
           [3, 9, 4, 3],
           [6, 9, 6, 6]])




```python
x
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
np.random.shuffle(x)
x
```




    array([ 7, 15, 12, 11, 10, 13,  1,  9,  4,  3, 14,  6,  8,  5,  2,  0])




```python
np.argsort(x) #索引数组，从小到大排序
```




    array([15,  6, 14,  9,  8, 13, 11,  0, 12,  7,  4,  3,  2,  5, 10,  1],
          dtype=int64)




```python
np.partition(x, 3)#寻找一个标定点，小于这个标定点的都放在左侧，大于都放在右侧
```




    array([ 0,  1,  2,  3,  4,  6,  5,  7, 15, 13, 14, 10,  8, 11, 12,  9])




```python
np.argpartition(x,3)#返回标定点以及比标定点大或小的索引
```




    array([15,  6, 14,  9,  8, 11, 13,  0,  1,  5, 10,  4, 12,  3,  2,  7],
          dtype=int64)




```python
X
```




    array([[6, 2, 6, 6],
           [3, 9, 1, 2],
           [2, 9, 0, 0],
           [1, 0, 4, 3]])




```python
np.argsort(X,axis = 1)
```




    array([[1, 0, 2, 3],
           [2, 3, 0, 1],
           [2, 3, 0, 1],
           [1, 0, 3, 2]], dtype=int64)




```python
np.argpartition(X,2,axis = 1)
```




    array([[1, 0, 2, 3],
           [2, 3, 0, 1],
           [2, 3, 0, 1],
           [1, 0, 3, 2]], dtype=int64)



## Fancy Indexing


```python
x = np.arange(16)
x
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
x[3]
```




    3




```python
x[3:9] #切片索引
```




    array([3, 4, 5, 6, 7, 8])




```python
x[2:6:2]
```




    array([2, 4])




```python
ind = [3, 5, 8]
```


```python
x[ind]
```




    array([3, 5, 8])




```python
ind = np.array([[0,2],
               [1,2]])
x[ind]
```




    array([[0, 2],
           [1, 2]])




```python
X = x.reshape(4,-1)
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])




```python
row = np.array([0,1,2])
col = np.array([1,2,3])
X[row,col]
```




    array([ 1,  6, 11])




```python
X[:2,col]
```




    array([[1, 2, 3],
           [5, 6, 7]])




```python
col = [True, False, True, False]
```


```python
X[1:3,col]
```




    array([[ 4,  6],
           [ 8, 10]])



## numpy.array的比较


```python
x
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
x < 3
```




    array([ True,  True,  True, False, False, False, False, False, False,
           False, False, False, False, False, False, False])




```python
x>3
```




    array([False, False, False, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True])




```python
2*x == 24 - 4*x
```




    array([False, False, False, False,  True, False, False, False, False,
           False, False, False, False, False, False, False])




```python
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])




```python
X<6
```




    array([[ True,  True,  True,  True],
           [ True,  True, False, False],
           [False, False, False, False],
           [False, False, False, False]])




```python
x
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
np.sum(x <= 3)
```




    4




```python
np.count_nonzero(x <=3)
```




    4




```python
np.any(x == 0)
```




    True




```python
np.any(x<0)
```




    False




```python
np.all(x>=0)
```




    True




```python
np.all(x>0)
```




    False




```python
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])




```python
np.sum(x%2 == 0)
```




    8




```python
np.sum(X%2 == 0,axis = 1)
```




    array([2, 2, 2, 2])




```python
np.sum(X>0,axis = 1)
```




    array([3, 4, 4, 4])




```python
x
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
np.sum((x>3)&(x<6))
```




    2




```python
np.sum((x%2 ==0) | (x>10))
```




    11




```python
np.sum(~(x==0))
```




    15




```python
x[x<5]
```




    array([0, 1, 2, 3, 4])




```python
np.random.shuffle(x)
```


```python
x[x%2==0]
```




    array([ 8,  6, 12, 14, 10,  2,  0,  4])




```python
np.sort(x)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
x
```




    array([ 1,  3,  9,  5, 11,  8,  6, 12,  7, 14, 10, 15,  2,  0, 13,  4])




```python
x.sort()
```


```python
x
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])




```python
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])




```python
X[X[:,3]%3 ==0,:]
```




    array([[ 0,  1,  2,  3],
           [12, 13, 14, 15]])


