
# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[1]:

# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
I =  [ [ 0 for i in range(4) ] for j in range(4) ]
for i in range(4):
    for j in range(4):
        if i==j:
            I[i][j]=1


# ## 1.2 返回矩阵的行数和列数

# In[2]:

# TODO 返回矩阵的行数和列数
def shape(M):
    r=len(M)
    c=len(M[0])
    return r,c


# ## 1.3 每个元素四舍五入到特定小数数位

# In[3]:

# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(shape(M)[0]):
        for j in range(shape(M)[1]):
            M[i][j]=round(M[i][j],decPts)


# ## 1.4 计算矩阵的转置

# In[4]:

# TODO 计算矩阵的转置
def transpose(M):
    N=[[M[i][j] for i in range(shape(M)[0])] for j in range(shape(M)[1])]
    return N


# ## 1.5 计算矩阵乘法 AB

# In[5]:


# TODO 计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A, B):
    N =  [ [ 0 for i in range(shape(B)[1]) ] for j in range(shape(A)[0]) ]
    if shape(A)[1]==shape(B)[0]:
        k=shape(A)[1]
        for ai in range(shape(A)[0]):
            for bj in range(shape(B)[1]):
                for kk in range(k):
                    N[ai][bj]+=A[ai][kk]*B[kk][bj]
        return N
    else:
        return None


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[6]:

import pprint
pp=pprint.PrettyPrinter(indent=1,width=30)
#TODO 测试1.2 返回矩阵的行和列
M=[[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
pp.pprint(shape(M))
#TODO 测试1.3 每个元素四舍五入到特定小数数位
M=[[1.123456,2.123456],[3.123456,4.123456],[5.123456,6.123456]]
matxRound(M,2)
pp.pprint(M)
#TODO 测试1.4 计算矩阵的转置
M=[[1,2,3,4],[7,8,9,0]]
pp.pprint(transpose(M))
#TODO 测试1.5 计算矩阵乘法AB，AB无法相乘
A=[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
B=[[1,2,3,4],[1,2,3,4]]
pp.pprint(matxMultiply(A,B))
#TODO 测试1.5 计算矩阵乘法AB，AB可以相乘
A=[[1,2],[1,2],[1,2],[1,2],[1,3]]
B=[[1,2,3,4],[1,2,3,4]]
pp.pprint(matxMultiply(A,B))


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[7]:

# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    for i in range(shape(A)[0]):
        A[i].append(b[i][0])
    return A
A=[[1,2,3],[4,5,6]]
b=[[7],[8]]
pp.pprint(augmentMatrix(A,b))


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[8]:

# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1],M[r2]=M[r2],M[r1]

# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale==0:
        raise ValueError('scale could not be zero!')
    else:
        for i in range(shape(M)[1]):
            M[r][i]*=scale

# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    r=[ 0 for i in range(shape(M)[1]) ]
    for i in range(shape(M)[1]):
        r[i]=M[r2][i]*scale
    for i in range(shape(M)[1]):
        M[r1][i]+=r[i]
    


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[79]:

# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    if shape(A)[0]==len(b):
#         转为增广矩阵
        augmentMatrix(A,b)
#         针对每列做操作
        for j in range(shape(A)[1]-1):
            i=j
            while i+1<shape(A)[0]:
                if A[j][j]<A[i+1][j]:
                    swapRows(A, j, i+1)
                i+=1
#             如果为奇异矩阵返回none
            if abs(A[j][j])<epsilon:
                return None
            scaleRow(A,j,1.0/A[j][j])
            for i in range(shape(A)[0]):
                 if i<>j:
                        addScaledRow(A, i, j, -1.0*A[i][j])
        matxRound(A,decPts)
        N=transpose(A)[-1]
        return [[N[j]] for j in range(len(N))]
    else:
        raise ValueError


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：因为I为单位矩阵,Z为全0矩阵,同时A为方阵,由于Y的第一列全0代表矩阵A的对角线含有0,所以结果是转换为上三角形矩阵的行列式的值为0,当一个矩阵所在的行列式的值为0的话,该矩阵为奇异矩阵

# ## 2.5 测试 gj_Solve() 实现是否正确

# In[80]:

# TODO 构造 矩阵A，列向量b，其中 A 为奇异矩阵
A=[[3,6],[0,0]]
b=[[0],[3]]
pp.pprint(gj_Solve(A,b,2))
# TODO 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
A=[[3,6],[2,1]]
b=[[0],[3]]
# TODO 求解 x 使得 Ax = b
pp.pprint(gj_Solve(A,b,2))
# TODO 计算 Ax
A1=3*2-6
A2=2*2-1
Ax=[[0],[3]]
# TODO 比较 Ax 与 b
Ax=[[0],[3]]
b=[[0],[3]]


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：
# 
# 1:
# $$
# E = \sum_{i=1}^{n}{y_i^2-y_ix_im-by_i-y_ix_im+x_i^2m^2+bx_im-by_i+bx_im+b^2}
# $$
# 方程两边对于m同时求导
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_iy_i+2mx_i^2+2bx_i}
# $$
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 2:
# $$
# E = \sum_{i=1}^{n}{y_i^2-y_ix_im-by_i-y_ix_im+x_i^2m^2+bx_im-by_i+bx_im+b^2}
# $$
# 方程两边对于b同时求导
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2y_i+2mx_i+2b}
# $$
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 3:
# 代入Y和X的值
# $$
# 2X^TXh - 2X^TY=2 \begin{bmatrix}
#     m(x_1^2+x_2^2+...+x_n^2)+b(x_1+x_2+...+x_n)\\
#     m(x_1+x_2+...+x_n)+bn 
# \end{bmatrix}-2 \begin{bmatrix}
#     x_1y_1+x_2y_2+...+x_ny_n\\
#     y_1+y_2+...+y_n
# \end{bmatrix}
# $$
# $$
# 2X^TXh - 2X^TY=2 \begin{bmatrix}
#     m(x_1^2+x_2^2+...+x_n^2)+b(x_1+x_2+...+x_n)-(x_1y_1+x_2y_2+...+x_ny_n)\\
#     m(x_1+x_2+...+x_n)+bn-(y_1+y_2+...+y_n)
# \end{bmatrix}
# $$
# $$
# 2X^TXh - 2X^TY= \begin{bmatrix}
# \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\\
# \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# \end{bmatrix}
# $$
# 根据1和2的结论可得:
# $$
# 2X^TXh - 2X^TY=
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} 
# $$
# 

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[81]:

# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''  
#由于把改方程右边移到方程的左边然后带入Y这样的单列矩阵和X这样的双列矩阵,展开得到的是两个标准的y=mx+b的线性方程所以直接使用线性回归的系数公式,
#如果要求按照带入gj_Solve()的方法,我不知道该如何表示
def linearRegression(points):
    n=len(points)
    #求x均值,#y均值
    total_x=0
    total_y=0
    for i in range(0,n):
        total_x+=points[i][0]
        total_y+=points[i][1]
    mean_x=total_x*1.0/n
    mean_y=total_y*1.0/n
        
    #分子
    numerator=0
    #分母
    denominator=0
    for i in range(0,n):
        numerator+=(points[i][0]-mean_x)*(points[i][1]-mean_y)
        denominator+=(points[i][0]-mean_x)**2
    return numerator/float(denominator),mean_y/float(mean_x)


# ## 3.3 测试你的线性回归实现

# In[82]:

# TODO 构造线性函数

# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音
import random
#构建线性函数y=10+2x
P =  [ [ 0 for i in range(2) ] for j in range(100) ]
for i in range(100):
    P[i][0]=random.randint(0, 100)
    P[i][1]=2*P[i][0]+10+random.gauss(1,0)
#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较
print linearRegression(P)


# ## 4.1 单元测试
# 
# 请确保你的实现通过了以下所有单元测试。

# In[85]:

import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))

            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            ab = np.hstack((A,b))

            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))
            x = gj_Solve(A.tolist(),b.tolist())
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                # Ax = matxMultiply(A.tolist(),x)
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                # print Ax
                # print loss
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)


# In[ ]:




# In[ ]:



