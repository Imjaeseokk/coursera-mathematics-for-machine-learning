# Mathematics for Machine Learning: Linear Algebra

Coursera - Imperial College London

Link to [Course](https://www.coursera.org/learn/linear-algebra-machine-learning)

## Matrices, Vectors, and solving simultaneous equation problems

### Apples and Bananas 문제 복습
- 예를 들어, 어떤 가게에서 사과 2개, 바나나 3개를 사는데 8유로가 들었다고 하자.
- 다른 날에는 사과 10개, 바나나 1개를 사는데 13유로가 들었다.
- 이때 사과 1개, 바나나 1개의 가격을 알아내는 것이 목표.

이 문제는 단순한 simultaneous equations(연립 방정식) 문제이지만, **Matrices(행렬)** 을 이용해서 표현할 수 있다.

### Matrix 표현
다음과 같은 형태로 행렬을 설정할 수 있다:

- Coefficient Matrix:
$$
\begin{bmatrix}
2 & 3 \\
10 & 1
\end{bmatrix}
$$

- Variable Vector:

$$
\begin{bmatrix}
a \\
b
\end{bmatrix}
$$

- Result Vector:

$$
\begin{bmatrix}
8 \\
13
\end{bmatrix}
$$

이것은 다음 연립방정식과 동일함:

$$
2a + 3b = 8, \quad 10a + 1b = 13
$$

### Matrix의 연산 방법
- Matrix와 Vector를 곱할 때는 **Row × Column** 연산을 한다.
- 첫 번째 행을 변수 벡터와 내적하여 결과 벡터의 첫 번째 원소를 얻고, 두 번째 행으로 두 번째 원소를 얻는다.
- 즉, 
  - 첫 번째 행: \(2a + 3b = 8\)
  - 두 번째 행: \(10a + 1b = 13\)

→ 이는 기존 연립방정식과 정확히 일치한다.

### Matrix의 의미
- 행렬은 벡터를 **회전(rotation)** 하고 **늘리거나 줄이는(stretch)** 연산을 수행하는 도구이다.
- 또한 행렬은 벡터 공간을 변환하여 문제를 해결하는 데 사용된다.
- 즉, 행렬은 **입력 벡터(input vector)** 를 받아 **출력 벡터(output vector)** 로 변환하는 함수(function)처럼 동작한다.


### Basis Vector의 변환
- 표준 단위벡터 \( e_1 = [1,0] \), \( e_2 = [0,1] \)을 생각해보자.

행렬을 \( e_1 \)에 곱하면:

- \( 2 \times 1 + 3 \times 0 = 2 \)
- \( 10 \times 1 + 1 \times 0 = 10 \)

결과:

$$
\begin{bmatrix}
2 \\
10
\end{bmatrix}
$$

행렬을 \( e_2 \)에 곱하면:

$$
\begin{aligned}
2 \times 0 + 3 \times 1 &= 3 \\
10 \times 0 + 1 \times 1 &= 1
\end{aligned}
$$

결과:

$$
\begin{bmatrix}
3 \\
1
\end{bmatrix}
$$

→ 행렬은 단위벡터들을 새로운 벡터로 **변환(transform)** 한다.

---


### 선형대수(Linear Algebra)의 의미
- **Linear(선형성)** : 입력 값을 상수로 곱하고 더하는 방식으로 변환한다는 의미. (비틀거나 접지 않는다.)
- **Algebra(대수)** : 수학적 기호를 다루고 조작하는 체계(system).

즉, **Linear Algebra**는
- 벡터와 벡터 공간을 수학적으로 다루고 변환하는 방법론이다.
- 동시에, 연립방정식을 푸는 것도 결국 벡터 공간을 변환하는 과정임을 알 수 있다.

### Summary
- 사과와 바나나 문제는 연립방정식으로, 이를 **행렬**을 이용해 표현할 수 있다.
- 행렬은 입력 벡터를 받아서 출력 벡터로 변환한다.
- 행렬을 단위벡터에 곱하면, 새로운 위치로 이동된 벡터를 얻을 수 있다.
- **Linear Algebra**는 벡터와 그 변환에 관한 체계이며, 이 개념은 문제 해결에 필수적이다.


## How Matrices Transform Space

### Matrix와 공간 변환의 관계
- 이전에는 Matrix를 이용해 연립방정식을 표현하는 방법을 배웠고,  
- Matrix의 각 열은 단위벡터 \( e_1, e_2 \)에 대해 어떤 변화를 주는지를 나타낸다고 했음.

이번에는 다양한 종류의 Matrix가 공간에 어떤 변화를 일으키는지,  
그리고 하나의 Matrix 변환 뒤에 또 다른 Matrix를 적용하는 **composition(합성)** 을 다룬다.

---

### 벡터 변환과 선형성

벡터 \( r \)을 Matrix \( A \)로 변환한다고 하자:

$$
A r = r'
$$

이때, 중요한 성질은 다음과 같다:

- \( r \)에 어떤 스칼라 \( n \)을 곱한 후 변환하면:

$$
A(nr) = n(Ar) = nr'
$$

- 두 벡터의 합을 변환하면:

$$
A(r+s) = Ar + As
$$

즉,
- **Matrix 변환은 스칼라 곱과 벡터 합에 대해 선형(linear)이다.**
- 이로 인해 변환 후에도 공간의 격자선(grid lines)은 **평행**하고 **균등 간격**을 유지한다.
- 공간은 늘어나거나(Stretch), 기울어질(Shear) 수 있지만, 원점은 변하지 않고, 공간이 휘거나(curvy) 왜곡되지는 않는다.

---

### 변환된 Basis Vector로 이해하기

- 원래 벡터 \( r \)은 단위벡터들의 선형 결합으로 표현할 수 있다:

$$
r = n e_1 + m e_2
$$

- Matrix \( A \)를 적용하면:

$$
A r = n (A e_1) + m (A e_2)
$$

- 여기서 \( A e_1, A e_2 \)는 각각 변환된 Basis Vector로 생각할 수 있다:

$$
A e_1 = e_1', \quad A e_2 = e_2'
$$

따라서:

$$
A r = n e_1' + m e_2'
$$

→ 즉, 변환된 Basis Vector들의 선형 결합으로 결과를 얻을 수 있다.

---

### 구체적인 예시

Matrix \( A \)는 이전 Apples and Bananas 문제에서 사용한 다음과 같은 행렬이다:

$$
A = \begin{bmatrix}
2 & 3 \\
10 & 1
\end{bmatrix}
$$

벡터 
$$
r = \begin{bmatrix}
3 \\
2
\end{bmatrix}
$$ 
에 대해 계산해보자.

**직접 계산하면:**

$$
A r =
\begin{aligned}
& 2 \times 3 + 3 \times 2 = 6 + 6 = 12 \\
& 10 \times 3 + 1 \times 2 = 30 + 2 = 32
\end{aligned}
$$

결과:

$$
\begin{bmatrix}
12 \\
32
\end{bmatrix}
$$

---

### Basis Vector를 통한 접근

벡터 \( r \)은 다음처럼 표현할 수 있다:

$$
r = 3 \times \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 2 \times \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

Matrix를 적용하면:

$$
A r = 3(A e_1) + 2(A e_2)
$$

각 변환을 계산해보자:

$$ A e_1 = \begin{bmatrix} 2 \\ 10 \end{bmatrix} $$
$$ A e_2 = \begin{bmatrix} 3 \\ 1 \end{bmatrix} $$

따라서:

$$
A r = 3 \times \begin{bmatrix} 2 \\ 10 \end{bmatrix} + 2 \times \begin{bmatrix} 3 \\ 1 \end{bmatrix}
$$

계산하면:

$$
= \begin{bmatrix} 6 \\ 30 \end{bmatrix} + \begin{bmatrix} 6 \\ 2 \end{bmatrix}
= \begin{bmatrix} 12 \\ 32 \end{bmatrix}
$$

직접 계산한 결과와 동일하다.

---

### 요약

- Matrix는 단순히 단위벡터들을 새로운 위치로 이동시킨다.
- 벡터 전체는 이동된 단위벡터들의 선형 결합으로 표현할 수 있다.
- Matrix 변환은 공간을 휘거나 비틀지 않고, 선형(linear) 변환만 일으킨다.
- 변환 후에도 벡터 합, 스칼라 곱 등 벡터 연산의 규칙은 그대로 유지된다.

## Types of Matrices Transformation

### 1. Identity Matrix
- 단위행렬(Identity Matrix)은 아무것도 변환하지 않는 행렬이다.
- 구성:

$$
I = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

  이 행렬을 벡터 
$$ \begin{bmatrix} x \\ y \end{bmatrix} $$
에 곱하면 결과는 변하지 않는다:

$$
I \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix}
$$

---

### 2. Scaling
- 대각선에 서로 다른 값이 있으면 축 방향으로 공간이 늘어나거나 줄어든다.

예:

$$
\text{Scaling Matrix} = \begin{bmatrix}
3 & 0 \\
0 & 2
\end{bmatrix}
$$

- x축은 3배, y축은 2배로 스케일됨
- 만약 스케일 값이 1보다 작으면, 해당 방향으로 공간이 압축(squish)된다.

---

### 3. Reflection (뒤집기)

#### x축 반전

$$
\begin{bmatrix}
-1 & 0 \\
0 & 1
\end{bmatrix}
$$

- x축을 기준으로 좌우 반전

#### y축 반전

$$
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

- y축을 기준으로 위아래 반전

#### 원점 대칭 (Inversion)

$$
\begin{bmatrix}
-1 & 0 \\
0 & -1
\end{bmatrix}
$$

- 모든 방향 반전 (x, y 둘 다 뒤집음)

#### 45도 대칭 (축 뒤바꾸기)

$$
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
$$

- x축과 y축을 서로 교환하는 효과 (45도 대칭)

또는:

$$
\begin{bmatrix}
0 & -1 \\
-1 & 0
\end{bmatrix}
$$

- 반대 방향으로 교환하는 45도 대칭

---

### 4. Shear (전단 변환)

- 한 방향만 변형시키는 변환

예를 들어, e1은 그대로 두고, e2를 옆으로 밀면:

$$
\text{Shear Matrix} = \begin{bmatrix}
1 & 0 \\
1 & 1
\end{bmatrix}
$$

- 이 변환은 원래 정사각형을 평행사변형으로 변형시킨다.

---

### 5. Rotation (회전)

- 공간을 회전시키는 변환
- 90도 회전 행렬 예:

$$
\text{90도 회전 Matrix} = \begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}
$$

- 일반적인 2D 회전 행렬은 각도 \( \theta \)에 대해 다음과 같다:

$$
\text{Rotation Matrix} =
\begin{bmatrix}
\cos \theta & \sin \theta \\
-\sin \theta & \cos \theta
\end{bmatrix}
$$

- theta가 양수일 때, 반시계 방향 회전
- 예를 들어, -90도 회전이면 다음과 같이 계산한다:

$$
\sin(-90^\circ) = -1
$$

---

### 6. 데이터 과학에서의 활용
- 얼굴 인식(facial recognition)처럼 데이터의 orientation(방향성)을 맞추기 위해  
Stretch, Mirror, Shear, Rotation 변환을 종종 사용한다.
- 예를 들어, 얼굴을 카메라 방향에 맞추기 위해 회전 변환을 적용할 수 있다.

---

### Summary

| 변환 종류 | 특징 |
|:---|:---|
| Identity | 변환 없음 |
| Scaling | 특정 축 방향으로 확대/축소 |
| Reflection | 좌우, 위아래 뒤집기 |
| Inversion | 원점 기준 전체 반전 |
| Shear | 한 방향으로 밀기(평행사변형) |
| Rotation | 특정 각도만큼 회전 |

Matrix는 공간을 변환하는 다양한 방법을 제공하며, 이들을 조합하여 복잡한 변환을 구성할 수 있다.


## Composition or Combination of Matrix Transformations

### 1. 변환(Transformation) 조합의 필요성
- 복잡한 형태 변환(예: 얼굴 이미지 변형)은 회전(Rotation), 전단(Shear), 반사(Reflection), 확대/축소(Scaling) 변환을 조합해서 만들 수 있다.
- 변환을 순서대로 적용하여 복합적인 효과를 낼 수 있음.

---

### 2. 변환 순서: \( A_1 \) 후 \( A_2 \)

벡터 \( r \)에 대해:

1. 먼저 변환 \( A_1 \)을 적용
2. 그 결과에 다시 변환 \( A_2 \)를 적용

결국, 전체 변환은 \( A_2 A_1 r \) 이 된다.

---

### 3. 구체적인 예시

#### Step 1: Basis Vectors

초기 Basis Vectors:

$$
e_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad e_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

---

#### Step 2: 첫 번째 변환 \( A_1 \) — 90도 반시계 방향 회전

회전 행렬:

$$
A_1 = \begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

변환 결과:

$$
\begin{aligned}
e_1' &= \begin{bmatrix} 0 \\ 1 \end{bmatrix} \\
e_2' &= \begin{bmatrix} -1 \\ 0 \end{bmatrix}
\end{aligned}
$$

---

#### Step 3: 두 번째 변환 \( A_2 \) — 수직 반사(Vertical Reflection)

반사 행렬:

$$
A_2 = \begin{bmatrix}
-1 & 0 \\
0 & 1
\end{bmatrix}
$$

변환 결과:

- \( e_1 \)은 \( (-1, 0) \)으로 반사
- \( e_2 \)는 변하지 않음

---

#### Step 4: \( A_2 \)를 \( A_1 \) 결과에 적용 (즉, \( A_2 A_1 \))

\( A_2 \)를 \( A_1 \)에 곱한다:

$$
A_2 A_1 = \begin{bmatrix}
-1 & 0 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

계산:

$$
\begin{aligned}
\text{First column} &: (-1) \times 0 + 0 \times 1 = 0,\quad 0 \times 0 + 1 \times 1 = 1 \\
\text{Second column} &: (-1) \times (-1) + 0 \times 0 = 1,\quad 0 \times (-1) + 1 \times 0 = 0
\end{aligned}
$$

결과:

$$
A_2 A_1 = \begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}
$$

---

### 4. 변환 순서가 바뀔 경우 (즉, \( A_1 A_2 \))

\( A_1 \)을 \( A_2 \)에 곱한다:

$$
A_1 A_2 = \begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
-1 & 0 \\
0 & 1
\end{bmatrix}
$$

계산:

$$
\begin{aligned}
\text{First column} &: 0 \times (-1) + (-1) \times 0 = 0,\quad 1 \times (-1) + 0 \times 0 = -1 \\
\text{Second column} &: 0 \times 0 + (-1) \times 1 = -1,\quad 1 \times 0 + 0 \times 1 = 0
\end{aligned}
$$

결과:

$$
A_1 A_2 = \begin{bmatrix}
0 & -1 \\
-1 & 0
\end{bmatrix}
$$

---

### 5. 중요한 결론

- **Matrix multiplication은 교환법칙(commutative)을 만족하지 않는다.**

$$
A_2 A_1 \neq A_1 A_2
$$

- 하지만, **결합법칙(associative)은 성립한다.**

$$
A_3 (A_2 A_1) = (A_3 A_2) A_1
$$
---

### Summary

| 성질 | 설명 |
|:---|:---|
| 선형성(Linearity) | 벡터 합과 스칼라 곱에 대해 선형 유지 |
| 결합법칙(Associativity) | 괄호 위치에 관계없이 순차 적용 가능 |
| 비가환성(Non-commutativity) | 적용 순서가 결과에 영향을 줌 |

Matrix를 조합하여 복합 변환을 만들 수 있으며, 특히 데이터 과학, 컴퓨터 비전(얼굴 인식 등)에서 변환 순서를 정확히 관리하는 것이 중요하다.



## Solving the Apples and Bananas Problem: Gaussian Elimination

### 1. 문제 설정

- 두 번 쇼핑을 함:
  - 2 apples + 3 bananas = 8 euros
  - 10 apples + 1 banana = 13 euros

이를 Matrix 형태로 표현하면:

$$
A = \begin{bmatrix} 2 & 3 \\ 10 & 1 \end{bmatrix}, \quad
r = \begin{bmatrix} a \\ b \end{bmatrix}, \quad
s = \begin{bmatrix} 8 \\ 13 \end{bmatrix}
$$

- 식은 다음과 같이 정리할 수 있음:

$$
A r = s
$$

---

### 2. 역행렬(Inverse)을 통한 풀이 아이디어

- 역행렬 \( A^{-1} \)이 존재하면:

$$
A^{-1} A = I
$$

- 양변에 \( A^{-1} \)을 곱하면:

$$
A^{-1} A r = A^{-1} s
$$

즉,

$$
r = A^{-1} s
$$

- **역행렬을 구할 수 있다면 문제를 일반적으로 해결할 수 있음.**

---

### 3. 굳이 역행렬을 구하지 않고도 풀 수 있는 방법: 소거법(Elimination)

#### 예시: 3가지 품목(apple, banana, carrot)

Matrix:

$$
A = \begin{bmatrix}
1 & 1 & 3 \\
1 & 2 & 4 \\
1 & 1 & 2
\end{bmatrix}
$$

Vector:

$$
s = \begin{bmatrix}
15 \\ 21 \\ 13
\end{bmatrix}
$$

**Row 1**: 1 apple + 1 banana + 3 carrots = 15  
**Row 2**: 1 apple + 2 bananas + 4 carrots = 21  
**Row 3**: 1 apple + 1 banana + 2 carrots = 13  

---

#### Step 1: Row Operation (Elimination)

- \( R_2 - R_1 \) :

$$
(1,2,4) - (1,1,3) = (0,1,1)
$$

- \( R_3 - R_1 \) :

$$
(1,1,2) - (1,1,3) = (0,0,-1)
$$

오른쪽 벡터도 동일하게 연산:

- \( 21 - 15 = 6 \)
- \( 13 - 15 = -2 \)

---

#### Step 2: 삼각형 형태로 정리 (Echelon Form)

변형된 행렬:

$$
\begin{bmatrix}
1 & 1 & 3 \\
0 & 1 & 1 \\
0 & 0 & -1
\end{bmatrix}
$$

오른쪽:

$$
\begin{bmatrix}
15 \\ 6 \\ -2
\end{bmatrix}
$$

---

#### Step 3: Back Substitution

- 세 번째 식에서 바로 알 수 있음:

$$
-1 \times c = -2 \quad \Rightarrow \quad c = 2
$$

- 두 번째 식에 \( c = 2 \)를 대입:

$$
1 \times b + 1 \times 2 = 6 \quad \Rightarrow \quad b = 4
$$

- 첫 번째 식에 \( b = 4, c = 2 \)를 대입:

$$
1 \times a + 1 \times 4 + 3 \times 2 = 15 \quad \Rightarrow \quad a = 5
$$

---

### 4. 최종 해

$$
a = 5, \quad b = 4, \quad c = 2
$$

- Apple: 5 euros
- Banana: 4 euros
- Carrot: 2 euros

---

### 5. 요약

| 용어 | 설명 |
|:---|:---|
| Elimination | 하나의 row를 다른 row에서 빼서 하위 요소를 0으로 만드는 과정 |
| Echelon Form | 주대각선 아래가 모두 0인 삼각형 형태의 행렬 |
| Back Substitution | 마지막 식부터 역방향으로 값을 대입하여 해를 구하는 과정 |
| 결과 | 원래 행렬을 Identity Matrix로 변환 |

변형 과정:

$$
\text{최종 행렬} =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

---

### 6. 중요한 포인트

- 역행렬 없이도 Gaussian Elimination을 통해 문제를 해결할 수 있다.
- 하지만, 역행렬을 구하면 임의의 \( s \)에 대해 \( r \)을 빠르게 구할 수 있다.
- **Gaussian Elimination**은 연산 수가 적고 매우 효율적인 방법이다.

---


## Going from Gaussian Elimination to Finding the Inverse Matrix

### 1. 문제 설정

- 3×3 행렬 \( A \)와 그 역행렬 \( B \)가 존재한다고 하자.

$$
A \times B = I
$$

- 예시로 사용하는 \( A \):

$$
A = \begin{bmatrix}
1 & 1 & 3 \\
1 & 2 & 4 \\
1 & 1 & 2
\end{bmatrix}
$$

- \( B \)는 다음과 같은 미지수로 구성된 행렬이다:

$$
B = \begin{bmatrix}
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23} \\
b_{31} & b_{32} & b_{33}
\end{bmatrix}
$$

여기서 \( b_{ij} \)는 \( i \)번째 행, \( j \)번째 열의 원소를 의미한다.

- Identity Matrix \( I \):

$$
I = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

---

### 2. 역행렬을 구하는 아이디어

- \( A \times B = I \) 이므로,  
  \( B \)는 \( A \)의 역행렬 \( A^{-1} \)이다.
- 실제로는 다음과 같이 각각의 열(column)을 따로따로 푸는 방식으로 접근할 수 있다.
  - 첫 번째 열을 풀 때는 \( A \times \) (B의 첫 번째 열) = (I의 첫 번째 열) 로 생각
  - 두 번째 열, 세 번째 열도 동일

하지만! 모든 열을 **한 번에 동시에** 다루면 훨씬 효율적이다.

---

### 3. Gaussian Elimination 전체 과정을 통해 역행렬 구하기

초기 augmented matrix (확장된 행렬):

$$
\left[ \begin{array}{ccc|ccc}
1 & 1 & 3 & 1 & 0 & 0 \\
1 & 2 & 4 & 0 & 1 & 0 \\
1 & 1 & 2 & 0 & 0 & 1
\end{array} \right]
$$

---

#### Step 1: 첫 번째 열을 기준으로 제거

- \( R_2 - R_1 \) :

$$
(1,2,4) - (1,1,3) = (0,1,1)
$$

- \( R_3 - R_1 \) :

$$
(1,1,2) - (1,1,3) = (0,0,-1)
$$

오른쪽 벡터도 동일하게 갱신:

- \( (0,1,0) - (1,0,0) = (-1,1,0) \)
- \( (0,0,1) - (1,0,0) = (-1,0,1) \)

변형된 행렬:

$$
\left[ \begin{array}{ccc|ccc}
1 & 1 & 3 & 1 & 0 & 0 \\
0 & 1 & 1 & -1 & 1 & 0 \\
0 & 0 & -1 & -1 & 0 & 1
\end{array} \right]
$$

---

#### Step 2: 대각 성분을 1로 맞추기

- 세 번째 행 \( R_3 \)을 \(-1\)로 나눔:

$$
R_3 \div (-1)
$$

변형 결과:

$$
\left[ \begin{array}{ccc|ccc}
1 & 1 & 3 & 1 & 0 & 0 \\
0 & 1 & 1 & -1 & 1 & 0 \\
0 & 0 & 1 & 1 & 0 & -1
\end{array} \right]
$$

---

#### Step 3: Back Substitution (역방향 대입)

- 세 번째 행을 이용해 두 번째 행에서 제거:

$$
R_2 \leftarrow R_2 - (1) \times R_3
$$

- 세 번째 행을 이용해 첫 번째 행에서 제거:

$$
R_1 \leftarrow R_1 - (3) \times R_3
$$

변형 결과:

$$
\left[ \begin{array}{ccc|ccc}
1 & 1 & 0 & -2 & 0 & 3 \\
0 & 1 & 0 & -2 & 1 & 1 \\
0 & 0 & 1 & 1 & 0 & -1
\end{array} \right]
$$

---

- 두 번째 행을 이용해 첫 번째 행에서 제거:

$$
R_1 \leftarrow R_1 - (1) \times R_2
$$

최종 결과:

$$
\left[ \begin{array}{ccc|ccc}
1 & 0 & 0 & 0 & -1 & 2 \\
0 & 1 & 0 & -2 & 1 & 1 \\
0 & 0 & 1 & 1 & 0 & -1
\end{array} \right]
$$

---

### 4. 최종 결과: 역행렬

오른쪽 부분이 바로 \( A^{-1} \):

$$
A^{-1} =
\begin{bmatrix}
0 & -1 & 2 \\
-2 & 1 & 1 \\
1 & 0 & -1
\end{bmatrix}
$$

---

### 5. 요약

| 용어 | 설명 |
|:---|:---|
| Elimination | 위 삼각 형태로 만들기 위해 row를 변환 |
| Back Substitution | 아래에서부터 값들을 차례로 구하는 과정 |
| 역행렬 | \( A^{-1} \)을 구하면, 어떤 \( s \)가 주어져도 \( r = A^{-1}s \)로 바로 해를 구할 수 있음 |

---

### 6. 중요한 포인트

- Gaussian Elimination 과정을 통해 역행렬을 구할 수 있다.
- 수학적으로는 
$$ A \times A^{-1} = I$$
가 항상 성립한다.
- \( A^{-1} \)을 알고 있으면, **모든** 우변 \( s \)에 대해 빠르게 해를 찾을 수 있다.
- 이 방법은 특히 고차원(수백×수백) 문제에서도 유용하다.

---








