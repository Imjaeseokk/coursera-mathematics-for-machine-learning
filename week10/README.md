# Mathematics for Machine Learning: Linear Algebra

Coursera - Imperial College London

Link to [Course](https://www.coursera.org/learn/linear-algebra-machine-learning)

## The Gram-Schmidt Process

### 목적
- 주어진 선형 독립 벡터 집합 \( \{ \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n \} \) 으로부터
- **직교 정규 기저 (orthonormal basis)** \( \{ \mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n \} \) 를 구성

---

### Step 1: 첫 번째 벡터 정규화

- 첫 번째 벡터는 단순히 단위 벡터로 정규화:
  $$
  \mathbf{e}_1 = \frac{\mathbf{v}_1}{\lVert \mathbf{v}_1 \rVert}
  $$

---

### Step 2: 두 번째 벡터 정사영 후 직교화

- \( \mathbf{v}_2 \)를 \( \mathbf{e}_1 \)에 정사영 후 제거:
  $$
  \mathbf{u}_2 = \mathbf{v}_2 - (\mathbf{v}_2 \cdot \mathbf{e}_1) \mathbf{e}_1
  $$
- 직교한 벡터를 정규화:
  $$
  \mathbf{e}_2 = \frac{\mathbf{u}_2}{\lVert \mathbf{u}_2 \rVert}
  $$

---

### Step 3: 세 번째 벡터 정사영 후 직교화

- \( \mathbf{v}_3 \)에서 \( \mathbf{e}_1 \), \( \mathbf{e}_2 \) 성분 제거:
  $$
  \mathbf{u}_3 = \mathbf{v}_3 - (\mathbf{v}_3 \cdot \mathbf{e}_1) \mathbf{e}_1 - (\mathbf{v}_3 \cdot \mathbf{e}_2) \mathbf{e}_2
  $$
- 정규화:
  $$
  \mathbf{e}_3 = \frac{\mathbf{u}_3}{\lVert \mathbf{u}_3 \rVert}
  $$

---

### 일반화된 Gram-Schmidt 공식

- \( k \)-번째 벡터에 대해:

  1. 기존 기저 벡터에 대한 정사영 제거:
     $$
     \mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} (\mathbf{v}_k \cdot \mathbf{e}_j) \mathbf{e}_j
     $$
  2. 정규화:
     $$
     \mathbf{e}_k = \frac{\mathbf{u}_k}{\lVert \mathbf{u}_k \rVert}
     $$

---

### 요약

- **입력**: 선형 독립 벡터 집합 
\( \{ \mathbf{v}_1, \dots, \mathbf{v}_n \} \)
- **출력**: 직교 정규 벡터 집합 
\( \{ \mathbf{e}_1, \dots, \mathbf{e}_n \} \)
- 각 단계에서:
  - 기존 벡터들에 직교한 성분만 남기고
  - 단위 벡터로 정규화
- 결과: 계산이 편리한 **정규 직교 기저**, 변환 행렬을 간단히 역행렬 또는 전치 행렬로 다룰 수 있음

---

### 활용

- 투영 계산 시 단순화
- 역행렬 계산 시 효율적
- 선형 변환 및 회전 변환 등에서 유리

## Reflecting in a Plane

### 목표
- 주어진 벡터 \( \mathbf{r} \)을 어떤 **기울어진 평면** 기준으로 반사(reflection)하는 행렬 변환을 수행

---

### 주어진 벡터

- 평면 위에 있는 두 벡터:
  $$
  \mathbf{v}_1 = (1, 1, 1), \quad \mathbf{v}_2 = (2, 0, 1)
  $$
- 평면 밖의 벡터:
  $$
  \mathbf{v}_3 = (3, 1, -1)
  $$

---

### 1. Gram-Schmidt를 통한 직교 기저 구성

#### Step 1: \( \mathbf{e}_1 \)

- \( \mathbf{v}_1 \)을 정규화:
  $$
  \mathbf{e}_1 = \frac{1}{\sqrt{3}} (1, 1, 1)
  $$

#### Step 2: \( \mathbf{e}_2 \)

- \( \mathbf{v}_2 \)을 \( \mathbf{e}_1 \)에 정사영 후 제거:
  $$
  \mathbf{u}_2 = \mathbf{v}_2 - (\mathbf{v}_2 \cdot \mathbf{e}_1) \mathbf{e}_1 = (2, 0, 1) - (1,1,1) = (1, -1, 0)
  $$
- 정규화:
  $$
  \mathbf{e}_2 = \frac{1}{\sqrt{2}} (1, -1, 0)
  $$

#### Step 3: \( \mathbf{e}_3 \)

- \( \mathbf{v}_3 \)에서 \( \mathbf{e}_1 \), \( \mathbf{e}_2 \) 성분 제거:
  $$
  \begin{aligned}
  \mathbf{u}_3 &= \mathbf{v}_3 - (\mathbf{v}_3 \cdot \mathbf{e}_1) \mathbf{e}_1 - (\mathbf{v}_3 \cdot \mathbf{e}_2) \mathbf{e}_2 \\
  &= (3, 1, -1) - (1,1,1) - (1, -1, 0) = (1, 1, -2)
  \end{aligned}
  $$
- 정규화:
  $$
  \mathbf{e}_3 = \frac{1}{\sqrt{6}} (1, 1, -2)
  $$

---

### 2. 기저 행렬 \( E \) 정의

- 직교 정규 기저 벡터들을 열로 정리한 행렬:
  $$
  E = \left[ \mathbf{e}_1 \ \mathbf{e}_2 \ \mathbf{e}_3 \right] = 
  \begin{bmatrix}
  \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
  \frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
  \frac{1}{\sqrt{3}} & 0 & -\frac{2}{\sqrt{6}}
  \end{bmatrix}
  $$

---

### 3. 평면에 대한 Reflection Transformation

- 평면에 수직인 \( \mathbf{e}_3 \) 방향만 부호 반전
- E-basis에서의 변환 행렬:
  $$
  T_E = \begin{bmatrix}
  1 & 0 & 0 \\
  0 & 1 & 0 \\
  0 & 0 & -1
  \end{bmatrix}
  $$

---

### 4. 전체 변환

- 전체 변환 식:
  $$
  \mathbf{r}' = E \cdot T_E \cdot E^{-1} \cdot \mathbf{r}
  $$
- \( E \)는 직교 행렬이므로:
  $$
  E^{-1} = E^\top
  $$

- 따라서:
  $$
  \mathbf{r}' = E \cdot T_E \cdot E^\top \cdot \mathbf{r}
  $$

---

### 5. 예제

- \( \mathbf{r} = (2, 3, 5) \)에 대해 계산하면:
  $$
  \mathbf{r}' = \frac{1}{3}(11, 14, 5)
  $$

---

### 요약

- 복잡한 평면 반사 문제를 간단한 정사영 및 기저 변환으로 해결
- 핵심 전략:
  - 기저 변환을 통해 평면 좌표계로 이동
  - 평면에서 간단한 reflection 수행
  - 다시 원래 좌표계로 복원
- 수식:
  $$
  \mathbf{r}' = E \cdot T_E \cdot E^\top \cdot \mathbf{r}
  $$

## What are Eigenvalues and Eigenvectors?

### 1. 개념 소개
- 'Eigen'은 독일어로 **고유한(characteristic)** 이라는 의미를 지님.
- **Eigenproblem**이란 어떤 선형 변환에서 **고유한 성질을 가진 벡터와 값**을 찾는 문제.

---

### 2. 직관적 이해: 선형 변환과 도형의 왜곡

- 선형 변환(linear transformation)은 scaling, rotation, shear 등을 포함.
- 전체 공간의 벡터에 변환을 적용하면, 원점을 중심으로 한 정사각형이 어떻게 변형되는지 관찰 가능:
  - 수직 방향으로 2배 scaling ⇒ 직사각형으로 왜곡
  - 수평 shear ⇒ 사다리꼴로 왜곡

---

### 3. 고유벡터(Eigenvectors)

- 어떤 선형 변환을 적용한 후에도 **방향이 변하지 않는 벡터**들을 **고유벡터**라고 함.
- 예: 수직 방향 scaling 시,
  - **수직 벡터**: 방향 유지 + 길이 2배 증가 → eigenvector, eigenvalue = 2
  - **수평 벡터**: 방향 및 길이 유지 → eigenvector, eigenvalue = 1
  - **대각선 벡터**: 방향과 길이 모두 변경 → eigenvector 아님

#### 핵심 정의

- **고유벡터** \( \mathbf{v} \)와 **고유값** \( \lambda \)는 다음을 만족:
  $$
  A \mathbf{v} = \lambda \mathbf{v}
  $$

---

### 4. 예시들

#### 1) 수직 스케일링

- \( \mathbf{v}_x = (1, 0) \), \( \mathbf{v}_y = (0, 1) \)
- 변환: 수직으로 2배 scaling
  - \( \mathbf{v}_x \): 방향 및 크기 불변 ⇒ 고유값 = 1
  - \( \mathbf{v}_y \): 방향 유지, 크기 2배 ⇒ 고유값 = 2

#### 2) 순수 Shear (면적 보존)

- 수평 벡터만 방향 유지 ⇒ 고유벡터는 수평 방향 하나뿐

#### 3) Rotation

- 회전은 모든 벡터의 방향을 바꾸므로 **고유벡터가 존재하지 않음**

---

### 5. 고차원 일반화

- 3차원 이상에서도 개념 동일:
  - 변환 후에도 **방향을 유지하는 벡터**를 찾음
  - 이들의 **길이 변화 비율**이 고유값

---

### 6. 요약

- 고유벡터는 **선형 변환 후에도 방향이 유지되는 벡터**
- 고유값은 해당 고유벡터의 **스케일 변화 비율**
- 시각적으로 이해하면 개념이 명확해지며, 수학적 표현은 단순한 벡터-스칼라 곱
- 예외적으로 **회전 변환**에는 고유벡터가 존재하지 않음
- 이 개념은 PCA 등 고차원 분석에서 매우 중요함

```math
A \mathbf{v} = \lambda \mathbf{v}
```
- 여기서,

\( A \): 선형 변환 행렬  
\( \mathbf{v} \): 고유벡터  
\( \lambda \): 고유값


## Special Eigen-Cases

### 개요
이 장에서는 선형 변환에서 **특별한 경우의 고유벡터(eigenvector)** 와 **고윳값(eigenvalue)** 들이 어떤 모습을 보이는지를 시각적으로 이해합니다. 고유벡터란, 변환 전후에도 같은 span(방향)에 남는 벡터이며, 고윳값은 해당 벡터가 얼마나 늘어났거나 줄어들었는지를 나타냅니다.

---

### 1. **Uniform Scaling (균일한 스케일링)**

- 모든 방향으로 같은 비율로 스케일링하는 경우
- 모든 벡터가 고유벡터가 됨
- 고윳값은 스케일링 비율

<img src="https://latex.codecogs.com/svg.image?\text{If&space;uniform&space;scaling:}&space;A&space;=&space;sI&space;\Rightarrow&space;\text{All&space;vectors&space;are&space;eigenvectors},&space;\lambda=s" />

---

### 2. **180° Rotation (180도 회전)**

- 일반적인 회전(예: 90도)은 고유벡터가 없음
- 그러나 **180도 회전은 모든 벡터가 고유벡터**가 됨
- 방향이 반대가 되므로 고윳값은 -1

<img src="https://latex.codecogs.com/svg.image?\text{For&space;180}^\circ\text{&space;rotation:}&space;\lambda=-1" />

---

### 3. **Horizontal Shear + Vertical Scaling**

- 수평 shear와 수직 scaling이 조합된 경우
- 육안으로 고유벡터를 찾기 어렵지만 존재함
- 일부 벡터(예: 수평 벡터)는 변환 전후 동일 방향을 유지
- 이런 벡터는 고유벡터이고, 대응하는 고윳값은 1

> 시각적으로는 변형된 평행사변형(parallelogram)에서 **변형 전후 방향이 바뀌지 않는 벡터**를 찾는 것으로 이해

---

### 4. **3D Rotation (3차원 회전)**

- 3D 회전에서도 2D와 동일하게 대부분의 벡터는 방향이 바뀜
- 그러나 회전의 **축 방향 벡터는 회전 전후 위치가 변하지 않음**
- 따라서 그 축이 **고유벡터**이며, 고윳값은 1

<img src="https://latex.codecogs.com/svg.image?\text{In&space;3D&space;rotation:}&space;\text{Eigenvector}&space;=&space;\text{Rotation&space;axis},&space;\lambda=1" />

---

### 요약

- 고유벡터는 변환 전후 동일한 방향을 유지하는 벡터
- 고윳값은 그 벡터가 얼마나 스케일링되는지를 나타냄
- 특별한 변환에서는 전 벡터가 고유벡터가 되기도 하며
- 3D 회전의 고유벡터는 회전축과 일치
- 머신러닝에서는 수백 차원의 공간에서 이러한 고유문제를 다루게 되며, 수학적으로 더 일반적인 정의가 필요함

## Calculating Eigenvectors

### 1. 고유문제의 수식화

선형변환 행렬 $$ A $$에 대해, 고유벡터 $$ \mathbf{x} $$와 고윳값 $$ \lambda $$는 다음 조건을 만족한다:

$$
A\mathbf{x} = \lambda \mathbf{x}
$$

- 이 식은 변환 $$ A $$가 어떤 벡터 $$ \mathbf{x} $$를 **방향은 유지한 채로** 크기만 $$ \lambda $$배 만큼 스케일링한다는 의미.
- 모든 성분이 0인 벡터(즉, $$ \mathbf{x} = \mathbf{0} $$)는 의미 없는 trivial 해이므로 제외함.

위 식을 정리하면,

$$
(A - \lambda I)\mathbf{x} = \mathbf{0}
$$

- 여기서 $$ I $$는 항등 행렬(identity matrix)이며, $$ \lambda $$는 스칼라(수), $$ A $$는 정사각 행렬이다.

이제 **고유값을 구하는 핵심 조건**은 다음과 같다:

$$
\det(A - \lambda I) = 0
$$

이 식을 **특성 방정식(characteristic equation)**이라고 하며, 여기서 나온 $$ \lambda $$가 고유값이 된다.

---

### 2. 예제: 수직 스케일링 변환

변환 행렬:

$$
A = \begin{bmatrix}
1 & 0 \\
0 & 2
\end{bmatrix}
$$

특성 방정식:

$$
\det\left(
\begin{bmatrix}
1 - \lambda & 0 \\
0 & 2 - \lambda
\end{bmatrix}
\right) = (1 - \lambda)(2 - \lambda) = 0
$$

⇒ 고유값: $$ \lambda_1 = 1, \quad \lambda_2 = 2 $$

고유벡터:

- $$ \lambda = 1 $$
인 경우:
  
  $$ (A - I)\mathbf{x} = 0 \Rightarrow \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \mathbf{0} $$

  ⇒ 
$$ x_2 = 0, x_1 은  자유롭게  선택 가능 $$

  ⇒ $$ \mathbf{x} = \begin{bmatrix} t \\ 0 \end{bmatrix} $$ (수평 방향의 모든 벡터)

$$ \lambda = 2 $$
인 경우:

  $$ (A - 2I)\mathbf{x} = 0 \Rightarrow \begin{bmatrix} -1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \mathbf{0} 
  $$

  ⇒ $$ x_1 = 0, x_2 $$은 자유롭게 선택 가능

  ⇒ $$ \mathbf{x} = \begin{bmatrix} 0 \\ t \end{bmatrix} $$ (수직 방향의 모든 벡터)

---

### 3. 예제: 90도 회전

변환 행렬:

$$
A = \begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

특성 방정식:

$$
\det\left( \begin{bmatrix}
-\lambda & -1 \\
1 & -\lambda
\end{bmatrix} \right) = \lambda^2 + 1 = 0
$$

- 실수 범위에서는 해가 없음
- ⇒ **실수 고유벡터가 존재하지 않음**

(복소수 고유벡터는 존재하지만 여기서는 다루지 않음)

---

### 4. 고유값 계산의 현실적 의미

- 차원이 커질수록 (예: 100차원) 특성 다항식의 해를 구하는 것은 **대수적으로 불가능**해짐
- 실제로는 대부분 **수치적 반복 알고리즘(예: 파워 iteration, QR decomposition 등)**을 사용함
- 따라서 **핵심은 계산보다 직관과 해석**임

---

### 요약

- 고유벡터는 변환 전후에 방향이 변하지 않는 벡터
- 고윳값은 해당 벡터가 얼마나 스케일링되는지를 나타냄
- 일반적으로 다음 조건을 만족시켜 계산함:

  $$
  \det(A - \lambda I) = 0
  $$

- 고유값을 구한 후, 다음을 만족하는 벡터 $$ \mathbf{x} $$를 구함:

  $$
  (A - \lambda I)\mathbf{x} = \mathbf{0}
  $$

- 복잡한 계산은 컴퓨터에게 맡기고, **해석과 개념에 집중하는 것이 더 중요**함

다음은 고유벡터를 **기저(basis)**로 사용하면 어떤 일이 일어나는지 살펴본다.

## Changing to the Eigenbasis

선형 변환을 여러 번 반복 적용해야 하는 경우, 계산 효율성을 높이기 위해 고유기저(eigenbasis)를 이용한 **대각화(diagonalisation)** 기법을 사용할 수 있다. 이 챕터에서는 고유값과 고유벡터 개념을 활용하여 변환 행렬을 효율적으로 거듭제곱하는 방법을 설명한다.

### 행렬 거듭제곱의 문제점

변환 행렬을 반복하여 곱해야 하는 경우가 종종 있다. 예를 들어, 변환 행렬을 T라 할 때, 이 변환이 한 번 적용될 때마다 입자의 위치가 변한다고 하자. 초기 벡터 v₀에서 출발한 입자의 위치는 다음과 같이 나타난다.

$$
v_1 = T v_0,\quad v_2 = T^2 v_0,\quad \dots,\quad v_n = T^n v_0
$$

이때 n이 매우 크다면, 일반적인 행렬의 거듭제곱 계산은 계산량이 상당히 크고 비효율적이다.

### 대각행렬의 효율성

만약 행렬이 대각행렬이라면, 거듭제곱 계산은 훨씬 간단해진다. 예를 들어 대각행렬 D를 n제곱할 때는 각 대각 성분만 n제곱하면 된다.

$$
D^n = 
\begin{bmatrix}
a^n & 0 & 0 \\
0 & b^n & 0 \\
0 & 0 & c^n
\end{bmatrix}
$$

### 고유기저로의 변환과 대각화

모든 행렬이 대각행렬 형태는 아니지만, 특정한 고유기저로 좌표계를 바꾸면 변환 행렬을 대각행렬 형태로 나타낼 수 있다. 이 기저를 바로 **고유기저(eigenbasis)** 라고 한다. 고유벡터들을 열로 하는 변환 행렬 C를 구성하면, 원래의 변환 행렬 T는 다음과 같이 표현될 수 있다.

$$
T = C D C^{-1}
$$

여기서 D는 고유값들로 이루어진 대각행렬이다.

$$
D = 
\begin{bmatrix}
\lambda_1 & 0 & 0 \\
0 & \lambda_2 & 0 \\
0 & 0 & \lambda_3
\end{bmatrix}
$$

이러한 대각화를 통해 행렬의 거듭제곱을 효율적으로 계산할 수 있다.

### 행렬 거듭제곱의 일반적인 공식

대각화를 이용하면 행렬 T의 n제곱은 다음과 같이 간단히 표현할 수 있다.

$$
T^n = C D^n C^{-1}
$$

즉, 고유기저로 변환하여 거듭제곱을 계산한 뒤 다시 원래 기저로 돌아오는 과정을 통해 계산 효율성을 크게 높일 수 있다.

### 요약 및 의미

고유기저로의 변환은 복잡한 행렬 계산을 간단하게 만들어주는 중요한 기법이다. 고유벡터와 고유값을 통해 행렬을 대각화하여 계산량을 절감할 수 있으며, 이는 특히 머신러닝이나 데이터 분석에서 큰 차원의 행렬을 다룰 때 매우 유용하다.

## Introduction to PageRank

### PageRank 개요

PageRank는 1998년 구글의 창업자 Larry Page가 발표한 알고리즘으로, 웹사이트의 중요도를 평가하여 검색 결과의 순서를 결정하는 데 사용되었다. 이 알고리즘은 웹페이지 간의 링크 구조에 따라 각 페이지의 상대적 중요도를 결정하며, 이를 선형대수의 고유벡터(eigenvector) 개념과 연결하여 해석할 수 있다.

### 웹페이지 링크 구조 표현하기

PageRank 알고리즘의 핵심 아이디어는 임의의 인터넷 사용자가 웹페이지 링크를 무작위로 클릭할 때, 각 페이지에서 머무를 확률을 계산하는 것이다. 예를 들어 4개의 웹페이지 A, B, C, D가 있는 경우, 각 페이지에서 다른 페이지로의 링크를 확률 벡터로 나타낼 수 있다.

예를 들어, 페이지 A에서 다른 페이지로의 링크 확률 벡터는 다음과 같다.

$$
L_A = \left(0,\ \frac{1}{3},\ \frac{1}{3},\ \frac{1}{3}\right)
$$

이 벡터는 페이지 A가 자신을 제외한 B, C, D로 링크가 있고, 각 링크의 클릭 확률이 동일하도록 정규화된 것을 나타낸다.

다른 페이지들 역시 같은 방식으로 벡터를 구성한다.

### 링크 행렬 구축하기

링크 벡터들을 열(column)로 삼아 행렬 L을 구성할 수 있다. 예를 들어, 4개의 페이지가 있는 경우 다음과 같이 4×4 링크 행렬을 구성한다.

$$
L = \begin{bmatrix}
0 & 0 & 0 & 0 \\
\frac{1}{3} & 0 & \frac{1}{2} & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & 0 & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & \frac{1}{2} & 0
\end{bmatrix}
$$

이 행렬에서 각 열은 특정 페이지에서의 외부 링크 확률을 나타내며, 각 행은 특정 페이지로 들어오는 내부 링크를 나타낸다.

### PageRank 방정식

페이지의 중요도를 나타내는 랭크(rank) 벡터 r는 다음과 같이 표현할 수 있다.

$$
r = L r
$$

즉, 랭크 벡터 r는 링크 행렬 L과 랭크 벡터 r을 곱한 결과와 같아질 때까지 반복 계산하여 얻을 수 있다. 이는 결국 L의 고유값(eigenvalue)이 1인 고유벡터를 찾는 문제로 환원된다.

### 반복법을 통한 해 구하기 (Power method)

PageRank 문제는 처음에는 모든 페이지의 랭크가 동일하다고 가정하고, 이를 반복적으로 계산하여 점차 랭크 벡터를 갱신하는 방식을 사용한다. 이 반복법을 **Power method**라 하며, 수학적으로는 다음과 같은 식으로 표현된다.

$$
r_{i+1} = L r_i
$$

위 과정을 반복하면 r 벡터는 최종적으로 수렴하며, 이를 통해 각 웹페이지의 상대적 중요도를 얻을 수 있다.

### Damping factor의 역할

PageRank 알고리즘에는 **Damping factor (감쇠 인자)**라는 개념이 있다. 이는 사용자가 링크를 따라가지 않고 임의로 다른 웹페이지 주소를 입력할 확률을 반영한 것으로, 다음과 같은 수식으로 표현된다.

$$
r_{i+1} = d L r_i + \frac{1 - d}{n}
$$

여기서 \( d \) 는 0과 1 사이의 값이며, 보통 0.85가 사용된다. \( n \) 은 총 웹페이지의 개수이다. Damping factor는 계산의 안정성과 수렴 속도를 조절하는 역할을 한다.

### 실제 활용과 의의

실제 웹페이지는 매우 많은 페이지가 존재하며, 대부분의 페이지가 서로 링크되지 않는 **희소 행렬 (sparse matrix)** 형태를 띤다. 따라서 실제 PageRank 알고리즘은 이러한 희소 행렬을 다루는 데 최적화된 방법으로 효율적으로 계산한다.

결론적으로, PageRank 알고리즘은 고유벡터와 선형대수 이론을 활용하여 복잡한 웹페이지 구조 내에서 각 페이지의 상대적 중요도를 효율적으로 평가할 수 있는 강력한 도구이다.





