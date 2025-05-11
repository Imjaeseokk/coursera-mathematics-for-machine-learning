# Mathematics for Machine Learning: Linear Algebra

Coursera - Imperial College London

Link to [Course](https://www.coursera.org/learn/linear-algebra-machine-learning)

## Special matrices and coding up some matrix operations : Determinants and inverses

### 1. 행렬이 공간에 주는 영향: Determinant의 정의

- 행렬이 공간을 얼마나 확장 또는 축소시키는지를 나타내는 값이 바로 **행렬식(determinant)**이다.
- 예를 들어 다음과 같은 행렬을 보자:

$$
A = \begin{bmatrix}
a & 0 \\
0 & d
\end{bmatrix}
$$

- 이 행렬은 \( e_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \), \( e_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \) 를 각각 다음과 같이 변환한다:

$$
Ae_1 = \begin{bmatrix} a \\ 0 \end{bmatrix}, \quad Ae_2 = \begin{bmatrix} 0 \\ d \end{bmatrix}
$$

- 결과적으로 단위 정사각형은 면적이 \( a \times d \)인 직사각형으로 변환된다.  
  이때의 면적 비율이 **행렬식(determinant)**이며:

$$
\det(A) = ad
$$

---

### 2. 일반적인 2×2 행렬의 Determinant

- 보다 일반적인 행렬 \( A \):

$$
A = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

- 이 행렬이 단위 정사각형을 어떤 평행사변형으로 변환한다고 했을 때, 그 평행사변형의 면적은 다음과 같다:

$$
\det(A) = ad - bc
$$

- 이 determinant는 변환된 공간의 면적을 의미한다.

---

### 3. 2×2 행렬의 역행렬과 determinant의 관계

- 행렬 \( A \)의 역행렬은 다음과 같이 주어진다:

$$
A^{-1} = \frac{1}{\det(A)} \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

- 이 공식은 고등학교에서 배운 역행렬 정의로, 여전히 수학적으로 정확하다.
- 단, 역행렬이 존재하려면 다음 조건이 반드시 만족해야 한다:

$$
\det(A) \neq 0
$$

- 이 말은 즉, 변환이 공간을 어떤 **선형 독립된 형태**로 유지해야 함을 의미한다.

---

### 4. 선형 독립성과 determinant = 0의 관계

- 만약 \( e_1 \)과 \( e_2 \)가 같은 방향을 가리키거나, 하나가 다른 하나의 배수라면 이들은 선형 독립이 아니다.
- 예: \( e_1 \rightarrow \begin{bmatrix} 1 \\ 1 \end{bmatrix} \), \( e_2 \rightarrow \begin{bmatrix} 2 \\ 2 \end{bmatrix} \)

⇒ 변환 결과가 같은 선 위의 점들이 되며, 이 경우 평행사변형이 아닌 선분이 되어 **면적 = 0**, 즉:

$$
\det(A) = 0
$$

- 따라서 이 경우에는 **역행렬이 존재하지 않는다.**

---

### 5. 3차원에서도 마찬가지

- 3×3 행렬에서도 하나의 기저 벡터가 나머지 둘의 선형 결합이라면:

  - 변환된 결과는 평면(면적)이나 선(길이)로 **부피(volume) = 0**
  - 즉:

$$
\det(A) = 0 \Rightarrow A^{-1} \text{ does not exist}
$$

---

### 6. 가우스 소거법으로 본 선형 종속

예: 다음과 같은 행렬이 있다고 하자

$$
A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
5 & 7 & 9
\end{bmatrix}
$$

- 여기서 3번째 행은 첫 번째와 두 번째 행을 더한 것과 같음:

$$
R_3 = R_1 + R_2
$$

- 이는 선형 종속 관계를 의미하며, 결국 다음과 같은 결과를 낳음:

$$
\text{마지막 행} \Rightarrow 0 = 0
$$

- 이 경우 해가 무수히 많거나 없을 수 있으며, **유일해가 존재하지 않는다.**

---

### 7. 역행렬이 존재하지 않는 이유

- 선형 종속이 발생한 경우:

  - 변환이 공간 차원을 축소시킴 (예: 3D → 2D)
  - **정보의 일부가 사라졌기 때문에 되돌릴 수 없다**

⇒ 역행렬이 존재하지 않음

---

### 8. 요약

- **Determinant**는 행렬이 공간을 얼마나 확장 또는 축소시키는지를 나타내며, 면적 또는 부피 개념과 연결된다.
- \( \det(A) \neq 0 \): 역행렬이 존재, 해가 유일함
- \( \det(A) = 0 \): 역행렬 존재하지 않음, 해가 없거나 무한히 많음
- 선형 독립성은 determinant와 직결되며, 선형 종속이면 공간 축소 발생

---

### ✅ 핵심 공식 요약

- 2×2 행렬의 determinant:

$$
\det \begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
$$

- 역행렬 존재 조건:

$$
\text{If } \det(A) \neq 0, \text{ then } A^{-1} = \frac{1}{\det(A)} \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

$$det(A) = 0 \Rightarrow $$
역행렬 존재하지 않음

---

## Matrices as objects that map one vector another; all the types of matrices

### 1. 아인슈타인 합 기호 (Einstein Summation Convention)

- 행렬 연산을 간결하게 표현하는 기호로, **반복 인덱스가 있으면 자동으로 덧셈을 수행**한다고 가정한다.
- 일반적인 행렬 곱 \( C = AB \)에서:

$$
C_{ik} = \sum_j A_{ij} B_{jk}
$$

- 아인슈타인 합 기호를 사용하면, 이를 다음과 같이 간단히 쓴다:

$$
C_{ik} = A_{ij} B_{jk}
$$

- 위 표현에서 **반복되는 인덱스 \( j \)** 는 자동으로 **모든 \( j \)**에 대해 합산된다.

---

### 2. 행렬 곱 계산 예시

- 예를 들어 \( C_{23} \) (2행 3열의 원소)를 구할 때:

$$
C_{23} = A_{21} B_{13} + A_{22} B_{23} + \cdots + A_{2n} B_{n3}
$$

- 구현 관점에서는 3중 반복문으로 계산 가능:
  - 외부 루프: \( i \)
  - 중간 루프: \( k \)
  - 내부 루프 (누적합): \( j \)

---

### 3. 비정방 행렬 간의 곱

- \( A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p} \) 이라면:

$$
C = AB \in \mathbb{R}^{m \times p}
$$

- \( j \) 인덱스가 일치하면 곱셈 가능:

$$
C_{ik} = \sum_j A_{ij} B_{jk}
$$

- 예시:
  - \( A \): 2×3 행렬
  - \( B \): 3×4 행렬
  - 결과 \( C \): 2×4 행렬

---

### 4. Dot Product와 행렬 곱의 연결

- 벡터 \( \mathbf{u}, \mathbf{v} \in \mathbb{R}^n \) 에 대해 dot product:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i = u_i v_i
$$

- 벡터 \( \mathbf{u} \)를 행 벡터로, \( \mathbf{v} \)를 열 벡터로 보면:

$$
\mathbf{u}^\top \mathbf{v} = \text{dot product}
$$

⇒ **dot product는 행렬 곱의 특별한 경우**이다.

---

### 5. Dot Product의 기하학적 의미: 투영

- \( \hat{\mathbf{u}} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} \),  
  기준축: \( \hat{e}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \hat{e}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \)

- \( \hat{\mathbf{u}} \cdot \hat{e}_1 \):  
  → \( \hat{\mathbf{u}} \)를 x축으로 투영한 성분 = \( u_1 \)

- 반대로 \( \hat{e}_1 \cdot \hat{\mathbf{u}} \)도 동일한 값을 가진다.
  - 이는 dot product가 **교환 가능(commutative)** 하다는 사실을 의미

⇒ 투영된 길이와 dot product가 서로 대응하며, dot product는 대칭적인 기하 구조를 설명함

---

### 6. 요약

- 아인슈타인 합 기호는 반복 인덱스에 대해 자동으로 합산하는 간결한 표기법이다.
- 비정방 행렬도 \( j \) 인덱스가 일치하면 곱할 수 있으며, 결과 행렬은 \( \text{행렬 A의 행 수} \times \text{행렬 B의 열 수} \)
- **Dot product는 행렬 곱의 특수한 형태**이며, 기하적으로는 투영 의미를 갖는다.
- 행렬 연산은 수치적 작업일 뿐 아니라, 기하적 의미(공간 변환, 투영 등)를 내포한다.

---


## Matrices transform into the new basis vector set: Matrices changing basis

### 1. 기준 벡터와 변환 행렬

- **기존 기준**: \( \hat{e}_1 = [1, 0], \hat{e}_2 = [0, 1] \)
- **곰(Panda)의 기준 벡터 (우리 기준에서)**:
  - \( \mathbf{b}_1 = [3, 1] \)
  - \( \mathbf{b}_2 = [1, 1] \)

⇒ **곰의 기준을 나타내는 행렬** \( B \):

$$
B = \begin{bmatrix}
3 & 1 \\
1 & 1
\end{bmatrix}
$$

---

### 2. 곰의 좌표계에서의 벡터 → 나의 좌표계로 변환

- 곰 기준 벡터: \( \mathbf{v}_{\text{Bear}} = \begin{bmatrix} \frac{3}{2} \\ \frac{1}{2} \end{bmatrix} \)

- 내 기준에서의 벡터:

$$
\mathbf{v}_{\text{Me}} = B \cdot \mathbf{v}_{\text{Bear}} = \begin{bmatrix}
3 & 1 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
\frac{3}{2} \\
\frac{1}{2}
\end{bmatrix} = \begin{bmatrix}
5 \\
2
\end{bmatrix}
$$

---

### 3. 나의 좌표계 → 곰의 좌표계

- **역변환 행렬** \( B^{-1} \):

$$
B^{-1} = \frac{1}{2}
\begin{bmatrix}
1 & -1 \\
-1 & 3
\end{bmatrix}
$$

- 내 기준 벡터: \( \mathbf{v}_{\text{Me}} = \begin{bmatrix} 5 \\ 2 \end{bmatrix} \)

- 곰 기준 벡터로 변환:

$$
\mathbf{v}_{\text{Bear}} = B^{-1} \cdot \mathbf{v}_{\text{Me}} = \frac{1}{2}
\begin{bmatrix}
1 & -1 \\
-1 & 3
\end{bmatrix}
\begin{bmatrix}
5 \\
2
\end{bmatrix}
= \begin{bmatrix}
\frac{3}{2} \\
\frac{1}{2}
\end{bmatrix}
$$

---

### 4. 직교정규(Orthonormal) 기준의 예시

- 곰의 기준 벡터:

$$
\mathbf{b}_1 = \frac{1}{\sqrt{2}} [1, 1], \quad
\mathbf{b}_2 = \frac{1}{\sqrt{2}} [-1, 1]
$$

⇒ **변환 행렬** \( B \):

$$
B = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & -1 \\
1 & 1
\end{bmatrix}
$$

- 곰 기준 벡터: \( \mathbf{v}_{\text{Bear}} = \begin{bmatrix} 2 \\ 1 \end{bmatrix} \)

- 내 기준 벡터로:

$$
\mathbf{v}_{\text{Me}} = B \cdot \mathbf{v}_{\text{Bear}} = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & -1 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
2 \\
1
\end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix}
1 \\
3
\end{bmatrix}
$$

---

### 5. 역변환 (내 기준 → 곰 기준)

- \( B^{-1} = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & 1 \\
-1 & 1
\end{bmatrix} \)

- 내 기준 벡터: \( \begin{bmatrix} 1 \\ 3 \end{bmatrix} \)

- 곰 기준 벡터:

$$
B^{-1} \cdot \begin{bmatrix} 1 \\ 3 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 \cdot 1 + 1 \cdot 3 \\
-1 \cdot 1 + 1 \cdot 3
\end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix}
4 \\
2
\end{bmatrix} = \begin{bmatrix}
2 \\
1
\end{bmatrix}
$$

---

### 6. 직교 기준일 때는 **투영(dot product)** 으로 변환 가능

- 내 벡터: \( \mathbf{v} = \begin{bmatrix} 1 \\ 3 \end{bmatrix} \)

- 곰의 기준 \( \mathbf{b}_1, \mathbf{b}_2 \)에 대한 투영:

$$
\mathbf{v}_{\text{Bear}, 1} = \mathbf{v} \cdot \mathbf{b}_1 = \frac{1}{\sqrt{2}} (1 + 3) = 2 \\
\mathbf{v}_{\text{Bear}, 2} = \mathbf{v} \cdot \mathbf{b}_2 = \frac{1}{\sqrt{2}} (-1 + 3) = 1
$$

⇒ 곰 기준 벡터: \( \begin{bmatrix} 2 \\ 1 \end{bmatrix} \)

※ 단, 곰의 기준 벡터가 **직교가 아닐 경우**에는 dot product를 통한 변환은 **불가능**, 반드시 행렬 \( B^{-1} \) 이용해야 함

---

### 7. 요약

- **기준 변환은 변환 행렬 B 또는 그 역행렬 \( B^{-1} \) 을 이용하여 수행**
- **직교 기준(orthonormal basis)** 인 경우 **dot product** 를 통해도 변환 가능
- **B: 곰의 기준 벡터들을 내 좌표계에서 표현한 것**
- **\( B^{-1} \): 내 기준 벡터를 곰의 좌표계에서 표현한 것**




## Doing a transformation in a changed basis

### 1. 문제 설정

- **Bear의 기준 벡터**:
  - \( \mathbf{b}_1 = [3, 1] \)
  - \( \mathbf{b}_2 = [1, 1] \)

- 벡터 \( \begin{bmatrix} x \\ y \end{bmatrix} \) 가 **Bear의 기준**에서 주어졌을 때,
- 이 벡터에 **45도 회전 변환**을 적용하고자 함.

---

### 2. 회전 행렬 \( R \) (45도)

- 회전 행렬은 **기본 좌표계** 기준:

$$
R = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & -1 \\
1 & 1
\end{bmatrix}
$$

---

### 3. 기준 변환 행렬 \( B \)

- Bear의 기준을 내 기준에서 표현한 행렬:

$$
B = \begin{bmatrix}
3 & 1 \\
1 & 1
\end{bmatrix}
$$

- 역행렬 \( B^{-1} \):

$$
B^{-1} = \frac{1}{2} \begin{bmatrix}
1 & -1 \\
-1 & 3
\end{bmatrix}
$$

---

### 4. Basis에서의 변환 순서

1. Bear 기준 벡터 \( \mathbf{v}_{\text{Bear}} \) 를 내 기준으로 변환:  
   \( \mathbf{v}_{\text{Me}} = B \cdot \mathbf{v}_{\text{Bear}} \)

2. 내 기준에서 45도 회전 적용:  
   \( \mathbf{v}'_{\text{Me}} = R \cdot \mathbf{v}_{\text{Me}} \)

3. 결과를 다시 Bear 기준으로 변환:  
   \( \mathbf{v}'_{\text{Bear}} = B^{-1} \cdot \mathbf{v}'_{\text{Me}} \)

⇒ 전체 변환은 다음과 같이 표현 가능:

$$
\mathbf{v}'_{\text{Bear}} = B^{-1} R B \cdot \mathbf{v}_{\text{Bear}}
$$

---

### 5. 핵심 결론

- **\( B^{-1} R B \)** 는 Bear의 좌표계에서 수행된 회전 행렬을 의미함.
- **중요한 공식**:  
  $$ T_{\text{new}} = B^{-1} R B $$
  이 식은 **기준이 변한 공간에서 변환을 수행하는 핵심 공식**이다.

---

### 6. 응용

- 이 접근은 **기준 좌표계가 직교가 아니더라도** 사용할 수 있으며,
- **PCA(주성분 분석)** 와 같은 기법에서 **새로운 기준축**에서의 변환을 구현할 때 매우 유용하다.

---

### 7. 요약

- 기준이 바뀌면 변환 행렬도 바뀜.
- **기준 변환 + 변환 적용 + 기준 역변환**의 조합을 사용해야 함.
- 공식:
  $$
  \text{Transformed Vector in New Basis} = B^{-1} R B \cdot \text{Vector in New Basis}
  $$

## Making Multiple Mappings, deciding if these are reversible: Orthogonal matrices


### 1. Transpose (전치행렬)

- 행렬 \( A \)의 전치 \( A^T \): 행과 열을 뒤바꿈
- 즉, 원래 행렬 \( A \)의 \( (i, j) \) 원소는 \( A^T \)에서 \( (j, i) \) 위치에 존재함
- 예시:
  $$
  A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \Rightarrow A^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
  $$

---

### 2. Orthonormal Basis (직교 정규 기저)

- 행렬 \( A \)의 열벡터 \( \mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n \) 이 다음 조건을 만족한다면:

  1. \( \mathbf{a}_i \cdot \mathbf{a}_j = 0 \) for \( i \ne j \) (서로 수직)
  2. \( \mathbf{a}_i \cdot \mathbf{a}_i = 1 \) for all \( i \) (단위 벡터)

- 이 조건을 만족하는 기저를 **Orthonormal Basis** 라고 부르며,
- 이 열벡터들로 구성된 행렬 \( A \)를 **Orthogonal Matrix (직교행렬)** 라고 부름

---

### 3. Transpose = Inverse

- **중요 성질**:  
  만약 \( A \)가 직교행렬이라면,

  $$
  A^T A = AA^T = I
  $$

  → 전치 행렬 \( A^T \)이 \( A \)의 역행렬이 됨

- **따라서**:
  $$
  A^{-1} = A^T
  $$

---

### 4. Orthogonal Matrix의 성질

- 모든 열벡터와 행벡터가 **서로 직교하고 단위 벡터**
- **공간을 보존**함 (길이와 각도를 유지)
- **행렬식 (Determinant)** 은 반드시:

  $$
  \det(A) = \pm 1
  $$

  - \( +1 \): 공간의 방향 유지 (우수 좌표계)
  - \( -1 \): 공간 반전 (좌수 좌표계)

---

### 5. 데이터 변환 관점에서의 이점

- Orthonormal basis를 사용할 경우:

  - 역행렬 계산이 쉬움 (\( A^T \) 사용)
  - 변환이 **가역적**
  - **투영 (Projection)** 계산이 단순한 **점곱(Dot Product)** 으로 가능
  - 공간 왜곡 없이 보존됨

- ⇒ 데이터 과학에서는 가능한 한 **직교행렬 기반의 변환 행렬**을 선호

---

### 6. 요약

- 전치: 행렬의 행과 열을 교환
- 직교행렬: 열벡터가 직교 정규 기저인 행렬
- **직교행렬의 역행렬은 전치행렬**
```math
A^{-1} = A^T \quad \text{(if  A  is  orthogonal)}
```
- 행렬식은 \( \pm 1 \), 공간 보존 및 가역성 보장
- 데이터 변환 시 계산 효율성과 직관성을 모두 제공함











