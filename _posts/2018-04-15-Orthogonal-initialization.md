---
title: Orthogonal Initialization
date: 2018-04-15
categories:
- NLP
tags:
- NLP
- Linear Algebra
- Initialization
- RNN
- LSTM
- GRU
---
## 1. Introduction
#### Orthogonal

두 벡터 $$u, v$$ 에 대해, $$u \cdot v  = 0$$ 이 성립할때 $$orthogonal$$  하다고 정의하고 $$u \bot v$$ 로 표기한다.

$$R^{2}$$ 상에서  $$u \cdot v = \parallel\ u\parallel\parallel\ v\parallel cos \theta$$ 이 성립하므로, 

$$0$$이 아닌 두 벡터의 내적이 $$0$$ 이면 $$cos \theta = 0$$이 되어  $$ \theta = \pi/2$$, 또는  $$ \theta = 3\pi/2$$ 이 성립한다.

즉 기하학적으로 수직임을 의미한다.


#### Orthogonal Matrix

행벡터와 열벡터가 서로 $$orthogonal$$ 하고, 모두 단위벡터($$unit\ vector$$)인 실수 정방 행렬을 $$Orthogonal\  Matrix$$라 한다.
$$Q$$ 를 $$Orthogonal\  Matrix$$ 이라 하고 아래와 같이쓰자.

$$Q =  \begin{bmatrix}
-& u_1 & - \\
- & u_2 & - \\
 & \vdots &  \\
- & u_n & - 
\end{bmatrix}$$

그러면,

$$Q^{T} =  \begin{bmatrix}
| & | & \dots & |\\
u_1 & u_2 & \dots & u_n\\
| & |& \dots & | 
\end{bmatrix}$$

이 되고, $$Orthogonal\  Matrix$$ 정의에 의해 

$$QQ^{T} = Q^{T}Q = I$$ 

가 성립한다(사실 정의와 동치이다)

#### Property of Orthogonal Matrix

$$Orthogonal\  Matrix$$와 gradient vanishing, exploding 문제를 고찰하기 위해 한가지 사실을 짚고 가자.


$$Proposition 1.$$  $$Orthogonal\  Matrix$$ 의  eigen value 는 $$\pm 1$$이다.

$$proof)$$ $$x$$를 orthogonal matrix $$Q$$의 eigenvector라 하고 대응되는 eigenvalue 를 $$\lambda$$ 하자. 
그러면 eigenvector 정의에 의해 $$x\ne0$$ 이고 $$Qx = \lambda\ x$$ 을 얻는다.
위 식의 양변에 transepose 를 취하면, $$x^{T}Q^{T} = \lambda\ x^{T}$$이 되고 앞의 식과 곱하면, $$x^{T}Q^{T}Qx = \lambda\ x^{T}\lambda\ x$$이 된다.
정리하면 $$x^{T}x \ = \lambda\ x^{T}\lambda\ x \ = \lambda^{2}\ x^{T}x $$ 가 된다. 이를 한 변으로 이항하면 $$(1-\lambda^{2})x^{T}x = 0$$ 이 되고,  eigenvector 는 nonzero vector 이므로, $$\lambda^{2} = 1$$, 따라서 $$\lambda = \pm1$$이 된다 $$\Box$$



## 2. Orthogonal Matrix 는 어떻게 gradient vanishing, exploding을 막을 수 있는가?

앞서 RNN 에서의 gradient vanishing problem은 chain rule로 계산된 Loss 에 대한 weight의 편미분,

$$\begin{align}\frac{\partial Loss}{\partial W}& = \frac{\partial Loss}{\partial f(z_3)} \cdot \frac{\partial f(z_3)}{\partial f(z_2)} \cdot \frac{\partial f(z_2)}{\partial f(z1)} \cdot \frac{\partial f(z_1)}{\partial W} \\
& = \frac{\partial Loss}{\partial f(z_3)} \cdot f'(z_3) \cdot W \cdot f'(z_2) \cdot W \cdot f'(z_1) \cdot W \end{align}$$

에서 activation function $$f'(z)$$의 값들이 작아 발생함을 논증하였다. 그럼 vanising과 exploding 을 막기위해 activation function 으로 $$identity\ function$$을 사용하면 어떻게 될까? 위 식은 아래와 같이 바뀔 것이다.


$$\frac{\partial Loss}{\partial W_1} = \frac{\partial Loss}{\partial f(z_3)} \cdot W \cdot W  \cdot W = \frac{\partial Loss}{\partial f(z_3)} \cdot W^{3} $$

$$W$$의 diagonalization 을  $$W=P\mathit{\Lambda} P^{-1}$$라 하면,  $$W^3 =(P\mathit{\Lambda} P^{-1})(P\mathit{\Lambda} P^{-1})(P\mathit{\Lambda} P^{-1}) = P\mathit{\Lambda}^{3} P^{-1} $$ 을 얻게된다.

만약, $$W$$ eigenvalue가 $$1$$ 보다 크거나 작다면, vanishing 혹은 exploding 문제를 겪을 것이다.

이 문제가 초기부터 발생하는 것을 막기 위해, $$W$$ 을 $$orthogonal\ matrix$$ 로 초기화 하는 것이다.
