---
title:  How to Solve Gradient Vanishing in NLP
date: 2018-04-17
categories:
- NLP
tags:
- NLP
- Gradient Vanishing Problem
- RNN
- LSTM
- GRU
---


### Gradient Vanishing Problem



![simple-nn-dev-weight-respect-to-1](https://user-images.githubusercontent.com/27984736/38817428-4e544384-41d3-11e8-9a22-a5e99e27b579.png)

$$W_{1}$$ 에 대한 편미분은 아래와 같음.

$$\begin{align}\frac{\partial Loss}{\partial W_1}& = \frac{\partial Loss}{\partial f(z_3)} \cdot \frac{\partial f(z_3)}{\partial f(z_2)} \cdot \frac{\partial f(z_2)}{\partial f(z1)} \cdot \frac{\partial f(z_1)}{\partial W_1} \\
& = \frac{\partial Loss}{\partial f(z_3)} \cdot f'(z_3) \cdot W_3 \cdot f'(z_2) \cdot W_2 \cdot f'(z_1) \cdot W_1 \end{align}$$

 Activation funtion 으로 많이 사용 되는 sigmoid 함수는 다음과 같다.

$$\sigma(x) = \frac{1}{1+e^{-x}}$$ 

미분을 하면

$$\begin{align}
\frac{d}{dx}\sigma(x) & = \frac{d}{dx}{(1+e^{-x})^{-1}} \\ 
& = (-1)\frac{1}{(1+e^{-x})^{2}}\frac{d}{dx}(1+e^{-x}) \\ 
& = (-1)\frac{1}{(1+e^{-x})^{2}}(0+e^{-x})\frac{d}{dx}(-x) \\ 
& = (-1)\frac{1}{(1+e^{-x})^{2}}e^{-x}(-1)  \\ 
& = \frac{e^{-x}}{(1+e^{-x})^{2}}  \\ 
& = \frac{1+e^{-x}-1}{(1+e^{-x})^{2}}  \\ 
& = \frac{(1+e^{-x})}{(1+e^{-x})^{2}}-\frac{1}{(1+e^{-x})^{2}}  \\ 
& = \frac{1}{1+e^{-x}}-\frac{1}{(1+e^{-x})^{2}}  \\ 
& = \frac{1}{1+e^{-x}}(1-\frac{1}{1+e^{-x}}) \\ 
& = \sigma(x)(1-\sigma(x))
\end{align}$$

$$\sigma(x)$$ 를 $$t$$ 로 치환하여, 미분하면, $$\sigma(x)(1-\sigma(x))$$의 최대값이 $$0.25$$ 임을 쉽게 확인 할 수 있다.

따라서 층이 깊어질수록 에러가 잘 전파되지 않음

아래는 $$\sigma'(x)$$의 그래프이다.

![dev-of-simoid](https://user-images.githubusercontent.com/27984736/38817443-5adecab6-41d3-11e8-9d57-4b6633af2355.png)


##### $$tanh$$
다음은 vanila rnn 등 여러 곳에서 사용되는 activation funtion인 $$tanh$$ 의 정의이다.

$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}}$$

분모 분자에  $$e^{-x}$$ 를 곱하면, 

$$ \tanh(x) = \frac{1 - e^{-2x}}{1+e^{-2x}}$$

을 얻는다. 이를 미분하면, 

$$ \tanh'(x) = (1+tanh(x))(1-tanh(x))$$

가 되어 Gradient Vanishing Problem 이 발생함을 쉽게 확인 할 수 있다.


### ReLU 의 등장

##### ReLU 의 정의 

$$ ReLU(x) =  max(x, 0)$$ 

![relu](https://user-images.githubusercontent.com/27984736/38817445-5c77f050-41d3-11e8-8a7e-d12394b39302.PNG)

##### ReLU의 derivation





### RNN with ReLU



### GRU


### LSTM








