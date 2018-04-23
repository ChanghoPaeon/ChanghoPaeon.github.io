---
title:  How to Solve Gradient Vanishing in NLP
date: 2018-04-15
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

작성중


![simple-nn-dev-weight-respect-to-1](https://user-images.githubusercontent.com/27984736/38817428-4e544384-41d3-11e8-9a22-a5e99e27b579.png)

$$W_{1}$$ 에 대한 편미분은 아래와 같음.

$$\begin{align}\frac{\partial Loss}{\partial W_1}& = \frac{\partial Loss}{\partial f(z_3)} \cdot \frac{\partial f(z_3)}{\partial f(z_2)} \cdot \frac{\partial f(z_2)}{\partial f(z_1)} \cdot \frac{\partial f(z_1)}{\partial W_1} \\
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


#### $$tanh$$
다음은 vanila rnn 등 여러 곳에서 사용되는 activation funtion인 $$tanh$$ 의 정의이다.

$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}}$$

분모 분자에  $$e^{-x}$$ 를 곱하면, 

$$ \tanh(x) = \frac{1 - e^{-2x}}{1+e^{-2x}}$$

을 얻는다. 이를 미분하면, 

$$ \tanh'(x) = (1+tanh(x))(1-tanh(x))$$

가 된다.

$$sigmoid$$ 함수와 같은 방법으로 최대값을 계산하면, $$\sigma(0)=1$$ 이고 $$0$$을 제외한 임의의 실수 $$x$$ 에 대해 $$\sigma(x) < 0 $$ 임을 알수 있다.
따라서 Gradient Vanishing Problem 이 발생함을 쉽게 확인 할 수 있다.

### ReLU 의 등장

#### ReLU 의 정의 

$$ ReLU(x) =  max(x, 0)$$ 

![relu](https://user-images.githubusercontent.com/27984736/38817445-5c77f050-41d3-11e8-8a7e-d12394b39302.PNG)

#### ReLU의 derivation







### RNN
RNN 을 설명하기에 앞서, 앞으로 나올 image 에 대한 기호를 정의하하겠다.

![lstm2-notation](https://user-images.githubusercontent.com/27984736/38818204-2a3f1080-41d5-11e8-90e0-25c484c99209.png)


 - 추후 나올 이미지의 출처는 http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 이다.


![lstm3-simplernn](https://user-images.githubusercontent.com/27984736/38818175-11eed2e0-41d5-11e8-9eb3-2e109b0c0c12.png)

#### Formula of RNN

$$h_{t}$$ : hidden state at $$t$$

$$x_{t}$$ : input at $$t$$

$$W$$ : weight matrix

$$h_{t} = tanh(Wh_{t-1} + W^{e}x_{t}) $$

$$y_{t} = W^{s}h_{t} $$

#### Error back-propagation of RNN

Error of RNN defined as summation of error at each time step.

i.e. 

$${\partial E\over\partial W} = 	\sum_{t=1}^N{\partial E_{t}\over\partial W} $$

and

$${\partial E_{t}\over\partial W} = 	\sum_{k=1}^N{\partial E_{t}\over\partial y_{t}}{\partial y_{t}\over\partial h_{t}}{\partial h_{t}\over\partial h_{k}}{\partial h_{k}\over\partial W}$$

Chain Rule 에 의해 

$${\partial h_{t}\over\partial h_{k}} =\prod_{j=k+1}^t {\partial h_{j}\over\partial h_{j-1}}  $$

이다.  $${\partial h_{j}\over\partial h_{j-1}}$$ 에 $$norm$$ 을 취하면,아래와 같은 부등식이 성립한다.

$$ \parallel  {\partial h_{j}\over\partial h_{j-1}} \parallel \le \parallel W^{T} \parallel \parallel diag(f'(h_{j-1})) \parallel$$

vanila RNN 에서의 activation function $$f$$는 $$tanh$$이므로, 

$${\partial h_{t}\over\partial h_{k}} =\prod_{j=k+1}^t {\partial h_{j}\over\partial h_{j-1}}  $$ 

에서 gradient vanishing이 발생한다.




### RNN with ReLU

ReLU 도함수의 특성으로, 더 깊은 layer를 학습시킬 수 있었던 것에 착안하여, RNN 에서도 Activation 함수를 ReLU로 바꾸어 학습시킬려는 시도가 있었다.(A Simple Way to Initialize Recurrent Networks of Rectified Linear Units, Le et al. 2015)



### GRU

![gru](https://user-images.githubusercontent.com/27984736/38818965-8dd64f5a-41d5-11e8-952b-53ebf8050119.png)


### LSTM


![lstm3-chain](https://user-images.githubusercontent.com/27984736/38818188-20edfdca-41d5-11e8-9aef-700f1969260d.png)






