---
layout:     post
title:      "Support Vector Machine 2"
subtitle:   "Non-Linearly Separable Cases "
date:       2020-10-24 12:00:00
author:     "Jennycs005"
header-img: "img/post-bg-SVM-background2.jpg"
catalog: true
tags:
    - ML
    - Classification
    - SVM
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

## Introduction

From [last post](https://jennycs005.github.io/2020/10/10/Support-Vector-Machine/), we got the basic algorithm of SVM in linearly separable cases. In this post, we are going to take one step further and talk about SVM in non-linearly separable cases. In that case, it's hard to find a hyperplane, even if we got one, it maybe a result of overfitting and has very bad generalization ability. So we introduce kernel functon and kernel tricks to convert input space into a high dimensional feature space which makes it possible to get a optimal separating hyperplane.

## Functional and Geometric Margin
![img](/img/in-post/post-2020-10-10-SVM/post-SVM-01.png)

We take 2-dimensional feature space for example. In the figure above, we have feature $X_1$ and feature $X_2$, and now we want to classify the input data into two groups(Of course SVM is capable of milti-class classification, we'll talk about this in another post). It's very obviously that we can simply draw a line to seperate them, and define the line as a 'hyperplane', but it looks like there are many hyperplanes could be drawn. 

So which one is the **optimal hyperplane**? 

The idea is to choose the one with the **largest margin**. The larger the margin is, the more confidence the classifier will have. In other words, we choose the one with the best generalization ability. 

![img](/img/in-post/post-2020-10-10-SVM/post-SVM-02.png)

Let's discuss it in depth. We define the hyperplane as $f(x)=ωx+b$. when $f(x)=0$, it means $x$ is exactly on the hyperplane; when $f(x)>0$, then $x$ is on the positive category, where $f(x)=1$; when $f(x)<0$, $x$ is on the negative category, where $f(x)=-1$. 

For any point $(x_i, y_i)$ in feature space, $\lvert ω{x_i}+b\rvert$ is the distance from $x_i$ to the hyperplane. Let's consider the sign of $y_i(ω{x_i}+b)$, if it's positive, that means the point $(x_i, y_i)$ is properly classifed. $\hat γ_i = y_i(ω{x_i}+b)$ is called the **functional margin**, it's just a testing function that tell us whether the point is properly classified or not.

Scalling functional margin by $\lVert ω \rVert$, we got **geometric margin** $γ_i = y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert})$. The geometric margin is showing not only if the point is properly classified or not, but also the magnitude of that distance in term of units of $\lVert ω \rVert$. The smallest geometric margin(this margin is from data points in feature space to separating hyperplane) is half of the largest margin(this margin is the distance between red line and green line) we're looking for.


To get the largest margin, the problem could be descriped as follow:

$\mathop{max}\limits_{ω,b} γ$

$s. t.  y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert}) > γ, i = 1, 2, ..., N$

Since 
* $γ = \frac{\hat γ}{\lVert ω \rVert}$ 
* $(λω, λb)$ is the same hyperplane with $(ω, b)$, set functional margin $\hat γ = 1$ 
* $\mathop{max} \frac{1}{\lVert ω \rVert}$ is equal to $\mathop{min} \frac{1}{2} \lVert ω \rVert$

the furmula could be convert to:

$\mathop{min}\limits_{ω,b} \frac{1}{2}{\lVert ω \rVert}^2$     (form 1)

$s. t.  y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert}) - 1> 0, i = 1, 2, ..., N$

We got $ω^\*, b^\*$ from the function above and have the optimal hyperplane

$f(x) = {ω^\*}x+{b^\*}$

In linear separable cases, the closest points to separating hyperplane are called **support vectors**. Thoses are the points where $y_i(ωx_i+b)-1=0$(red line and green line in the picture). The distance between red line and green line is the margin, which is $\frac{2}{\lVert ω \rVert}$. Support Vectors are the only points deciding the optimal separating hyperplane. That means if support vectors are moved, hyperplane is moved; if other points being moved or deleted, hyperplane is not moved. Because of that, SVM could be used on small dataset since only a small part of set is being used.

## Conclusion

Now we come to the summary of SVM algorithm in linearly separable cases.

**Input:** Linear separable training dataset $T = \{(x_1,y_1),(x_2,y_2),...(x_n,y_n)\}$, for which $x_i \in X=R^n$, $y_i \in Y=\{-1, +1\}, i = 1,2,...,N$

**Output:** Maximum margin separating hyperplane and classify decision function

* Step 1: Construct and solve constrained optimization problem:

$\mathop{min}\limits_{ω,b} \frac{1}{2}{\lVert ω \rVert}^2$     (form 1)

$s. t.  y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert}) - 1> 0, i = 1, 2, ..., N$

Get optimal solution $ω^\*, b^\*$.

* Step 2: Get separating hyperplane:

$ω^\* x_i + b^\* = 0$

classify decision function:

$f(x) = sign(ω^\* x_i + b^\*)$

In my [next post](https://jennycs005.github.io/2020/10/24/Support-Vector-Machine3/), we're goint to talk about non-linearly separable cases.

—— Jennycs005 @ 10102020
