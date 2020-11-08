---
layout:     post
title:      "Support Vector Machine 1"
subtitle:   "Linear Separable Cases "
date:       2020-10-10 12:00:00
author:     "Jennycs005"
header-img: "img/post-bg-SVM-background.jpg"
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

**Support Vector Machine (SVM)** is a supervised machine learning algorithm and it can be used for both classification and regression challenges. However, it is mostly used in classification problems. SVM is highly preferred by many as it produces significant accuracy with less computation power. The basic idea of SVM is to find the optimal separating hyperplane in feature space which maximizes the margin and then categorize training data set into different groups. It can also be considered as a convex quadratic programming problem. We're going to discuss linear SVM in this post, and we suppose the feature space is linear separatable.

## Functional and Geometric Margin
![img](/img/in-post/post-2020-10-10-SVM/post-SVM-01.png)

We take 2-dimensional feature space for example. In the figure above, we have feature $X_1$ and feature $X_2$, and now we want to classify the input data into two groups(Of course SVM is capable of milti-class classification, we'll talk about this in another post). It's very obviously that we can simply draw a line to seperate them, and define the line as a 'hyperplane', but it looks like there are many hyperplanes could be drawn. 

So which one is the **optimal hyperplane**? 

The idea is to choose the one with the **largest margin**. The larger the margin is, the more confidence the classifier will have. In other words, we choose the one with the best generalization ability. 

![img](/img/in-post/post-2020-10-10-SVM/post-SVM-02.png)

Let's discuss it in depth. We define the hyperplane as 

$f(x)=ωx+b$  **(SVM1-form-1)**

For any point $(x_i, y_i)$ in feature space, $\lvert ω{x_i}+b\rvert$ is the distance from $x_i$ to the hyperplane. So the hyperplane is where $f(x)=ωx+b=0$. The red line and green line are the boundaries where $f(x)=1$ and $f(x)=-1$. And the data points beyond those boundaries are categorized to either positive or negative groups. Thus we got the decision function:

$f(x)=sign(ωx+b)$ **(SVM1-form-2)**

Next step is to find the value of $ω$ and $b$.

Now let's consider the sign of $y_i(ω{x_i}+b)$. If it's positive, that means the point $(x_i, y_i)$ is properly classifed($y_i$ and $(ω{x_i}+b)$ are both positive or negative at the same time. So we denote 

$\hat γ_i = y_i(ω{x_i}+b)$ **(SVM1-form-3)**

as the **functional margin**, it's just a testing function that tell us whether the point is properly classified or not.

Scalling functional margin by $\lVert ω \rVert$, we got **geometric margin** 

$γ_i = y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert})$. **(SVM1-form-4)**

The geometric margin is showing not only if the point is properly classified or not, but also the magnitude of that distance in term of units of $\lVert ω \rVert$. This is import because $(λω, λb)$ is the same hyperplane with $(ω, b)$, but they have different functional margin. The smallest geometric margin(this margin is from data points in feature space to separating hyperplane) is half of the largest margin(this margin is the distance between red line and green line) we're looking for.

## Largest Margin

To get the largest margin, the problem could be descriped as follow:

$\mathop{max}\limits_{ω,b} γ$                                            **(SVM1-form-5)**

$s. t.  y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert}) > γ, i = 1, 2, ..., N$

Since 
* $γ = \frac{\hat γ}{\lVert ω \rVert}$ 

* $(λω, λb)$ is the same hyperplane with $(ω, b)$, and we set functional margin $\hat γ = 1$ 

* $\mathop{max} \frac{1}{\lVert ω \rVert}$ is equal to $\mathop{min} \lVert ω \rVert$, and we add $\frac{1}{2}$ in front of it for further calculation convenience

the SVM1-form-4 could be convert to:

$\mathop{min}\limits_{ω,b} \frac{1}{2}{\lVert ω \rVert}^2$               **(SVM1-form-6)**

$s. t.  y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert}) - 1> 0, i = 1, 2, ..., N$

When we got $ω^\*, b^\*$ from the function above and have the **optimal hyperplane**

$0 = {ω^\*}x+{b^\*}$                                                    **(SVM1-form-7)**

also **decision function**

$f(x) = sign({ω^\*}x+{b^\*})$                                           **(SVM1-form-8)**

## Lagrangian Function and Dual Problem

Now we talk about how to get the solutions of SVM1-form-5. 

This is an example of a quadratic programming problem. In order to solve this constrained optimization problem, we introduce **Lagrange multiplier $α_i\geqslant 0$**, giving the **Lagrangian function**

$L(ω, b, α) = \frac{1}{2}{\lVert ω \rVert}^2 + \sum\limits_{i=1}^{N} α_i(1-y_i(ωx_i+b))$ **(SVM1-form-9)**

According to Lagrange duality, the dual problem of origenal primary problem is:

$\mathop{max}\limits_{α}\mathop{min}limits_{ω,b}L(ω, b, α)$            

First we calculate $\mathop{min}limits_{ω,b}L(ω, b, α)$: setting the derivatives of $L(ω, b, α)$ with respect to $ω$ and $b$ equal to zero, we obtain the following two conditions:

* $ω = \sum\limits_{i=1}^{N} α_iy_ix_i$                                 **(SVM1-form-10)**

* $0 = \sum\limits_{i=1}^{N} α_iy_i$                                    **(SVM1-form-11)**

Eliminating  $ω$ and $b$ from $L(ω, b, α)$ using these conditions then giving the dual representation of the maximum margin problem in which we maximize

$\mathop{max}\limits_{α}L(ω, b, α)=\tilde{L}(α) = \sum\limits_{i=1}^{N} α_i - \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} α_i α_j y_i y_j(x_i\bullet x_j)$  **(SVM1-form-12)**

Using it's dual problem again we have

$\mathop{min}\limits_{a}\frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} α_i α_j y_i y_j(x_i, x_j)-\sum\limits_{i=1}^{N} α_i$   **(SVM1-form-13)**

with respect to subject to the constraints

* $α_i\geqslant 0 $     **(SVM1-form-14)**

* $\sum\limits_{i=1}^{N} α_i y_i =0$       **(SVM1-form-15)**

In order to classify new data points using the trained model, we evaluated the sign of $y(x)$ defined by SVM1-form-1 $f(x)=ωx+b$, this can be expressed in terms of the parameters $α$ and the kernel function by substituting for $ω$ using SVM1-form-10 to give

$y(x) = \sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} α_i y_i(x_i\bullet x_j) +b$  **(SVM1-form-16)**

## KKT Conditions and Support Vectors

A constrainted optimization of this form satisfies the [**Karush-Kuhn-Tucker conditions**](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions), which in this case requires the following three properties hold

* $α_i\geqslant 0$

* $y_if(x_i)-1\geqslant 0$

* $α_i(y_if(x_i)-1)=0$

So for every data points, either $α_i = 0$ or $y_if(x_i)=1$. Any points for which $α_i = 0$ will not appear in the SVM1-form-16 and could be discard. The remaining data points which safisfied $y_if(x_i)=1$ are the **support vectors**, those are the vectors that determine the hyperplane and decision boundaries. That's why SVM got it's name and the reason why it could be used on small dataset and deliver promising result.

Having solved the quadratic programming problem and found a value for $α^\*$, we can then determine the value of the threshold parameter $b^\*$ by noting that any support vector $x_i$ satifies $y_if(x_i) = 1$. This gives 

$y_j(\sum\limits_{i=1}^{N} α_iy_i(x_i\bullet x_j)+b)=1$ **(SVM1-form-17)**

where $N$ denotes the set of inices of the support vectors.

we first multiply through by $y_i$, making use of $(y_n)^2 = 1$, and then averaging these equations over all support vectors and solving for $b$ to give 

$b^\* = y_i - \sum\limits_(i=1)^{N}α_i^\*y_i(x_i\bullet x_j)$  **(SVM1-form-18)**

where $N$ is the total number of the support vectors.

## Conclusion

Now we come to the summary of SVM algorithm in linear separable cases.

**Input:** Linear separable training dataset $T = \lbrace(x_1,y_1),(x_2,y_2),...(x_n,y_n)\rbrace$, for which $x_i \in X=R^n$, $y_i \in Y=\lbrace-1, +1\rbrace, i = 1,2,...,N$

**Output:** Maximum margin separating hyperplane and classify decision function

* Step 1: Construct and solve constrained optimization problem:

$\mathop{min}\limits_{α} \frac{1}{2}\sum\limits_(i=1)^{N}\sum\limits_(j=1)^{N}α_iα_jy_iy_j(x_i\bullet x_j)-\sum\limits_{i=1}^{N}α_i$    

$s. t. \sum\limits_{i=1}^{N} α_i y_i =0$  

$α_i\geqslant 0  , i = 1, 2, ..., N$

Get optimal solution $α^\*=(α_1^\*,α_2^\*,...α_N^\*)^T$.

* Step 2: Calculate 

$ω^\*=\sum\limits_{i=1}^{N} α_i^\*y_ix_i$

choose $α_j^\* > 0$, calculate

$b^\*=y_i - \sum\limits_(i=1)^{N}α_i^\*y_i(x_i\bullet x_j)$

* Step 3:Get separating hyperplane:

$ω^\* x_i + b^\* = 0$

classify decision function:

$f(x) = sign(ω^\* x_i + b^\*)$

In my [next post](https://jennycs005.github.io/2020/10/24/Support-Vector-Machine2/), we're goint to talk about non-linear separable cases.

—— Jennycs005 @ 10102020


