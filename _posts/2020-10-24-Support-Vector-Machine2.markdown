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

From [last post](https://jennycs005.github.io/2020/10/10/Support-Vector-Machine1/), we got the basic algorithm of SVM in linearly separable cases. In this post, we are going to take one step further and talk about SVM in non-linearly separable cases. In that case, it's hard to find a hyperplane, even if we got one, it maybe a result of overfitting and has very bad generalization ability(Like shown in the figure below). So we introduce kernel functon and kernel tricks to convert input space into a high dimensional feature space which makes it possible to get a optimal separating hyperplane.
![img](/img/in-post/post-2020-10-10-SVM/post-SVM-01.png)

## Lagrange multiplier and Lagrangian function

We have the dicision function now, and now we're going to talk about how to come to the solutions of it.

$\mathop{min}\limits_{ω,b} \frac{1}{2}{\lVert ω \rVert}^2$ **(SVM1-form-2)**

$s. t. y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert}) - 1> 0, i = 1, 2, ..., N$

This is an example of a quadratic programming problem. In order to solve this constrained optimization problem, we introduce **Lagrange multiplier $α_i\geqslant 0$**, giving the **Lagrangian function**

$L(ω, b, α) = \frac{1}{2}{\lVert ω \rVert}^2 + \Sigma α_i(1-y_i(ω^Tx_i+b))$ **(SVM2-form-1)**

Setting the derivatives of $L(ω, b, α)$ with respect to $ω$ and $b$ equal to zero, we obtain the following two conditions:

$ω = \Sigma α_iy_ix_i$  **(SVM2-form-2)**

$0 = \Sigma α_iy_i$

Eliminating  $ω$ and $b$ from $L(ω, b, α)$ using these conditions then giving the dual representation of the maximum margin problem in which we maximize

$\tilde{L}(α) = \Sigma α_i - \frac{1}{2}\Sigma\Sigma α_i α_j y_i y_j k(x_i, x_j)$ **(SVM2-form-3)**

with respect to subject to the constraints

$α_i\geqslant 0 \$

$\Sigma α_i y_i =0$

In order to classify new data points using the trained model, we evaluated the sign of $y(x)$ defined by SVM1-form-1 $f(x)=ωx+b$, this can be expressed in terms of the parameters $α$ and the kernel function by substituting for $ω$ using SVM2-form-2 to give

$y(x) = \Sigma α_iy-i{x_i}^Tx_j +b $
$y(x) = \Sigma α_iy-ik(x_i,x_j) +b$  **(SVM2-form-4)**

## KKT conditions

A constrainted optimization of this form satisfies the **Karush-Kuhn-Tucker conditions**, which in this case requires the following three properties hold

* $α_i\geqslant 0$

* $y_if(x_i)-1\geqslant 0$

* $α_i(y_if(x_i)-1)=0$



## Conclusion

Now we come to the summary of SVM algorithm in linearly separable cases.

**Input:** Linear separable training dataset $T = \lbrace(x_1,y_1),(x_2,y_2),...(x_n,y_n)\rbrace$, for which $x_i \in X=R^n$, $y_i \in Y=\brace-1, +1\rbrace, i = 1,2,...,N$

**Output:** Maximum margin separating hyperplane and classify decision function

* Step 1: Construct and solve constrained optimization problem:

$\mathop{min}\limits_{ω,b} \frac{1}{2}{\lVert ω \rVert}^2$     **(SVM1-form-2)**

$s. t.  y_i(\frac{ω}{\lVert ω \rVert}x_i+\frac{b}{\lVert ω \rVert}) - 1> 0, i = 1, 2, ..., N$

Get optimal solution $ω^\*, b^\*$.

* Step 2: Get separating hyperplane:

$ω^\* x_i + b^\* = 0$

classify decision function:

$f(x) = sign(ω^\* x_i + b^\*)$

In my [next post](https://jennycs005.github.io/2020/10/24/Support-Vector-Machine3/), we're goint to talk about non-linearly separable cases.

—— Jennycs005 @ 10102020
