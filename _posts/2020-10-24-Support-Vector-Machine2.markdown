---
layout:     post
title:      "Support Vector Machine 2"
subtitle:   "Non-Linear Separable Cases "
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

From [last post](https://jennycs005.github.io/2020/10/10/Support-Vector-Machine1/), we got the basic algorithm of SVM in linear separable cases. In this post, we are going to take one step further and talk about SVM in non-linear separable cases. In that case, it's hard to find a hyperplane in the orginal input space, even if we got one, it maybe a result of overfitting and has very bad generalization ability(Like shown in the figures below). So we introduce kernel functon and kernel tricks to convert input space into a high dimensional feature space, so the origenal non-linear separable problem now being convert to a linear separable problem and possible to find an optimal separating hyperplane.
![img](/img/in-post/post-2020-10-24-SVM2/post-SVM2-01.png)


## Kernel Function and Kernel Tricks

## Common Kernels

## Conclusion

In summary, the SVM algorithm in non-linearly separable cases could be expressed as follows:

**Input**: 

Linearly separable training dataset $T = \lbrace(x_1,y_1),(x_2,y_2),...(x_n,y_n)\rbrace$, for which $x_i \in X=R^n$, $y_i \in Y=\lbrace-1, +1\rbrace, i = 1,2,...,N$


**Output**:

Maximum margin separating hyperplane and classify decision function

* Step 1: 

Construct and solve constrained optimization problem:

$\mathop{min}\limits_{ω,b} \frac{1}{2}\frac{1}{2}\Sigma\Sigma α_i α_j y_i y_j k(x_i, x_j)-\Sigma α_i$     **(SVM1-form-2)**

$s. t.  \Sigma α_i y_i = 0$

$0 \leqslant α_i \leqslant C, i = 1, 2, ..., N$

Get optimal solution $α^\*=(α_1^\*,α_2^\*,...,α_N^\*)^T$.

* Step 2: 

Choose one of the positive component of $0 \leqslant (α_j)^T \leqslant C$, calculate:

$b^\* = y_i - \Sigma α_i^\* y_i K(x_i, x_j)$


* Step 3: 

classify decision function:

$f(x) = sign(\Sigma α_i^\* y_i K(x_i, x_j)+ b^\*)$


In my [next post](https://jennycs005.github.io/2020/10/24/Support-Vector-Machine3/), we're goint to talk about non-linear SVM, which means SVM with outliers.

—— Jennycs005 @ 10242020
