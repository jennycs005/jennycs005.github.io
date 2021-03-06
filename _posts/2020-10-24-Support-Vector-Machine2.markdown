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

From [last post](https://jennycs005.github.io/2020/10/10/Support-Vector-Machine1/), we obtained the basic algorithm of SVM in linear separable cases. In this post, we are going to take one step further and talk about SVM in non-linear separable cases. In thses cases, it's hard to find a hyperplane in the orginal input space, even if we got one, it maybe a result of overfitting and has very bad generalization ability(Like shown in the images below). So we introduce kernel functon and kernel trick to transform input space into a higher dimensional feature space, so the origenal non-linear separable problem now being transfered to a linear separable problem and possible to find an optimal separating hyperplane.
![img](/img/in-post/post-2020-10-24-SVM2/post-SVM2-01.png)


## Kernel Function and Kernel Tricks

Let's talk about it in details.

Like the left image above, we couldn't divide original dataset into different groups by a line(like in linear separable model), but there exists a curve to separate the data points. We transform it to linear separable case by applying the kernel trick in which adding one more dimention to the input space and get the new feature space. The hyperplane could be drawn as shown in the image below.

![img](/img/in-post/post-2020-10-24-SVM2/post-SVM2-02.png)

It's the same for the right image above, we use kernel function to transform the two-dimensional input space to three-dimensional feature space and could easily find the separating hyperplane.

![img](/img/in-post/post-2020-10-24-SVM2/post-SVM2-03.png)

The transformed data points are $z= \phi(x)$

If for all $x_i, x_j\in X=R^n$, function $k(x_i, x_j)$ satisfied $k(x_i,x_j)=\phi(x_i)\bullet \phi(x_j)$, then

$k(x_i, x_j)$ is the kernel function.

It's noteworth that in kernel trick, we only define kernel function, but not the mapping function, because usually  we can directly obtain $k(x_i, x_j)$ much easier than calculate from $\phi(x)$.

Next let's talk about how to use kernel trick in SVM. It's notable that in linear saperable cases, it's only about the dot product of input examples, so we can replace the dot product of the objective function of linear separable model

$\mathop{min}\limits_{α} \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N}α_iα_jy_iy_j(x_i\bullet x_j)-\sum\limits_{i=1}^{N}α_i$ 

with kernel function $k(x_i,x_j)=\phi(x_i)\bullet \phi(x_j)$ and the objective function for non-linear SVM is

$\mathop{min}\limits_{α} \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N}α_iα_jy_iy_jk(x_i, x_j)-\sum\limits_{i=1}^{N}α_i$

the classify decision function is

$f(x)=sign(\sum\limits_{i=1}^{N} α_i^\*y_ik(x,x_i)+b^\*)$

## Common Kernels

We list some common used Kernels here:

* **Linear Kernel Function**.

$K(x_i,x_j) = (x_i\bullet x_j)$

* **Polynomial Kernel Function**. 

$K(x_i,x_j) = (x_i\bullet x_j+1)^d$ 
where d is the degree of the polynomial.

* **Gaussian Kernel Function**. 

$K(x_i,x_j) = exp\lgroup-\frac{\Arrowvert x_i - x_j\Arrowvert^2}{2σ^2}\rgroup$

* **Radial Basis Function (RBF)**

$k(x_i,x_j) = exp\lgroup-\gamma\Arrowvert x_i -x_j \Arrowvert^2 \rgroup$
where $\gamma > 0$

* **Sigmoid Kernel**(also known as Tanh kernel)

$k(x_i,x_j) = tanh(\beta x_i x_j + c)$

My experience for choosing the proper Kernelfunction:

Let's say D is the number of dimentional of feature space, N is the number of data records,

* When D > N( which usually happens in Text classification problem), choose linear Kernel;

* When D is small and N is not very big, choose RBF Kernel;

* When D is small and N is very big, forget about SVM, choose Deep neural network.


## Conclusion

In summary, the SVM algorithm in non-linearly separable cases could be expressed as follows:

**Input**: 
Training dataset $T = \lbrace(x_1,y_1),(x_2,y_2),...(x_n,y_n)\rbrace$, for which $x_i \in X=R^n$, $y_i \in Y=\lbrace-1, +1\rbrace, i = 1,2,...,N$

**Output**:

Classify decision function

Step1. **Choose kernel function and parameter $C$, construct and solve constrained optimization problem**:

$\mathop{min}\limits_{\alpha}\frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} α_i α_j y_i y_j K(x_i, x_j)-\sum\limits_{i=1}^{N} α_i$     

$s. t.  \sum\limits_{i=1}^{N} α_i y_i = 0$

$0 \leqslant α_i \leqslant C, i = 1, 2, ..., N$

Get optimal solution $α^\*=(α_1^\*,α_2^\*,...,α_N^\*)^T$.

Step2. **Choose one of the positive component of $0 \leqslant (α_j)^T \leqslant C$, calculate**:

$b^\* = y_i - \sum\limits_{i=1}^{N} α_i^\* y_i K(x_i, x_j)$


Step3. **Obtain classify decision function**:

$f(x) = sign(\sum\limits_{i=1}^{N} α_i^\* y_i K(x_i, x_j)+ b^\*)$


In my [next post](https://jennycs005.github.io/2020/10/24/Support-Vector-Machine3/), we're goint to talk about SVM with outliers.

—— Jennycs005 @ 10242020
