---
layout:     post
title:      "Support Vector Machine 1"
subtitle:   "Linear SVM -- Trying so hard to find largest margin"
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

Let's discuss it in depth. We define the hyperplane as $f(x)=w^Tx+b$. when $f(x)=0$, it means $x$ is exactly on the hyperplane; when $f(x)>0$, then $x$ is on the positive category, where $f(x)=1$; when $f(x)<0$, $x$ is on the negative category, where $f(x)=-1$. 

For any point $(x_i, y_i)$ in feature space, $\lvert w^T{x_i}+b\rvert$ is the distance from $x_i$ to the hyperplane. Let's consider the sign of $y_i(w^T{x_i}+b)$, if it's positive, that means the point $(x_i, y_i)$ is properly classifed. $\hat γ_i = y_i(w^T{x_i}+b)$ is called the **functional margin**, it's just a testing function that tell us whether the point is properly classified or not.

Scalling functional margin by $\lVert w \rVert$, we got **geometric margin** $γ_i = y_i(\frac{w}{\lVert w \rVert}x_i+\frac{b}{\lVert w \rVert})$. The geometric margin is showing not only if the point is properly classified or not, but also the magnitude of that distance in term of units of $\lVert w \rVert$. The smallest geometric margin(this margin is from data points in feature space to separating hyperplane) is half of the largest margin(this margin is the distance between red line and green line) we're looking for.


To get the largest margin, the problem could be descriped as follow:

$\mathop{max}\limits_{w,b} γ$

$s. t.  y_i(\frac{w}{\lVert w \rVert}x_i+\frac{b}{\lVert w \rVert}) > γ, i = 1, 2, ..., N$

Since 
* $γ = \frac{\hat γ}{\lVert w \rVert}$ 
* $(λw, λb)$ is the same hyperplane with $(w, b)$, set functional margin $\hat γ = 1$ 
* $\mathop{max} \frac{1}{\lVert w \rVert}$ is equal to $\mathop{min} \frac{1}{2} \lVert w \rVert$

the furmula could be convert to:

$\mathop{min}\limits_{w,b} \frac{1}{2}{\lVert w \rVert}^2$     (function 1)

$s. t.  y_i(\frac{w}{\lVert w \rVert}x_i+\frac{b}{\lVert w \rVert}) - 1> 0, i = 1, 2, ..., N$

We got $w^\*, b^\*$ from the function above and have the optimal hyperplane

$f(x) = {w^\*}x+{b^\*}$

In linear separable cases, the closest points to separating hyperplane are called **support vectors**. Thoses are the points where $y_i(wx_i+b)-1=0$(red line and green line in the picture). The distance between red line and green line is the margin, which is $\frac{2}{\lVert w \rVert}$. Support Vectors are the only points deciding the optimal separating hyperplane. That means if support vectors are moved, hyperplane is moved; if other points being moved or deleted, hyperplane is not moved. Because of that, SVM could be used on small dataset since only a small part of set is being used.

## Dual Problem

Now we're going to discuss how to solve the function 1 above. It is a convex quadratic programming problem, which is easy to solve by standard methods. We solve it by geting the results of it's dual problem.






—— Jennycs005 @ 10102020


