---
layout:     post
title:      "Support Vector Machine"
subtitle:   "Trying so hard to find the largest margin"
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

**Support Vector Machine (SVM)** is a supervised machine learning algorithm and it can be used for both classification and regression challenges. However, it is mostly used in classification problems. SVM is highly preferred by many as it produces significant accuracy with less computation power. The basic idea of SVM is to find the optimal separating hyperplane in feature space which maximizes the margin and then categorize training data set into different groups. It can also be considered as a convex quadratic programming problem. We're going to discuss both linear and non-linear SVM in this post.


## Linear SVM
![img](/img/in-post/post-2020-10-10-SVM/post-SVM-01.png)

We take 2-dimensional feature space for example. In the figure above, we have feature $X_1$ and feature $X_2$, and now we want to classify the input data into two groups(Of course SVM is capable of milti-class classification, we'll talk about this in another post). It's very obviously that we can simply draw a line to seperate them, and define the line as 'hyperplane', but it looks like there are many hyperplanes could be drawn. 

So which one is the **optimal hyperplane**? 

The idea is to choose the one with the **largest margin**. In other words, we choose the one with the best generalization ability. The larger the margin is, the more confidence the classifier will have.

So What is 'margin'? 

It’s the distance between the closest points from different classes in feature space( the distance between red line and green line in the figure below). We name the closest points as **Support Vectors**( Marked as red and green spots). Those are the actual data records that determine the hyperplane. 

![img](/img/in-post/post-2020-10-10-SVM/post-SVM-02.png)

Let's discuss it in depth. We define the hyperplane as $f(x)=wx+b$. when $f(x)=0$, it means $x$ is exactly on the hyperplane; when $f(x)>0$, then $x$ is on the positive category, where $f(x)=1$; when $f(x)<0$, $x$ is on the negative category, where $f(x)=-1$. 

For any point $(x_i, y_i)$ in feature space, $|wx_i+b|$ is the distance from $x_i$ to the hyperplane. Let's consider the sign of $y_i(wx_i+b)$, if it's positive, that means the point $(x_i, y_i)$ is properly classifed. $γ = y_i(wx_i+b)$ is called the **functional margin**, it's just a testing function that tell us whether the point is properly classified or not. The smallest $γ$ is the largest margin.

Scalling functional margin by $||w||$, we got **geometrical margin** $γ = y_i(\frac{w}{||w||}x_i+\frac{b}{||w||})$. to the hyperplane $γ$ can be write as:


 The geometric margin is just a scaled version of the functional margin.

You can think the functional margin, just as a testing function that will tell you whether a particular point is properly classified or not. And the geometric margin is functional margin scaled by ||w||

If you check the formula:

enter image description here

You can notice that independently of the label, the result would be positive for properly classified points (e.g sig(1*5)=1 and sig(-1*-5)=1) and negative otherwise. If you scale that by ||w|| then you will have the geometric margin.

Why does the geometric margin exists?

Well to maximize the margin you need more that just the sign, you need to have a notion of magnitude, the functional margin would give you a number but without a reference you can't tell if the point is actually far away or close to the decision plane. The geometric margin is telling you not only if the point is properly classified or not, but the magnitude of that distance in term of units of |w|



Let’s explain the algorithm briefly. 


## Non-Linear SVM


[跳过废话，直接看技术实现 ](#build) 


<p id = "build"></p>

 

正好之前就有关注过 [GitHub Pages](https://pages.github.com/) + [Jekyll](http://jekyllrb.com/) 快速 Building Blog 的技术方案，非常轻松时尚。

其优点非常明显：

* **Markdown** 带来的优雅写作体验
* 非常熟悉的 Git workflow ，**Git Commit 即 Blog Post**
* 利用 GitHub Pages 的域名和免费无限空间，不用自己折腾主机
	* 如果需要自定义域名，也只需要简单改改 DNS 加个 CNAME 就好了 
* Jekyll 的自定制非常容易，基本就是个模版引擎


本来觉得最大的缺点可能是 GitHub 在国内访问起来太慢，所以第二天一起床就到 GitCafe(Chinese GitHub Copy，现在被 Coding 收购了) 迁移了一个[镜像](http://huxpro.coding.me)出来，结果还是巨慢。

哥哥可是个前端好嘛！ 果断开 Chrome DevTool 查了下网络请求，原来是 **pending 在了 Google Fonts** 上，页面渲染一直被阻塞到请求超时为止，难怪这么慢。  
忍痛割爱，只好把 Web Fonts 去了（反正超时看到的也只能是 fallback ），果然一下就正常了，而且 GitHub 和 GitCafe 对比并没有感受到明显的速度差异，虽然 github 的 ping 值明显要高一些，达到了 300ms，于是用 DNSPOD 优化了一下速度。


---

配置的过程中也没遇到什么坑，基本就是 Git 的流程，相当顺手

大的 Jekyll 主题上直接 fork 了 Clean Blog（这个主题也相当有名，就不多赘述了。唯一的缺点大概就是没有标签支持，于是我给它补上了。）

本地调试环境需要 `gem install jekyll`，结果 rubygem 的源居然被墙了……后来手动改成了我大淘宝的镜像源才成功

Theme 的 CSS 是基于 Bootstrap 定制的，看得不爽的地方直接在 Less 里改就好了（平时更习惯 SCSS 些），**不过其实我一直觉得 Bootstrap 在移动端的体验做得相当一般，比我在淘宝参与的团队 CSS 框架差多了……**所以为了体验，也补了不少 CSS 进去

最后就进入了耗时反而最长的**做图、写字**阶段，也算是进入了**写博客**的正轨，因为是类似 Hack Day 的方式去搭这个站的，所以折腾折腾着大半夜就过去了。

第二天考虑中文字体的渲染，fork 了 [Type is Beautiful](http://www.typeisbeautiful.com/) 的 `font` CSS，调整了字号，适配了 Win 的渣渲染，中英文混排效果好多了。



—— Jennycs005 @ 10102020


