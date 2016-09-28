---
layout: default
title: SparseReg
---


## SparseReg Toolbox for Matlab

SparseReg toolbox is a collection of Matlab functions for sparse regressions. 

The toolbox is developed by [Hua Zhou](http://hua-zhou.github.io), with contributions from [Brian Gaines](http://brgaines.github.io/).  Artin Armagan and David Dunson provided testing and invaluable feedback.

### Compatibility

The code is tested on Matlab R2014a, but should work on other versions of Matlab with no or little changes. Current version works on these platforms: Windows 64-bit, Linux 64-bit, and Mac (Intel 64-bit). Type `computer` in Matlab's command window to determine the platform.

### Download

[SparseReg_toolbox_0.0.2.zip](https://github.com/Hua-Zhou/SparseReg/archive/v0.0.2.zip) (1.2MB)

### Installation

1. Download the zip package.
2. Extract the zip file.  
```
unzip SparseReg_toolbox_0.0.2.zip
```
3. Rename the folder from *SparseReg_toolbox_0.0.2* to *SparseReg*.  
```
mv SparseReg_toolbox_0.0.2 SparseReg
```
4. Add the *SparseReg* folder to Matlab search path. Start Matlab, cd to the *SparseReg* directory, and execute the following commands  
`addpath(pwd)	%<-- Add the toolbox to the Matlab path`  
`save path		%<-- Save for future Matlab sessions`
5. Go through following tutorials for the usage. For help of individual functions, type `?` followed by the function name in Matlab.

### Tutorial

* [Sparse linear regression (enet, power, log, MC+, SCAD)](./html/demo_lsq.html)
* [Sparse generalized linear model (GLM) (enet, power, log, MC+, SCAD)](./html/demo_glm.html)

### Licensing

SparseReg Toolbox for Matlab is licensed under the [BSD](./html/COPYRIGHT.txt) license. Please use at your own risk.

### How to cite

If you use this toolbox in any way, please cite the software itself along with at least one publication or preprint.

* Software reference  
H Zhou. Matlab SparseReg Toolbox Version 0.0.2, Available online, September 2016.  
* H Zhou, A Armagan, and D Dunson (2012) Path following and empirical Bayes model selection for sparse regressions. \[[arXiv:1201.3528](http://arxiv.org/abs/1201.3528)\]
* Default article to cite for least squares + generalized lasso penalty  
H Zhou and K Lange (2013) A path algorithm for constrained estimation, [_Journal of Computational and Graphical Statistics_](http://amstat.tandfonline.com/doi/full/10.1080/10618600.2012.681248), 22(2):261-283.
* Default article to cite for convex loss + generalized lasso penalty  
H Zhou and Y Wu (2012)  A generic path algorithm for regularized statistical estimation, [Journal of American Statistical Association](http://www.tandfonline.com/doi/full/10.1080/01621459.2013.864166#.Up5KiGRDt4A), 109(506):686-699. [[pdf](http://hua-zhou.github.io/media/pdf/ZhouWu14EPSODE.pdf)]
* Default article to cite for path following in constrained convex programming  
H Zhou and K Lange (2011) Path following in the exact penalty method of convex programming, [Computational Optimization and Applications](http://link.springer.com/article/10.1007/s10589-015-9732-x), 61(3):609-634. [[pdf](http://hua-zhou.github.io/media/pdf/XiaoWuZhou15ConvexLAR.pdf)]
* Default article to cite for constrained lasso path following  
B Gaines and H Zhou (2016) On fitting the constrained lasso.

### Contacts

Hua Zhou <huazhou@ucla.edu> | Brian Gaines <brgaines@ncsu.edu> | Artin Armagan <Artin.Armagan@sas.com> | David Dunson <dunson@stat.duke.edu>
