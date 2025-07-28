# Efficient Multi-robot Task and Path Planning in Large-Scale Cluttered Environments

Description
-----

We present the implementation of an efficient multi-robot task and path planning (**MRTPP**) method for multi-robot coordination in large-scale cluttered environments. The source code of our method, along with the compared state-of-the-art (SOTA) solvers, is implemented in Python and publicly available here. 

The main contributions are summarized as follows: 1) A fast path planner suitable for large-scale and cluttered workspaces that efficiently constructs the cost matrix of collision-free paths between tasks and robots for solving the MRTPP problem. 2) An efficient auction-based method for solving the MRTPP problem by incorporating a novel memory-aware strategy, aiming to minimize the maximum travel cost for robots to visit tasks. 

About
-----

**Paper**: [Efficient Multi-robot Task and Path Planning in Large-Scale Cluttered Environments](https://ieeexplore.ieee.org/abstract/document/11091464)  

**Authors**: Gang Xu, Yuchen Wu, Sheng Tao, Yifan Yang, Tao Liu, Tao Huang, Huifeng Wu, and Yong Liu  

**Accepted to**: IEEE Robotics and Automation Letters (**RA-L**), 2025

**Code**: The source code will be released soon.

Experimental Results
-----

#### Evaluation of Path Planners

<p align="center">
    <img src="figures/fig2.png" width="1050" height="350" />
</p>

<p align="center">
    <img src="figures/fig3.png" width="1500" height="300" />
</p>

#### Comparisons in Large-scale Cluttered Environments

<p align="center">
    <img src="figures/table1.png" width="1200" height="450" />
</p>

<p align="center">
    <img src="figures/fig4.png" width="1200" height="320" />
</p>

Citation
-----

```
@article{xu2025efficient,
  author={Xu, Gang and Wu, Yuchen and Tao, Sheng and Yang, Yifan and Liu, Tao and Huang, Tao and Wu, Huifeng and Liu, Yong},
  journal={IEEE Robotics and Automation Letters}, 
  title={Efficient Multi-Robot Task and Path Planning in Large-Scale Cluttered Environments}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2025.3592146}
}
```

