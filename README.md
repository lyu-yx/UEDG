#  UEDG:Uncertainty-Edge Dual Guided Camouflage Object Detection

Full version of Code will release once this paper accepted. While the testing results has been released.
## 1. Features

<p align="center">
    <img src="assest/features.png"/> <br/>
    <em> 
    Figure 1: In this paper, we present the visualization results of the edge and uncertainty guidance operation in a highly challenging scenario. The red boxes indicate regions where the indistinguishable parts yield higher uncertainty scores. By incorporating edge information, our UEDG (Uncertainty Edge Guidance) approach achieves favorable performance.
    </em>
</p>

- **Novel multi-task guided framework.** We propose a novel structure that can combine multiple prior (uncertainty and edge in UEDG) for backbone feature guidance.

- **Powerful feature fusion strategy.** We employed Uncertainty-Edge
Mutual Fusion (UEMF), Uncertainty Deduce Module (UDM), Edge Estimate Module (EEM), and  Uncertainty/Edge Guide Grouping (UGG/EGG) module with in a powerful end-to-end formation.

- **SOTA results.** Our proposed method achieve the SOTA performance under four metrics in CHAMELEON, CAMO, COD10K, and NC4K. We also achieve the best performance in medical application like polyb segmentation as well.


## 2. News
[2023-07-03] Paper has been accepted by IEEE Transaction on Multimedia. :partying_face: Congradulations!!! :partying_face:  
[2023-05-27] Detection results on four dataset: CHAMELEON, CAMO, COD10K-test, and NC4K are avilible: [Google Drive](https://drive.google.com/drive/folders/1FVmgbhKsKE6eG8wb1gMcgq1nQMgJBqOB?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1tUQ41eoah9vMCPDg_Ytg7Q)(9cu6).  
[2023-05-26] Initial repository.  
[2022-11-22] Manuscript uploaded.


## 3. Overview
<p align="center">
    <img src="assest/overview.png"/> <br/>
    <em> 
    Figure 2: UEDG strucuture overview.
    </em>
</p>

<p align="center">
    <img src="assest/qualitative results.png"/> <br/>
    <em> 
    Figure 3: UEDG strucuture overview.
    </em>
</p>

<p align="center">
    <img src="assest/quantitative results.png"/> <br/>

</p>

## 4. Thanks
Code copied a lot from  [thograce/BGNet](https://github.com/thograce/BGNet.git), [HUuxiaobin/HitNet](https://github.com/HUuxiaobin/HitNet.git), [GewelsJI/DGNet](https://github.com/GewelsJI/DGNet.git), [clelouch/BgNet](https://github.com/clelouch/BgNet.git), [fanyang587/UGTR](https://github.com/fanyang587/UGTR.git), [whai362/PVT](https://github.com/whai362/PVT.git). Thanks for their great works!
