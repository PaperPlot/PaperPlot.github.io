---
title: "BEVContrast: Self-Supervision in BEV Space for Automotive Lidar Point Clouds"
date: 2025-01-08 17:47:17.617253
# weight: 1
# aliases: ["/first"]
tags: ['self-supervised learning', '3D point clouds', 'autonomous driving', 'contrastive learning']
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: ""
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

# BEVContrast: Self-Supervision in BEV Space for Automotive Lidar Point Clouds

## TL;DR

- BEVContrast is a simple yet effective self-supervised learning method for 3D point cloud processing in autonomous driving scenarios.
- It uses contrastive learning on Bird's Eye View (BEV) representations of Lidar point clouds.
- The method outperforms state-of-the-art self-supervised approaches on semantic segmentation tasks and achieves competitive results on 3D object detection.
- BEVContrast is easier to implement and tune compared to more complex methods that rely on object segmentation.

## Introduction

Self-supervised learning has become increasingly important in the field of 3D point cloud processing, especially for autonomous driving applications. The high cost of annotating Lidar point clouds makes self-supervised methods particularly attractive, as they can leverage large amounts of unlabeled data to learn meaningful representations.

In this blog post, we'll dive deep into BEVContrast, a novel self-supervised learning method for 3D point clouds introduced by Corentin Sautier, Gilles Puy, Alexandre Boulch, Renaud Marlet, and Vincent Lepetit. This method, presented in their paper "BEVContrast: Self-Supervision in BEV Space for Automotive Lidar Point Clouds," offers a simple yet powerful approach to learning 3D representations without the need for expensive annotations.

## Background and Related Work

Before we delve into the details of BEVContrast, let's briefly review some of the existing self-supervised learning approaches for 3D point clouds:

1. **PointContrast** [1]: This method applies contrastive learning at the point level, contrasting features of corresponding points in different views of the same scene.

2. **SegContrast** [2] and **TARL** [3]: These approaches perform contrastive learning at the segment level, roughly corresponding to objects in the scene. They require a preprocessing step to segment the point cloud.

3. **ALSO** [4]: This method uses unsupervised surface reconstruction as a pretext task to train 3D backbones.

While PointContrast is simple to implement, it doesn't perform as well as segment-based methods. On the other hand, segment-based methods like SegContrast and TARL require complex preprocessing steps and have many hyperparameters to tune.

## BEVContrast: A New Approach

BEVContrast takes a different approach by performing contrastive learning on Bird's Eye View (BEV) representations of the point cloud. This method strikes a balance between the simplicity of PointContrast and the effectiveness of segment-based methods.

### Key Idea

The core idea of BEVContrast is to project point features onto a 2D BEV grid and perform contrastive learning on these grid cell representations. This approach is motivated by the observation that objects in urban scenes are naturally well-separated in the BEV plane.

### Method Details

Let's break down the BEVContrast method step by step:

1. **Input**: The method takes as input two partially overlapping Lidar scans $(\mathcal{P}_1, \mathcal{P}_2)$ captured from different viewpoints in the same scene. Each point in these scans is described by a 4D vector containing its 3D Cartesian coordinates and the measured return intensity.

2. **3D Backbone**: A 3D backbone network $f_\theta(\cdot)$ processes each point cloud, outputting a D-dimensional feature vector for each point.

3. **BEV Projection**: The point features are projected onto a 2D BEV grid. The features of points falling into the same grid cell are averaged, resulting in BEV representations $\mathcal{B}_1 = g(f_\theta(\mathcal{P}_1), \mathcal{P}_1)$ and $\mathcal{B}_2 = g(f_\theta(\mathcal{P}_2), \mathcal{P}_2)$, where $g(\cdot, \cdot)$ represents the projection and pooling step.

4. **Alignment**: The BEV representations are aligned using the known transformation between the two scans. This is done by applying a 2D affine transformation to $\mathcal{B}_2$, resulting in $\tilde{\mathcal{B}}'$.

5. **Contrastive Loss**: The method uses a contrastive loss to enforce similarity between corresponding cells in the aligned BEV representations. The loss is defined as:

   $$
   \mathcal{L}(\theta) = - \sum_{l \in \mathcal{N}} \log \left[  
   \frac
   {\exp\left({\tilde{\mathcal{B}}_{l}'} \cdot \mathcal{B}_{1l} / \tau\right)}
   {\sum_{m \in \mathcal{N}} \exp\left({\tilde{\mathcal{B}}_{l}' \cdot \mathcal{B}_{1m}} / \tau \right)}
   \right]
   $$

   where $\mathcal{N}$ is a set of randomly sampled non-empty BEV cells, $\tau$ is a temperature parameter, and $\cdot$ denotes the scalar product.

### Implementation Details

The authors provide several important implementation details:

- The BEV cell size $b_s$ and time difference $\Delta_{\text{time}}$ between scans are key hyperparameters.
- The method uses AdamW optimizer with a learning rate of 0.001 and weight decay of 0.001.
- The temperature $\tau$ is set to 0.07, following previous work.
- The number of samples $\mathcal{N}$ in the loss is set to 4096.

## Experimental Results

The authors conducted extensive experiments to evaluate BEVContrast on various datasets and tasks. Let's look at some of the key results:

### Semantic Segmentation

BEVContrast was evaluated on semantic segmentation tasks using the nuScenes and SemanticKITTI datasets. The method consistently outperformed other self-supervised approaches across different percentages of labeled data used for fine-tuning.

Here's a summary of the results on SemanticKITTI using 1% of the annotated scans for fine-tuning:

| Method | mIoU (%) |
|--------|----------|
| No pre-training | 46.2 ± 0.6 |
| PointContrast | 47.9 ± 0.5 |
| SegContrast | 48.9 ± 0.3 |
| STSSL | 49.4 ± 1.1 |
| ALSO | 50.0 ± 0.4 |
| TARL | 52.5 ± 0.5 |
| BEVContrast (ours) | 53.8 ± 1.0 |

As we can see, BEVContrast achieves the highest mean Intersection over Union (mIoU) score, outperforming even complex methods like TARL.

### 3D Object Detection

The authors also evaluated BEVContrast on 3D object detection tasks using the KITTI 3D dataset. They pre-trained the backbone of popular object detectors like SECOND and PV-RCNN using BEVContrast on nuScenes data.

Here are some results for the PV-RCNN backbone using the $R_{40}$ metric:

| Method | Car AP | Pedestrian AP | Cyclist AP | mAP | Diff. |
|--------|--------|---------------|------------|-----|-------|
| No pre-training | 84.5 | 57.1 | 70.1 | 70.6 | - |
| STRL | 84.7 | 57.8 | 71.9 | 71.5 | +0.9 |
| ALSO | 84.9 | 57.8 | 75.0 | 72.5 | +1.9 |
| BEVContrast (ours) | 84.8 | 57.3 | 74.2 | 72.1 | +1.5 |

While BEVContrast doesn't achieve the highest overall mAP, it still shows significant improvement over no pre-training and remains competitive with state-of-the-art methods.

## Analysis and Discussion

### Advantages of BEVContrast

1. **Simplicity**: BEVContrast is much simpler to implement compared to segment-based methods like TARL, which require complex preprocessing steps.

2. **Effectiveness**: Despite its simplicity, BEVContrast outperforms more complex methods on semantic segmentation tasks and remains competitive on object detection.

3. **Robustness**: The method shows good performance across different datasets and Lidar types, making it versatile for various autonomous driving scenarios.

4. **Efficiency**: BEVContrast doesn't require expensive preprocessing, making it faster to train and easier to apply to new datasets.

### Limitations and Future Work

1. **Sequential Data Requirement**: Like other multi-frame methods, BEVContrast requires sequential data, which may not always be available (e.g., KITTI detection benchmark).

2. **Object-level Representation**: While BEV cells provide a good proxy for object-level representations, they may not always align perfectly with actual objects. Future work could explore ways to improve this alignment without sacrificing simplicity.

3. **Performance on Object Detection**: While competitive, BEVContrast doesn't consistently outperform all methods on object detection tasks. Further investigation into why contrastive methods seem less effective for detection compared to segmentation could be valuable.

## Conclusion

BEVContrast presents a simple yet powerful approach to self-supervised learning for 3D point clouds in autonomous driving scenarios. By leveraging Bird's Eye View representations, it achieves state-of-the-art performance on semantic segmentation tasks while remaining competitive on object detection.

The method's simplicity and effectiveness make it an attractive option for researchers and practitioners working with 3D point cloud data. As self-supervised learning continues to advance, approaches like BEVContrast that balance simplicity and performance will likely play an increasingly important role in developing robust perception systems for autonomous vehicles.

For those interested in trying out BEVContrast, the authors have made their code available on GitHub: [https://github.com/valeoai/BEVContrast](https://github.com/valeoai/BEVContrast)

## References

[1] Xie, S., Gu, J., Wang, D., Fei, Z., & Feng, Y. (2020). PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding. In European Conference on Computer Vision (pp. 574-591). Springer, Cham.

[2] Jiang, L., Zhao, H., Liu, S., Shen, X., Fu, C. W., & Jia, J. (2021). SegContrast: 3D Point Cloud Feature Representation Learning through Self-supervised Segment Discrimination. arXiv preprint arXiv:2104.09340.

[3] Puy, G., Boulch, A., & Marlet, R. (2022). TARL: Temporally Aggregated Representations for Self-Supervised Learning on Automotive Lidar Point Clouds. arXiv preprint arXiv:2201.07250.

[4] Puy, G., Boulch, A., & Marlet, R. (2022). ALSO: Automotive Lidar Self-supervision by Occupancy estimation. arXiv preprint arXiv:2212.05867.

