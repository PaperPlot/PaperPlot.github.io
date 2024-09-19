---
title: "MACE: Mass Concept Erasure in Diffusion Models"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["mace"]
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Desc Text."
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


# MACE: A Novel Framework for Mass Concept Erasure in Text-to-Image Diffusion Models

![MACE Overview](0_overview.png#center)

*Figure 1: Overview of the MACE framework*

## TL;DR

- MACE is a new framework for erasing multiple concepts from text-to-image diffusion models
- It achieves superior balance between generality and specificity compared to prior methods
- Key components: closed-form cross-attention refinement, concept-specific LoRA modules, and efficient multi-LoRA integration
- Outperforms existing methods on tasks like object, celebrity, explicit content, and artistic style erasure
- Enables safer and more controlled text-to-image generation by removing unwanted concepts

## Introduction

Text-to-image diffusion models have made remarkable progress in recent years, enabling high-quality image generation from natural language descriptions. However, their ability to generate almost any content also raises concerns about potential misuse, such as creating harmful, copyrighted, or explicit images. To address these issues, researchers have been exploring ways to selectively remove or "erase" certain concepts from these models.

In this blog post, we'll dive deep into a new paper titled "MACE: Mass Concept Erasure in Diffusion Models" by Shilin Lu et al. This work introduces an innovative framework called MACE (MAss Concept Erasure) that can effectively erase a large number of concepts from text-to-image diffusion models while maintaining a good balance between generality and specificity.

## Background and Motivation

Before we delve into the details of MACE, let's briefly discuss the context and motivation behind this research:

1. **Risks of uncontrolled generation**: Text-to-image models trained on web-scraped data can inadvertently generate inappropriate content, including copyrighted material, explicit images, or deepfakes.

2. **Limitations of existing approaches**: Current methods for concept erasure often struggle to balance generality (removing all variations of a concept) and specificity (preserving unrelated concepts).

3. **Scalability challenges**: Most existing techniques are limited to erasing a small number of concepts (typically fewer than five) simultaneously.

The authors of MACE identified three key issues that hinder the effectiveness of prior works:

1. Residual information in co-existing words
2. Impact on early denoising steps affecting specificity
3. Performance degradation when erasing multiple concepts

With these challenges in mind, let's explore how MACE addresses them.

## The MACE Framework

MACE is designed to erase a large number of concepts from pretrained text-to-image diffusion models. It takes two inputs:

1. A pretrained model
2. A set of target phrases expressing the concepts to be removed

The framework returns a finetuned model incapable of generating images depicting the targeted concepts. Let's break down the key components of MACE:

### 1. Closed-Form Cross-Attention Refinement

The first step in MACE is to remove the residual information of target concepts from co-existing words in the prompt. This is achieved through a closed-form refinement of the cross-attention modules.

![Closed-Form Refinement](1_close-form_v2.png#center)

*Figure 2: Illustration of closed-form cross-attention refinement*

The objective function for this refinement is:

$$
\min_{\mathbf{W}^{\prime}_k} \sum_{i=1}^n \| \mathbf{W}_k^{\prime} \cdot \mathbf{e}^f_i - \mathbf{W}_k \cdot \mathbf{e}_i^g  \|_2^2 + \lambda_1 \sum_{i=n+1}^{n+m} \| \mathbf{W}_k^{\prime} \cdot \mathbf{e}^p_i - \mathbf{W}_k \cdot \mathbf{e}_i^p  \|_2^2
$$

Where:
- $\mathbf{W}_k^{\prime}$ is the refined projection matrix
- $\mathbf{e}^f_i$ is the embedding of a word co-existing with the target phrase
- $\mathbf{e}_i^g$ is the embedding of that word when the target phrase is replaced with its super-category or a generic concept
- $\mathbf{e}_i^p$ is the embedding for preserving prior knowledge
- $\lambda_1$ is a hyperparameter

This optimization problem has a closed-form solution:

$$
\mathbf{W}_k^\prime = \left( \sum_{i=1}^n \mathbf{W}_k \cdot \mathbf{e}^g_i \cdot (\mathbf{e}_i^f)^\top + \lambda_1 \sum_{i=n+1}^{n+m} \mathbf{W}_k \cdot \mathbf{e}^p_i \cdot (\mathbf{e}_i^p)^\top \right) \cdot \left( \sum_{i=1}^n \mathbf{e}^f_i \cdot (\mathbf{e}^f_i)^\top + \lambda_1 \sum_{i=n+1}^{n+m} \mathbf{e}^p_i \cdot (\mathbf{e}_i^p)^\top \right)^{-1}
$$

This refinement encourages the model to refrain from embedding residual information of the target phrase into other words.

### 2. Target Concept Erasure with LoRA

After removing residual information, MACE focuses on erasing the intrinsic information within the target phrase itself. This is achieved using Low-Rank Adaptation (LoRA) modules.

![LoRA Training](2_new_lora1.png#center)

*Figure 3: Training process with LoRA to erase intrinsic information*

The loss function for this step is designed to suppress the activation in certain regions of the attention maps corresponding to the target phrase tokens:

$$
\min \sum_{i \in S} \sum_{l}\| \mathbf{A}_{t,l}^i \odot \mathbf{M} \|^2_F
$$

Where:
- $S$ is the set of indices corresponding to the tokens of the target phrase
- $\mathbf{A}_{t,l}^i$ is the attention map of token $i$ at layer $l$ and timestep $t$
- $\mathbf{M}$ is the segmentation mask
- $\|\cdot\|_F$ is the Frobenius norm

The LoRA decomposition is applied to the weight modulations:

$$
\hat{\mathbf{W}}_k = \mathbf{W}_k^{\prime} + \Delta \mathbf{W}_k = \mathbf{W}_k^{\prime} + \mathbf{B} \times \mathbf{D}
$$

Where $\mathbf{B} \in \mathbb{R}^{d_\text{in} \times r}$ and $\mathbf{D} \in \mathbb{R}^{r \times d_\text{out}}$ are learned matrices, and $r \ll \min(d_\text{in}, d_\text{out})$ is the decomposition rank.

### 3. Concept-Focal Importance Sampling (CFIS)

To maintain specificity and focus on erasing particular concepts without affecting related ones, MACE introduces Concept-Focal Importance Sampling (CFIS). This technique assigns greater probability to smaller values of $t$ when sampling timesteps during LoRA training.

The probability density function for sampling $t$ is defined as:

$$
\xi(t) = \frac{1}{Z} \left( \sigma\left( \gamma(t-t_1) \right) - \sigma\left( \gamma(t-t_2) \right) \right)
$$

Where:
- \(Z\) is a normalizer
- $\sigma(x)$ is the sigmoid function
- $t_1$ and $t_2$ are the bounds of a high probability sampling interval
- $\gamma$ is a temperature hyperparameter

### 4. Fusion of Multi-LoRA Modules

To integrate multiple LoRA modules without mutual interference, MACE introduces a novel fusion technique. The objective function for this fusion is:

$$
\begin{aligned}
\min_{\mathbf{W}_k^*} &\sum_{i=1}^q \sum_{j=1}^p \|  \mathbf{W}_k^* \cdot \mathbf{e}_j^f  - (\mathbf{W}_{k}^\prime+ \Delta \mathbf{W}_{k,i}) \cdot \mathbf{e}^f_j \|_2^2 \\
& +  \lambda_2 \sum_{j=p+1}^{p+m} \| \mathbf{W}_k^* \cdot \mathbf{e}_j^p - \mathbf{W}_k \cdot \mathbf{e}^p_j  \|_2^2
\end{aligned}
$$

Where:
- $\mathbf{W}_{k}^\prime$ is the closed-form refined weight
- $\Delta \mathbf{W}_{k,i}$ is the LoRA module associated with the $i$-th concept
- $q$ is the number of erased concepts
- $\lambda_2$ is a hyperparameter

This fusion technique preserves the capability of individual LoRA modules while preventing interference between them.

## Experimental Results

The authors conducted extensive evaluations of MACE against prior methods across four different tasks:

1. Object erasure
2. Celebrity erasure
3. Explicit content erasure
4. Artistic style erasure

Let's look at some key results from these experiments.

### Object Erasure

For object erasure, the authors used the CIFAR-10 dataset and evaluated the methods on erasing each of the 10 object classes. They measured efficacy, specificity, and generality using CLIP classification accuracies.

![Object Erasure Results](3_teaser1.png#center)

*Figure 4: Example results of object erasure*

The results showed that MACE achieved the highest harmonic mean across the erasure of nine object classes, demonstrating superior erasure capabilities and an effective balance between specificity and generality.

### Celebrity Erasure

The celebrity erasure task involved erasing 1, 5, 10, and 100 celebrities from a dataset of 200 recognizable celebrities. The authors evaluated efficacy using the GIPHY Celebrity Detector (GCD) and measured specificity on retained celebrities and regular content.

![Celebrity Erasure Results](4_cele_results.png#center)

*Figure 5: Evaluation results for celebrity erasure*

MACE showed a notable enhancement in overall erasure performance, particularly when erasing 100 concepts. It maintained a good balance between efficacy and specificity, outperforming other methods.

### Explicit Content Erasure

For explicit content erasure, the authors finetuned Stable Diffusion v1.4 to erase four target phrases: 'nudity', 'naked', 'erotic', and 'sexual'. They evaluated the methods using the NudeNet detector on the Inappropriate Image Prompt (I2P) dataset.

| Method | Total Detected | FID | CLIP Score |
|--------|----------------|-----|------------|
| MACE   | 111            | 13.42 | 29.41     |
| SD v1.4 (original) | 743 | 14.04 | 31.34 |

MACE significantly reduced the amount of explicit content generated while maintaining good FID scores on regular content.

### Artistic Style Erasure

The authors evaluated MACE on erasing 100 artistic styles from a dataset of 200 artists. They used CLIP scores to measure efficacy and specificity.

| Method | CLIP_e ↓ | CLIP_s ↑ | H_a ↑ | FID-30K ↓ | CLIP-30K ↑ |
|--------|----------|----------|-------|-----------|------------|
| MACE   | 22.59    | 28.58    | 5.99  | 12.71     | 29.51      |
| SD v1.4 (original) | 29.63 | 28.90 | - | 14.04 | 31.34 |

MACE demonstrated superior ability to erase artistic styles on a large scale while maintaining good performance on regular content generation.

## Key Takeaways and Future Directions

The MACE framework introduces several innovative techniques that address the limitations of previous concept erasure methods:

1. **Closed-form cross-attention refinement**: Effectively removes residual information of target concepts from co-existing words.
2. **Concept-specific LoRA modules**: Enable targeted erasure of intrinsic information for each concept.
3. **Concept-Focal Importance Sampling**: Maintains specificity by focusing on later denoising steps.
4. **Efficient multi-LoRA integration**: Allows for the erasure of a large number of concepts without mutual interference.

These components collectively enable MACE to achieve superior performance in mass concept erasure while maintaining a good balance between generality and specificity.

Future research directions based on this work could include:

1. Scaling up the erasure scope to handle even more concepts (e.g., thousands)
2. Investigating the impact of concept erasure on downstream tasks and model capabilities
3. Exploring ways to further improve the specificity of erasure, especially for closely related concepts
4. Adapting MACE to other types of generative models beyond text-to-image diffusion models

The MACE framework represents a significant step forward in making text-to-image models safer and more controllable. By enabling efficient removal of unwanted concepts, it paves the way for more responsible deployment of these powerful generative models in real-world applications.

## Implementation Details

For those interested in implementing or experimenting with MACE, the authors have made their code available on GitHub: [https://github.com/Shilin-LU/MACE](https://github.com/Shilin-LU/MACE)

The framework is implemented using PyTorch and builds upon the Stable Diffusion v1.4 model. Key hyperparameters and training configurations for different erasure tasks are provided in the paper's appendix.

## Conclusion

MACE represents a significant advancement in the field of concept erasure for text-to-image diffusion models. By addressing the limitations of previous methods and introducing novel techniques, it enables the effective removal of a large number of concepts while maintaining model performance on unrelated tasks.

As generative AI continues to evolve and become more powerful, frameworks like MACE will play a crucial role in ensuring these technologies can be deployed safely and responsibly. By providing fine-grained control over model outputs, MACE and similar approaches can help mitigate risks associated with uncontrolled generation while preserving the creative potential of text-to-image models.

## References

1. Lu, S., Wang, Z., Li, L., Liu, Y., & Kong, A. W. (2024). MACE: Mass Concept Erasure in Diffusion Models. CVPR 2024.

2. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10684-10695).

3. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

5. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (pp. 8748-8763). PMLR.

6. Schramowski, P., Tuggener, L., Jentzsch, S., Bogun, I., & Kersting, K. (2023). Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 23744-23754).

7. Gandikota, R., Bansal, A., Gimpel, K., & Livescu, K. (2023). Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 16764-16775).

8. Zhang, Y., Ling, H., Gao, J., Yin, K., Lafleche, J. F., Barriuso, A., ... & Torralba, A. (2023). Forget-me-not: Learning to forget in text-to-image diffusion models. arXiv preprint arXiv:2303.17591.

9. Kumari, N., Zhang, R., Shechtman, E., & Efros, A. A. (2023). Ablating concepts in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1059-1069).

10. Heng, L., Choi, J., Lee, J. Y., & Kweon, I. S. (2023). Selective amnesia: A continual learning approach for forgetting in deep generative models. arXiv preprint arXiv:2305.10120.

---