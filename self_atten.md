# AI-generated image detector——Mz DOU

## Background

**要点**: 
 - 研究AI-generated image detector的原因
 - 当前AGI的常用模型

**原文**: 
    With the rapid advancement of artificial intelligence, generative AI has been widely applied across various domains. Currently, mainstream AI image generation techniques primarily include Generative Adversarial Networks (GANs, e.g., StyleGAN), diffusion models (e.g., Stable Diffusion), and Transformer-based models (e.g., DALL·E)¹. These technologies are capable of producing highly realistic and creative visual content, significantly advancing fields such as digital art, advertising, and media production. However, this technological progress also brings serious ethical and security challenges. The proliferation of deepfakes, synthetic identity fabrication, and misleading visual content is intensifying risks such as online fraud, misinformation dissemination, reputational harm, and political manipulation². Therefore, the development of efficient and robust methods for detecting AI-generated images has become an urgent necessity to preserve digital content authenticity and uphold. societal trust.
	
**译文**: 
随着人工智能技术的飞速发展，生成式AI被广泛应用于各个领域。目前，主流的AI图像生成技术主要包括以下几类：生成对抗网络（StyleGAN等）、扩散模型（Stable Diffusion）和基于Transformer的模型（DALL·E）¹。这些技术能够生成高度逼真且富有创意的视觉内容，极大地推动了数字艺术、广告设计和媒体制作的发展。然而，这种技术的进步也带来了严重的伦理与安全挑战。深度伪造（Deepfake）、虚假身份生成和误导性视觉内容的泛滥，正在加剧网络欺诈、虚假信息传播、名誉侵害和政治操纵等风险²。因此，开发高效、鲁棒的AI生成图像检测方法已成为维护数字内容真实性、保障社会信任体系的迫切需求。

**References**:
[1]A. Heidari et al., "Deepfake detection using deep learning methods: A systematic and comprehensive review," *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, vol. 14, no. 2, p. e1520, 2024.  

[2]D. Park, H. Na, and D. Choi, "Performance comparison and visualization of ai-generated-image detection methods," *IEEE Access*, vol. 12, pp. 62609–62627, 2024.

> 非最终版本，可修改


## Methodology——Self attention

**原文**：
The Vision Transformer (ViT), renowned for its powerful global modeling capability, has gained widespread recognition in AI-generated image detection due to its high accuracy and robustness. Various ViT variants—such as those trained on different datasets, DeiT (designed for small-sample scenarios), Swin Transformer, and CvT—are fundamentally built upon the original ViT architecture with structural modifications. Given the high cost and complexity of training from scratch, and considering the availability of well-established pretrained models, this study adopts the ViT-B/16 model pretrained on ImageNet-21k as the backbone network. Input images are resized to 224×224 RGB format, and the model leverages self-attention mechanisms to capture long-range dependencies across image patches. A single fully connected layer serves as the classification head, mapping the [CLS] token output to a binary classification space. During fine-tuning, the backbone is frozen while only the classification head is trained. Should performance prove insufficient, additional feature extraction layers may be introduced atop the backbone for further refinement.


**译文**:
具有强大全局建模能力的Vision Transformer目前在AGI detector中以较高的准确率和鲁棒性得到广泛认可。各版本ViT（不同训练集）， DeiT（小样本），Swin Transformer，CvT，基本都是以ViT为基础做出变动。由于重新训练成本高难度大，加之上述均有可调用预训练版本，本研究拟采用在 ImageNet-21k 上预训练的ViT-B16 作为主干网络。输入图像统一调整至RGB 224×224，模型通过自注意力机制建模全局依赖。分类头由一个全连接层构成，将输出映射至二分类空间。微调时冻结主干仅训练分类头。若如效果不佳，则尝试在主干网络之上增加特征提取层。

**调研结果**：
基本都写在译文里了，我认为可行的优化只有对图像做特征提取（滤波，卷积等）或者优化分类头（叠mlp，改激活函数），修改结构太难了不考虑。目前论文使用的也就是这些。还有一个思路是改变训练任务，比如从二分类变成多分类（ai图由什么模型生成），似乎可以增强性能。
这个帖子提供了更多可选的vit种类https://github.com/lucidrains/vit-pytorch

