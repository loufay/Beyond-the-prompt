
# Beyond the Prompt: Deploying Medical Foundation Models on Diverse Chest X-ray Populations

## Overview


<p align="center">
  <a href="URL_OF_THE_WORK" target="_blank">
    <img src="figures/Fig1_Framework.png" alt="Framework" width="60%">
  </a>
</p>


Foundation models (FMs) have shown impressive performance in medical image analysis tasks, but their deployment in real-world clinical settings, especially across diverse patient populations such as adult and pediatric cases, remains challenging. Key open questions include optimal prompting techniques and strategies for model adaptation or fine-tuning for clinical use. In this study, we evaluated different approaches for deploying FMs in clinical scenarios for diverse patient populations. We use the lightweight, embedding-based vision-language FM \textit{MedImageInsight} to predict pneumonia from chest X-rays, a condition common in both adult and pediatric patients.
We observed large variation in model predictive performance depending on the chosen prompt design, highlighting the importance of text prompt design for successful zero-shot (ZS) application. On in-domain datasets, we found performance differences of up to 15\% in AUC across different text prompts. By introducing text and vision embedding ensembles, we achieved substantial ZS improvements, outperforming fine-tuning´with LoRA in low-data scenarios by up to 10\% for adults and pediatric populations (AUC).  

## Foundation Model
### MedImageInsight (MI2)
More information about [MedImageInsight](https://arxiv.org/abs/2410.06542).

Running MedImageInsight [locally](https://huggingface.co/lion-ai/MedImageInsights).

## Datasets

### MIMIC-CXR (in-domain, adult dataset)
### CheXpert (out-of-domain, adult dataset)
### VinDR-PCXR (out-of-domain, pediatric dataset)

## Comparing FM
* CheXagent
* RAD-DINO
* Biomed-CLIP





