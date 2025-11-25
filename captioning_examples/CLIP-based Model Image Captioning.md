# CLIP-based Model Image Captioning

Utilizes the implementation and pre-trained model from:

[“ClipCap: CLIP Prefix for Image Captioning” (arXiv:2111.09734)](https://www.notion.so/27d548de34b480019dc8d89439eaec43?pvs=21)

## Key points to examine:

- Extracting images from source documents
- Quality and coherence of final captions generated for each image
- Time used

## Method:

1. Document input
2. Preprocessing (extract images AND context from source document)
3. CLIP model for generating caption only based on the image
4. LLM to refine caption when given context from source document as well

## Example documents used:

| Use Case | Document |
| --- | --- |
| Specific figures (mathematics) | [5.1_ Approximating Areas - Mathematics LibreTexts.pdf](clip_results/5%201_%20Approximating%20Areas%20-%20Mathematics%20LibreTexts%20%202b6548de34b481648af6e261a9ee7fe0.md) |
| Concept Images (depicts software features) | [Apple Intelligence features are now available in traditional Chinese - Apple (HK).pdf](clip_results/Apple%20Intelligence%20features%20are%20now%20available%20in%20t%202b6548de34b481638d84f670b8d0c8af.md) |
| Informational Images | [Transport Arrangement for Fall Term _ Campus Services Office.pdf](clip_results/Transport%20Arrangement%20for%20Fall%20Term%20_%20Campus%20Servi%202b6548de34b48173b9dddfcf71968bad.md) |
| Advanced concepts with context from text only | [Underwater SLAM for Marine Robot Navigation _ HKUST CSE.pdf](clip_results/Underwater%20SLAM%20for%20Marine%20Robot%20Navigation%20_%20HKUS%202b6548de34b48167ac00dfb8a29b79c9.md) |


## Statistics on time used for each image:

- Mean: 2.7507 s
- Variance: 1.0171 s$^2$
- Maximum: 7.2277 s
- Minimum: 1.7325 s

## Accuracy Results:

### Scoring Rules
- 0: completely out-of-context.
- 0.25: partially correct, but brings in incorrect context.
- 0.5: partially correct. does not connect the correct part with the incorrect part as context.
- 0.75: largely correct. little incorrect context is correct.
- 1: completely correct.
    
### Scores per Document

#### Document 1
CLIP 0.03125, Enhanced 0.515625

| CLIP | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Enhanced | 1 | 0.75 | 0.75 | 0.5 | 0.5 | 0.5 | 0.25 | 0.5 | 0.25 | 0.5 | 0.5 | 0.25 | 1 | 0.5 | 0 | 0.5 |

#### Document 2
CLIP 0.02083, Enhanced 0.479167

| CLIP | 0.25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Enhanced | 0.75 | 0.5 | 0.5 | 0.25 | 0.5 | 0.5 | 0.5 | 0.5 | 1 | 0 | 0.5 | 0.25 |

#### Document 3
CLIP 0.1875, Enhanced 0.6875

| CLIP | 0.25 | 0 | 0 | 0.5 |
| --- | --- | --- | --- | --- |
| Enhanced | 1 | 0.75 | 0.25 | 0.75 |

#### Document 4
CLIP 0.25, Enhanced 0.875

| CLIP | 0.25 | 0.25 |
| --- | --- | --- |
| Enhanced | 1 | 0.75 |
- Overall statistics: CLIP 0.05882, Enhanced 0.544118

## Problems:

1. We can see that captioning image with this method takes 2 to 3 seconds in general, and it involves passing through an LLM. This turns out to be a slow and expensive method, especially when the source document has a long length, or consists of a large number of images.
2. Another issue is the content of captions. Notice that this implementation is rather old (from 2021) and utilizes GPT-2, a rather weak model in today’s standards. Out-of-context captions are frequently observed as a result. While enhancements with an LLM include also context from the source document in text blocks surrounding the concerned image; caption generation in this manner is also highly influenced by the caption from the CLIP-based model.

## Potential Workarounds:

- Perform captioning using a more powerful out-of-the-box Vision Language Model (VLM)
- Experiment with pretrained vision transformers as encoders and more powerful pretrained text decoders
- VLM / MLLM image captioning is investigated in part of “How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites” (arXiv:2404.16821)
- Image captioning with LLMs benchmark: "CapArena: Benchmarking and Analyzing Detailed Image Captioning in the LLM Era” (arXiv: 2505.12329)