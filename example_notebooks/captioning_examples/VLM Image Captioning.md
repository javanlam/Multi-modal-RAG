# VLM Image Captioning

## Key points to examine:

- Extracting images from source documents
- Quality and coherence of final captions generated for each image
- Time used

## Method:

1. Document input
2. Preprocessing (extract images AND context from source document)
3. VLM to provide caption for images

## Example documents used:

| Use Case | Document |
| --- | --- |
| Specific figures (mathematics) | [5.1_ Approximating Areas - Mathematics LibreTexts.pdf](vlm_results/5%201_%20Approximating%20Areas%20-%20Mathematics%20LibreTexts%20%202b6548de34b481648af6e261a9ee7fe0.md) |
| Concept Images (depicts software features) | [Apple Intelligence features are now available in traditional Chinese - Apple (HK).pdf](vlm_results/Apple%20Intelligence%20features%20are%20now%20available%20in%20t%202b6548de34b481638d84f670b8d0c8af.md) |
| Informational Images | [Transport Arrangement for Fall Term _ Campus Services Office.pdf](vlm_results/Transport%20Arrangement%20for%20Fall%20Term%20_%20Campus%20Servi%202b6548de34b48173b9dddfcf71968bad.md) |
| Advanced concepts with context from text only | [Underwater SLAM for Marine Robot Navigation _ HKUST CSE.pdf](vlm_results/Underwater%20SLAM%20for%20Marine%20Robot%20Navigation%20_%20HKUS%202b6548de34b48167ac00dfb8a29b79c9.md) |


## Statistics on time used for each image:

- Mean: 3.8861 s
- Variance: 2.9140 s$^2$
- Maximum: 10.1240 s
- Minimum: 1.7750 s


## Accuracy Results:

### Scoring Rules
- 0: completely out-of-context.
- 0.25: partially correct, but brings in incorrect context.
- 0.5: partially correct. does not connect the correct part with the incorrect part as context.
- 0.75: largely correct. little incorrect context is correct.
- 1: completely correct.

### Scores per Document
    
#### Document 1
0.984375

| Scores | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.75 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

#### Document 2
0.979167

| Scores | 0.75 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

#### Document 3
0.9375

| Scores | 1 | 1 | 1 | 0.75 |
| --- | --- | --- | --- | --- |

#### Document 4
1

| Scores | 1 | 1 |
| --- | --- | --- |

Overall statistics: 0.977941

## Problems:

1. The mean amount of time used for each image is around 1 second longer than the previous method. Preprocessing long documents may take up a large amount of time.

## Evaluation:

1. The accuracy is significantly higher than the previous method; only few captions with hallucination effects were seen. This is a much more viable method for captioning.