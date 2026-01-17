# Rad-Flamingo: A Multimodal Prompt-Driven Radiology Report Generation Framework with Patient-Centric Explanations

This repository contains the official implementation of **Rad-Flamingo**, accepted at **EACL 2026 (Findings)**. Rad-Flamingo is a multimodal framework for radiology report generation that integrates medical images with carefully designed prompt templates to produce clinically grounded reports along with patient-centric explanations.

---

## Overview

Rad-Flamingo is designed to improve both the **clinical utility** and **understandability** of automatically generated radiology reports. The framework operates in two inference stages and leverages prompt-driven generation.

---

## Implementation Details

### Fine-tuning

We follow the same training pipeline and experimental setup introduced in **XrayGPT**. Please refer to the original repository for dataset preparation, preprocessing, and training configurations:

- XrayGPT: https://github.com/mbzuai-oryx/XrayGPT

### Stage I Inference

Stage I inference generates patient-centric explanations based on a given xray and a report. This stage serves as a data augmentation technique to complement the scarcity of datasets containing patinet-centric explanation. The complete implementation for this stage is provided in this repository.

### Stage II Inference

Stage II inference produces radiology reports with patient-centric explanations. This stage leverages the multimodal few-shot prompting abilities of the model utilizing the augmented data from stage-I. The full inference code for this stage is also included in this repository.

---

## Prompt Templates

- All prompt templates used in the repository are **similar to those reported in the paper**.

Refer to Appendix for detailed prompt templates for both the stages.

---

## Running Stage I Inference

```
python stage1_inference.py --cfg-path **.yaml --gpu-id 0 --projections-csv **.csv --reports-xlsx **.csv --image-root <path_to_images> --few-shot-file <path_to_prompt_file> --instruction "Instruction goes here" --output-file <path_to_output_file>
```

## Running Stage II Inference

```
python med_flamingo_inference.py --llama-path /path/to/llama --image-list <path_to_images> --prompt-file <path_to_prompt_file> --output-file <path_to_output_file>--max-new-tokens <number>
```

---

## Acknowledgement
This work is heavily inspired by the previous works of [XrayGPT](https://github.com/mbzuai-oryx/XrayGPT), [Med-Flamingo](https://github.com/snap-stanford/med-flamingo) and [FEB](https://github.com/allenai/feb)
