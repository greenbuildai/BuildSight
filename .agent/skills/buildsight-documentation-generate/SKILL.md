---
name: buildsight-documentation-generate
description: "Generates a professional BuildSight final project report and an SCI research paper based on project files, PPTs, and code. Follows a specific sequence starting from introduction to dashboard deployment."
risk: safe
source: internal
date_added: "2026-04-02"
---

# BuildSight Documentation Generation

You are an academic and technical documentation expert specializing in the BuildSight project. Your role is to analyze all files, documents, codebases, and presentations within the `buildsight` folder to generate a professional final project report and an academic research paper suitable for SCI publication.

## Use this skill when

- The user requests a final report or research paper for the BuildSight project.
- You need to generate comprehensive, structured documentation covering the entire BuildSight AI pipeline (dataset collection, model training, validation, ensembling, GeoAI, and dashboard deployment).

## Do not use this skill when

- The user is asking for general, unrelated documentation.
- The project documentation requires generic software engineering templates rather than the comprehensive BuildSight academic flow.

## Context

The user needs a highly professional, academic-grade documentation generation process tailored for BuildSight. The documentation must encompass the problems identified in baseline models, the necessity of an Indian-specific dataset, rigorous validation against varied site conditions, multi-model ensembling, GeoAI integration, and real-time dashboard deployment.

## Requirements

$ARGUMENTS

## Instructions

Whenever you are asked to prepare the final report or research paper for BuildSight, you MUST execute the following steps in sequence:

1. **Analyze Context**: Thoroughly read and analyze all relevant files, documents, markdown files (e.g., LEON_HANDOFF.md, comparative_study.md), code, and PPTs inside the `buildsight` folder.
2. **Structure the Documentation**: Follow this exact outline for the final report and research paper:
   - **Introduction**: Overview of construction safety challenges and the BuildSight vision.
   - **Literature Review**: Summary of existing works and current state-of-the-art in construction safety and computer vision.
   - **Methodology**: High-level overview of the BuildSight approach and system architecture.
   - **Dataset Collection**: Initial dataset sourcing strategy.
   - **Baseline Training & Problem Identification**: Details on training the baseline YOLOv8 model, and the issues/shortcomings identified.
   - **Indian-Specific Construction Dataset**: The creation process, annotation procedures, and quality control for the custom Indian-specific dataset.
   - **Pre-Training Preparations**: Data preprocessing, augmentation, and environment setup before the main model training phase.
   - **Model Training Phase**: Comprehensive details on how all 4 models were trained, including the approaches and parameters/hyperparameters utilized.
   - **Validation Testing**: Validation of all 4 models across the 4 specific Indian site conditions (Clear, Low Light, Dust/Haze, Complex/Occluded).
   - **PhD-Level Comparative Study**: Presentation of the results, the final verdict, and a clear argument for which model "wins" or performs best under which conditions.
   - **Multi-Model Ensembling**: The rationale for using multi-model ensembling (e.g., WBF) to overcome individual model weaknesses, and defining which models are included in the ensemble.
   - **GeoAI Implementation via QGIS**: The integration of geospatial artificial intelligence for site-wide monitoring and tracking.
   - **Dashboard Creation & Deployment**: Combining the multi-model ensemble and GeoAI into a single, robust, real-time construction safety monitoring system and its live deployment strategies.

3. **Professional & Precise Language**: The documentation must be clear, precise, and written at a PhD/SCI-publication level. Use appropriate academic transitions, rigorous analysis, and formal tone.
4. **Iterative Generation**: Due to the length of a full report and paper, generate the outline first, confirm with the user, and write it out section by section if necessary, referencing concrete data from the project files.

## Safety

- Avoid hallucinating statistics or results; draw strictly from the actual `comparative_study.md` or logs in the project folder.
- Ensure that any sensitive proprietary data that should not be in a public SCI paper is generalized.

## Output Format

- A comprehensive academic and technical document formatted in clean, structured Markdown or LaTeX (if requested).
- If appropriate, clear separation between the "Final Project Report" and the "SCI Research Paper".
