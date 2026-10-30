1
Foundations of Generative AI (ICT4403)
Assignment 3
1)
Select a suitable source language and target language of your choice and obtain a publicly available parallel-text dataset for the selected language pair. Develop machine-translation systems using the following models:
•
Basic Transformer — encoder–decoder architecture
•
mT5 — encoder–decoder architecture
•
LLaMA — decoder-only architecture
•
Gemma — decoder-only architecture
Suitable smaller model variants and a manageable subset of the dataset may be used according to the available computational resources.
i.
Describe and preprocess the selected parallel-text dataset and divide it into appropriate training, validation, and test sets. Implement and train the basic Transformer, fine-tune mT5, and apply suitable fine-tuning or prompting methods to LLaMA and Gemma. Generate translations for the same test set using all four models under comparable settings.
ii.
Evaluate the four models using BLEU score, chrF score, training or fine-tuning time where applicable, inference time, and number of model parameters. Compare the performance and computational requirements of the encoder–decoder and decoder-only architectures based on the obtained results, while considering differences in model size, pretraining, tokenization, and adaptation method.
iii.
Examine the translations produced by the four models and discuss responsible and ethical AI concerns such as bias, hallucinated information, cultural misrepresentation, and inappropriate translation of sensitive content. Suggest suitable measures to reduce these concerns and ensure the responsible use of machine-translation systems.
Submission Procedure
Step 1: Complete the Assignment
•
Complete all the implementation, evaluation, comparison, and analysis as specified in the question.
•
Ensure that the program executes correctly and produces the reported results.
Step 2: Upload Your Work to GitHub
1.
Create a GitHub repository with a clear and relevant name. If it is private, provide access to the evaluator.
2.
Upload the implementation as a Jupyter Notebook (.ipynb) or Python source files (.py), along with the required supporting files, results, plots, and generated-text samples.
3.
Provide the correct source link for the textual dataset used.
2
4.
Include a README.md file containing:
o
problem statement and objective,
o
dataset description and source link,
o
brief description of the models implemented,
o
execution steps and required dependencies,
o
results obtained,
o
model comparison and evaluation,
o
sampling-temperature analysis, and
o
conclusion identifying the best-performing model.
5.
Organize the repository properly and ensure that all files and links are accessible.
Step 3: Share the GitHub Repository Link
1.
Create a Word document named: YourName_RegNum_FGAI_Assignment3.docx
2.
Include the following details:
o
student name,
o
registration number,
o
assignment title,
o
title of the developed autoregressive text-generation system,
o
name and link of the dataset used, and
o
GitHub repository link.
Step 4: Convert to PDF
•
Convert the Word document to PDF format using File → Save As → PDF.
Step 5: Upload to LMS
•
Upload the final PDF file to the specified assignment submission section in the Learning Management System (LMS) before the deadline.