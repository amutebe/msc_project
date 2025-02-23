**Periportal Fibrosis Detection using Machine Learning:**
This repository contains the code and documentation for my MSc Data Science at Essex University project which focused on developing a deep learning model for automated detection of periportal fibrosis in ultrasound images.

**Project Overview:**

Periportal fibrosis is a common consequence of chronic schistosomiasis infection, leading to significant liver damage. Early and accurate detection is crucial for effective treatment and management. This project explores the potential of machine learning to automate the identification of periportal fibrosis from ultrasound images, potentially improving diagnostic efficiency and accessibility.

**Methodology:**

The project follows a standard machine learning workflow, including:

**Data Collection:** A dataset of liver ultrasound images, annotated by an expert study sonographer using the Niamey protocol, was used for model training and evaluation. The dataset was acquired from a study conducted by the Uganda Schistosomiasis Multidisciplinary Research Centre (U-SMRC), a leading Ugandan institution in schistosomiasis research. The dataset originates from an adult case-control study investigating risk factors for severe schistosomal morbidity in communities near Lake Victoria and Lake Albert, two distinct epidemiological settings. The study individuals were between 18 and 50 years old. Ultrasound sonography, a common diagnostic technique, was performed using the Logiq e ultrasound system.

**Image Preprocessing:** Images were preprocessed to enhance features and standardize input for the model. This included resizing, normalization, and potentially noise reduction techniques.

**Feature Extraction:** Relevant features were extracted from the images automatically using deep learning methods. Features could include texture, shape, and intensity patterns associated with periportal fibrosis.

**Model Selection and Training:** Convolutional Neural Networks were explored and trained to extract features and corresponding labels.

**Model Evaluation:** The trained model was rigorously evaluated on a separate test dataset to assess the performance using metrics such as accuracy, precision, recall, and F1-score.

**Visualization and Interpretation:** Results were visualized using techniques like confusion matrices and ROC curves to understand model behavior and potential areas for improvement.

**Key Findings:**

This study highlights the potential of a deep learning-based model, in this case, **model 2** for accurately identifying periportal fibrosis in ultrasound images. The model's impressive **AUC** of 0.87, coupled with strong performance metrics including 80% test **accuracy**, 76% **precision**, 84% **recall**, and an 80% **F1 score**, underscores its ability to effectively discriminate between images with and without periportal fibrosis. 
This robust performance, particularly the balanced sensitivity (84%) and specificity (84%) suggests its potential for real-world clinical application, offering an earlier and more accurate diagnosis of periportal fibrosis.


**Code Structure:**

model_1.ipynb: Contains scripts for  model 1 test results.

model_2.ipynb: Contains scripts for  model 2 test results.


**Future Work**
This study presents a promising approach to automated periportal fibrosis detection using ultrasound images and deep learning. 
**Enhancing Model Capabilities and Scope:**
1.	**Enhancing Data Diversity and Quality:**
Expand the dataset to include a variety of ultrasound images, patient demographics, disease severities, and image variations.
2.	**Exploring Advanced Image Processing and Model Architectures:**
Experiment with deeper CNNs, transfer learning, and hybrid models.
3.	**Enhancing Interpretability and Validation:**
Validate the model with independent datasets from different sources.
4.	**Longitudinal Studies and Clinical Integration:**
Collect longitudinal data, including follow-up ultrasound images and clinical outcomes.
Integrate the model into clinical workflows for personalised patient management.
5.	**Comparative Analysis:**
While direct comparisons are challenging due to variations in datasets, methodologies, and evaluation metrics, contextualising our findings within the existing literature is essential. Several studies have explored automated periportal fibrosis detection using ultrasound images, employing various techniques and achieving varying levels of success. 


Contact
mutebe2@gmail.com

