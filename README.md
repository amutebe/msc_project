**Periportal Fibrosis Detection using Machine Learning**
This repository contains the code and documentation for my MSc project focused on developing a machine learning model for automated detection of periportal fibrosis in ultrasound images.

**Project Overview**

Periportal fibrosis is a common consequence of chronic schistosomiasis infection, leading to significant liver damage. Early and accurate detection is crucial for effective treatment and management. This project explores the potential of machine learning to automate the identification of periportal fibrosis from ultrasound images, potentially improving diagnostic efficiency and accessibility.

**Methodology**

The project follows a standard machine learning workflow, including:

**Data Collection**: A dataset of liver ultrasound images, annotated by expert radiologists, was used for training and evaluation. The dataset was acquired as part of a study conducted by the Uganda Schistosomiasis Multidisciplinary Research Centre (U-SMRC), a leading Ugandan institution in schistosomiasis research. The dataset originates from an adult case-control study investigating risk factors for severe schistosomal morbidity in communities near Lake Victoria and Lake Albert, two distinct epidemiological settings. The study individuals were aged between 18 to 50 years. Ultrasound sonography, a common diagnostic technique, was performed using the Logiq e ultrasound system.

**Image Preprocessing**: Images were preprocessed to enhance features and standardize input for the model. This included resizing, normalization, and potentially noise reduction techniques.
**Feature Extraction**: Relevant features were extracted from the images automatically using deep learning methods. Features could include texture, shape, and intensity patterns associated with periportal fibrosis.
**Model Selection and Training**: Convolutional Neural Networks were explored and trained to extract features and corresponding labels.
**Model Evaluation**: The trained model was rigorously evaluated on a separate test dataset to assess the performance using metrics such as accuracy, precision, recall, and F1-score.
**Visualization and Interpretation**: Results were visualized using techniques like confusion matrices and ROC curves to understand model behavior and potential areas for improvement.
**Key Findings**
This study highlights the potential of a deep learning-based model, in this case, **model 2** for accurately identifying periportal fibrosis in ultrasound images. The model's impressive **AUC** of 0.87, coupled with strong performance metrics including 80% test **accuracy**, 76% **precision**, 84% **recall**, and an 80% **F1 score**, underscores its ability to effectively discriminate between images with and without periportal fibrosis. 
This robust performance, particularly the balanced sensitivity (84%) and specificity (84%) suggests its potential for real-world clinical application, offering an earlier and more accurate diagnosis of periportal fibrosis.


**Code Structure**

model_1.ipynb: Contains scripts for data model 1 test results.
model_2.ipynb: Contains scripts for data model 2 test results.


**Future Work**
This study presents a promising approach to automated periportal fibrosis detection using ultrasound images and deep learning. 
**Enhancing Model Capabilities and Scope**
1.	**Enhancing Data Diversity and Quality**
A key area for improvement lies in expanding the dataset to encompass peer-reviewed ultrasound images, a wider range of patient demographics, disease severities, and ultrasound image variations. This would significantly enhance the model's robustness and generalisability to real-world clinical scenarios. Collaborating with other institutions or leveraging publicly available datasets could facilitate access to larger and more diverse data. Exploring advanced data augmentation techniques, such as synthetic image generation, could further enhance the training process and address potential biases.
2.	**Exploring Advanced Image Processing and Model Architectures**
Investigating advanced image preprocessing techniques like noise reduction, speckle filtering, and contrast enhancement could potentially improve image quality and lead to better feature extraction, ultimately improving model performance. Experimenting with alternative deep learning architectures, such as deeper CNNs, transfer learning with pre-trained models, or hybrid models combining CNNs with other techniques, could unlock further performance gains. Fine-tuning hyperparameters through techniques like grid search or Bayesian optimisation would optimise model configurations for enhanced accuracy and generalisation.
3.	**Enhancing Interpretability and Validation**
Improving the interpretability of the model's predictions using techniques like saliency maps or attention mechanisms is crucial for building trust and understanding among healthcare professionals. External validation on an independent dataset from a different source is essential to robustly assess the model's generalisability and real-world performance, ensuring its reliability in diverse clinical settings.
4.	**Longitudinal Studies and Clinical Integration**
Collecting longitudinal data, including follow-up ultrasound images and clinical outcomes, would enable evaluation of the model's ability to predict disease progression and treatment response over time. This would pave the way for integrating the model into clinical workflows for personalised patient management, potentially leading to more targeted and effective treatment strategies.
5.	**Comparative Analysis**
While direct comparisons are challenging due to variations in datasets, methodologies, and evaluation metrics, contextualising our findings within the existing literature is essential. Several studies have explored automated periportal fibrosis detection using ultrasound images, employing various techniques and achieving varying levels of success. One notable study by Lee et al. (2019) focused on the application of a deep convolutional neural network (DCNN) for automated classification of liver fibrosis using ultrasonography (US) images.
**Similarities**
a)	Objective
Both our study and the referenced study share the primary objective of developing an automated system for assessing liver fibrosis severity using ultrasound images. This highlights a growing interest in leveraging AI for non-invasive liver disease diagnosis.
b)	Use of Deep Learning 
Both studies employed deep learning, specifically CNNs, as the core technology for image analysis and classification. This demonstrates the increasing recognition of deep learning's power in medical image interpretation.
c)	Focus on Cirrhosis Detection
Both studies emphasised the detection of cirrhosis, a severe stage of liver fibrosis, as a key clinical outcome. This underscores the importance of early and accurate cirrhosis identification for effective patient management.
d)	Strong Performance
Both our model and the DCNN in the referenced study achieved high AUC values (0.87 in our study and 0.857 in the referenced study) for classifying cirrhosis, indicating strong discriminatory power. This suggests the potential of deep learning-based approaches for accurate fibrosis assessment.
**Differences**
a)	Target Population
 While our study focused on periportal fibrosis, the referenced study targeted a broader population of patients with chronic liver diseases, including hepatitis B and C. This difference in target populations could influence the generalisability of the models to different disease contexts.
b)	Dataset Characteristics
The referenced study employed a dataset of ultrasound images paired with histopathological results from liver biopsy or liver resection. Our dataset may differ in terms of image acquisition protocols, patient demographics, and disease severity distribution, which could impact model training and performance comparison.
c)	Evaluation Metrics
While both studies utilised AUC as a primary performance metric, other evaluation metrics might have differed. This necessitates careful consideration when comparing the reported results and drawing conclusions about relative performance.
d)	Clinical Workflow Integration
The referenced study primarily focused on evaluating the DCNN's diagnostic performance compared to radiologists. Our study could potentially extend beyond diagnostic assessment to explore integration into clinical workflows for personalised patient management, such as predicting disease progression or treatment response.

Further research with larger and more diverse datasets, incorporating standardised evaluation protocols and head-to-head comparisons, is needed to definitively establish the relative strengths and limitations of different approaches. By building upon the findings of previous studies, including the referenced work, we can advance the development of robust and reliable AI-powered tools for liver fibrosis assessment and ultimately improve patient care.

By further developing and validating this AI-powered diagnostic tool, we can transform the landscape of liver disease management. This paves the way for more personalised, efficient, and accessible care for patients worldwide. This work represents a significant step towards realising the potential of AI in healthcare, empowering healthcare professionals, and improving the lives of patients affected by liver disease.


Contact
mutebe2@gmail.com

