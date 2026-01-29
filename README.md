# Pneumonia Detection with Explainable Deep Learning

## 1. Introduction

Pneumonia is a serious respiratory infection that affects the lungs and can be life-threatening if not diagnosed early. Chest X-ray imaging is one of the most common and cost-effective diagnostic tools used by clinicians. However, manual interpretation of X-rays is time-consuming and depends heavily on expert radiologists. This project aims to automate pneumonia detection from chest X-ray images using deep learning, while also incorporating explainability to make the model’s predictions interpretable and trustworthy.

The project focuses on two key goals:

1. Building an accurate deep learning model for pneumonia detection.
2. Providing visual explanations for the model’s predictions using Grad-CAM.

---

## 2. Dataset Description

The dataset used in this project is the **Chest X-Ray Pneumonia dataset** obtained from Kaggle. It is organized into three splits:

- **Training set**: Used to train the model.
- **Validation set**: Used to tune hyperparameters and monitor overfitting.
- **Test set**: Used for final evaluation.

Each split contains two classes:

- **NORMAL**: Chest X-rays of healthy patients.
- **PNEUMONIA**: Chest X-rays showing pneumonia infection.

The dataset is moderately imbalanced, with more pneumonia images than normal images, which is typical for real-world medical datasets.

---

## 3. Image Preprocessing

Before feeding images into the neural network, several preprocessing steps are applied:

1. **Resizing**: All images are resized to 224 × 224 pixels to match the input requirements of the pretrained DenseNet121 model.
2. **Channel Handling**: Although chest X-rays are grayscale, images are converted to 3 channels (RGB format) to remain compatible with ImageNet-pretrained models.
3. **Normalization**: Pixel values are scaled to the range [0, 1] by dividing by 255. This improves numerical stability during training.
4. **Batching and Shuffling**: Images are grouped into batches and shuffled during training to improve generalization.

These preprocessing steps ensure consistency between training, evaluation, and explainability stages.

---

## 4. Model Architecture

### 4.1 Transfer Learning

Transfer learning is used to leverage knowledge from large-scale image datasets. Instead of training a convolutional neural network from scratch, a pretrained model is adapted to the pneumonia detection task.

### 4.2 DenseNet121 Backbone

The backbone of the model is **DenseNet121**, a convolutional neural network known for its dense connectivity pattern. In DenseNet, each layer receives feature maps from all previous layers, which:

- Encourages feature reuse.
- Reduces vanishing gradient problems.
- Improves parameter efficiency.

The DenseNet121 model is initialized with weights pretrained on ImageNet. The top (classification) layers are removed, and the network is used as a feature extractor.

### 4.3 Classification Head

On top of the DenseNet backbone, a custom classification head is added:

- **Global Average Pooling (GAP)**: Reduces spatial feature maps to a single feature vector while preserving semantic information.
- **Dense layer (ReLU)**: Learns task-specific combinations of features.
- **Dropout**: Reduces overfitting by randomly deactivating neurons during training.
- **Output layer (Sigmoid)**: Produces a probability score for pneumonia.

This architecture balances performance, efficiency, and interpretability.

---

## 5. Training Strategy

### 5.1 Freezing and Fine-Tuning

Initially, most layers of the pretrained DenseNet are frozen to preserve generic visual features such as edges and textures. Only the top layers are trained. Later, selective fine-tuning is applied by unfreezing the last few convolutional blocks, allowing the model to adapt to medical imaging features.

### 5.2 Loss Function and Optimization

- **Loss function**: Binary Cross-Entropy, suitable for binary classification.
- **Optimizer**: Adam optimizer with a low learning rate during fine-tuning to prevent catastrophic forgetting.

### 5.3 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

Special emphasis is placed on **recall for the pneumonia class**, as missing positive cases (false negatives) is critical in medical diagnosis.

---

## 6. Model Evaluation Metrics

The performance of the pneumonia detection model is evaluated on a held-out **test set** using standard classification metrics. These metrics provide insight into both overall performance and class-specific behavior, which is especially important in medical applications.

### 6.1 Classification Report

```
precision    recall  f1-score   support

NORMAL (0)       0.87      0.88      0.88       234
PNEUMONIA (1)    0.93      0.92      0.93       390

accuracy                           0.91       624
macro avg       0.90      0.90      0.90       624
weighted avg    0.91      0.91      0.91       624
```

### 6.2 Metric Interpretation

* **Accuracy (91%)**
  Indicates strong overall classification performance on unseen chest X-ray images.

* **Recall for Pneumonia (92%)**
  The model correctly identifies **92% of pneumonia cases**, which is critical in medical screening scenarios where false negatives can have severe consequences.

* **Precision for Pneumonia (93%)**
  High precision suggests that most pneumonia predictions made by the model are correct, minimizing unnecessary alarms.

* **F1-Score (93% for Pneumonia)**
  Demonstrates a strong balance between precision and recall, indicating stable and reliable learning.

* **Macro vs Weighted Average**
  The close values between macro and weighted averages suggest that the model performs consistently across classes despite moderate class imbalance.

### 6.3 Clinical Relevance

From a clinical screening perspective:

* The **high recall for pneumonia** makes the model suitable for **early detection and triage support**.
* The balanced precision reduces the burden of false positives on clinicians.
* The model is best positioned as a **decision-support or screening tool**, rather than a standalone diagnostic system.

> *Threshold selection and recall-oriented optimization were prioritized due to the high clinical cost of missed pneumonia cases.*

---

## 7. Explainability and Interpretability

### 7.1 Need for Explainability

In medical AI applications, model predictions must be interpretable to gain clinical trust. Explainability helps answer the question:

_Which regions of the chest X-ray influenced the model’s decision?_

### 7.2 Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM is used to visualize important regions in the input image that contribute to the model’s prediction. It works by:

1. Computing gradients of the predicted class score with respect to the final convolutional feature maps.
2. Averaging these gradients to obtain importance weights for each feature map.
3. Combining weighted feature maps to produce a heatmap.
4. Applying ReLU and normalization to highlight positive contributions.

The resulting heatmap is overlaid on the original X-ray image.

### 7.3 Interpretation of Heatmaps

- Red regions indicate areas strongly influencing pneumonia predictions.
- Ideally, these regions align with lung fields and visible opacities.
- Observed attention outside lung regions is discussed as a potential limitation or dataset bias.

Grad-CAM provides transparency and helps validate whether the model is learning clinically relevant features.

---

## 8. Model Engineering and Robustness

### 8.1 Functional API Reconstruction

To enable stable explainability, the trained model is reconstructed using the TensorFlow Functional API. Model weights are carefully transferred layer by layer from the saved model. This ensures:

- A clean computational graph.
- Reliable access to intermediate convolutional layers.
- Compatibility with gradient-based explainability methods.

### 8.2 Modular Design

All inference and explainability logic is encapsulated in a class-based design. The model itself is stored using TensorFlow’s native format, while explainability functions are implemented separately. This improves:

- Reusability
- Maintainability
- Deployment readiness

---

## 9. Limitations and Future Work

### Limitations

- The model may exhibit attention outside lung regions due to dataset bias.
- Chest X-rays lack depth information, which can limit diagnostic specificity.

### Future Improvements

- Lung segmentation to restrict attention to lung fields.
- Use of Grad-CAM++ for improved localization.
- Validation on external clinical datasets.
- Integration into a web or API-based diagnostic tool.

---

## 10. Conclusion

This project demonstrates an end-to-end deep learning pipeline for pneumonia detection using chest X-rays, combined with explainability through Grad-CAM. The model achieves strong classification performance while maintaining interpretability, making it suitable for academic, educational, and exploratory medical AI applications. The emphasis on explainability ensures transparency and aligns the system with real-world clinical expectations.
