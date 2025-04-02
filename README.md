# Dog Breed Classification using Transfer Learning

## Project Overview
This project focuses on the development of a **deep learning-based dog breed classification model** utilizing **Convolutional Neural Networks (CNNs)** and **Transfer Learning**. Leveraging a pretrained **ResNet18** model, the system classifies images into **70 distinct dog breeds**. The project demonstrates the power of transfer learning for fine-grained classification tasks and analyzes model performance, including class-wise accuracy and misclassification patterns.

---

## Dataset
- Total Classes: 70 dog breeds
- Training Set: 7,946 images
- Validation Set: 700 images
- Test Set: 700 images

All images were resized to **224x224** pixels. Data augmentation techniques were applied to improve generalization.

---

## Model Architecture
- **Backbone**: ResNet18 pretrained on ImageNet
- **Classifier Head**: Fully connected layer mapping extracted features to 70 classes
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with a learning rate of 3e-5
- **Scheduler**: Cosine Annealing
- **Regularization**: Early Stopping & Data Augmentation

---

## Key Features
✅ **Transfer Learning**: Leveraging pretrained ResNet18 weights
✅ **Data Augmentation**: Rotation, flipping, color jittering, and more
✅ **Full Fine-Tuning**: All ResNet layers were trainable
✅ **Balanced Dataset**: Ensuring equal representation of all breeds
✅ **Confusion Matrix**: Detailed analysis of misclassifications

---

## Results
| Metric | Value |
|--------|-------|
| **Training Accuracy** | 83% |
| **Validation Accuracy** | 92% |
| **Test Accuracy** | 95% |

### Class-wise Highlights
- Breeds like **Afghan**, **Coyote**, **Dalmatian**, **Dhole**, and **Vizsla** were classified perfectly (100% accuracy)
- Confusions occurred mainly between visually similar breeds, e.g., **Boston Terrier** vs **Bulldog**

---

## Challenges
- Subtle differences between certain breeds
- Misclassifications in occluded or poorly lit images
- Difficulty generalizing to unseen breeds

---

## Future Work
- Expand dataset with more rare and underrepresented breeds
- Explore advanced architectures like **EfficientNet** or **Vision Transformers (ViTs)**
- Integrate attention mechanisms for focusing on discriminative features
- Implement curriculum learning and domain adaptation
- Experiment with **multimodal learning** combining images with textual descriptions

---

## Conclusion
The project successfully demonstrates the effectiveness of CNNs combined with transfer learning for dog breed classification. The model achieved strong generalization capabilities and performed exceptionally well, reaching **95% test accuracy**. Further extensions could push performance even higher and adapt the model for real-world applications.

---



