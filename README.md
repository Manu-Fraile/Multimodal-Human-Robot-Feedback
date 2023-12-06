# Multimodal-Human-Robot-Feedback
Master's Thesis research project developed at KTH. 

Supervisor of this thesis was Agnes Axelsson and Examiner Gabriel Skantze.

[Available Here](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1787723&dswid=9690)

# Abstract
When two human beings engage in a conversation, feedback is generally present since it helps in modulating and guiding the conversation for the involved parties. When a robotic agent engages in a conversation with a human, the robot is not capable of understanding the feedback given by the human as other humans would. In this thesis, we model human feedback as a Multivariate Time Series to be classified as positive, negative or neutral. We explore state-of-the-art Deep Learning architectures such as InceptionTime, a Convolutional Neural Network approach, and the Time Series Encoder, a Transformer approach. We demonstrate state-of-the art performance in accuracy, loss and f1-score of such models and improved performance in all metrics when compared to best performing approaches in previous studies such as the Random Forest Classifier. While InceptionTime and the Time Series Encoder reach an accuracy of 85.09% and 84.06% respectively, the Random Forest Classifier stays back with an accuracy of 81.99%. Moreover, InceptionTime reaches an f1-score of 85.07%, the Time Series Encoder of 83.27% and the Random Forest Classifier of 77.61%. In addition to this, we study the data classified by both Deep Learning approaches to outline relevant, redundant and trivial human feedback signals over the whole dataset as well as for the positive, negative and neutral cases.

# License
MIT License
