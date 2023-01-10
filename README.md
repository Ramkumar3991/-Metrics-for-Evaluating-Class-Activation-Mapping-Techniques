#Metrics-for-Evaluating-Class-Activation-Mapping-Techniques

**Abstract**

Over the last few years, many complicated computer vision problems have been solved successfully
using state-of-the-art Convolutional Neural Networks(CNN). It is always hard to understand the
inner functioning of deep models like CNN during their inference. Many recent works have shown
more interest in visualizing the CNN model’s behavior using Class Activation Mapping(CAM)-based
techniques. These techniques visualize the model’s prediction in terms of the class-discriminative
saliency maps. It is difficult to rate and compare the class-discriminative saliency maps of different
CAM-based methods because there is a lack of the appropriate metrics to evaluate the quality of
the class-discriminative saliency maps. The current metrics [1] such as "Average Drop"(AD) and
"Increase of Confidence"(IoC), assess the impact of removing irrelevant features from input images,
which will have a slight influence on the model’s prediction. In contrast to these metrics, we propose
metrics such as "Increased Average Drop"(IAD) and "Decrease of confidence"(DoC) to evaluate the
impact of removing the relevant features from input images because it can cause a drastic change
in the model’s prediction. We experiment and compare the four different CAM methods using two
datasets in this work to determine that the proposed metrics are suitable for evaluating CAM-based
methods and allow better interpretation when compared to the current metrics.

[1] -- Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, and Vineeth N. Balasubramanian. Grad-cam++:
Generalized gradient-based visual explanations for deep convolutional networks. CoRR, abs/1710.11063, 2017.


*Function folder* -- utility method needed for the main code
*Main folder* -- contains main code for evaluating class activation mappping techniques
