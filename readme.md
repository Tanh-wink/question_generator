English | [中文](readme_zh.md)

# 1.Dependencies

1. **OS:** Ubuntu 16.04 LTS

2. **Python version**  3.6.12

3. **Python package**
    + tensorflow-gpu       1.14.0

    + Keras                2.3.1
    + scikit-learn         0.23.2
    + tqdm                 4.50.0

4. **CUDA version**  10.0

5. **cuDNN version** 7.6.0

    


# 2. Solution

## 2.1 Model ideas
In response to the problem of "problem generation of Chinese medicine literature", our team's solution is as follows:

+ **Training:** 
(1) The pre-trained models of UniLM-MASK and BERT types are used as the baseline, and the sentence pair of "document + answer + question" is used as input. (2) Combining label smoothing with adversarial perturbations based on Embedding layers to prevent overfitting. (3)Using knowledge distillation techniques to improve the generalization performance of a single model.
+ **Generating：** (1) Using beam search strategy for question generation.(2) In the word prediction stage at each time step, use ensemble voting prediction based on **WoBERT** and **WoNEZHA**.

This solution ranks fourth in the semi-finals list，with **Rouge-L** score：0.6278.
	

## 2.2 Data preprocessing
Use the load_train_json function in main.py to extract the QA pair from the original data (round1_train_0907.json).   
We have tried removing some illegal characters from the original text and replacing some characters. However, the performance of the model has not been improved. Therefore, the data preprocessing module only plays the role of data extraction and encoding for the time being.    
Concretely, We take text and answer as Bert's first sentence, and question as Bert's second sentence.The encoding format is as follows:  
~~~
    token_ids: [CLS] + text + [SEP] + answer + [SEP] + question + [SEP]
    segment_ids:  0 + 0 + 0 + 0 + 0 + 1 + 1	
~~~
Data set division:
	We use sklearn's train_test_split function to split the training data, 90% as training set and 10% as validation set. Make the experiment reproducible by setting the random seed to 42 .

## 2.3 Model Building
### 2.3.1 Pretraining models
According to the experimental results that have been done, we choose the two best-performing bert models as the objects of the ensemble:   

+ The first is **WoBERT** ，which pre-training weights comes from the open source of **Chase Yi Technology** [chinese_wobert_L-12_H-768_A-12](https://github.com/ZhuiyiTechnology/WoBERT)  。
+ The second is **WoNEZHA** ，which pre-training weights comes from the open source of **Chase Yi Technology** [chinese_wonezha_L-12_H-768_A-12](https://github.com/ZhuiyiTechnology/WoBERT) 。

### 2.3.2 UniLM-MASK

In order to give the BERT model the ability of seq2seq and handle NLG tasks such as: question generation, 
the paper [《Unified Language Model Pre-training for Natural Language Understanding and Generation》](http://papers.nips.cc/paper/9464-unified-language-model-pre-training-for-natural-language-understanding-and-generation.pdf) was presented at the NIPS 2019. We use the Seq2seq Mask mentioned in the paper to replace the attention mask in the original Bert Multi-head attention, so that Bert has the following characteristics during the training process： 

+ First sentence (text + answer) can only see its own tokens，but not the tokens of the second sentence (question).   
+ Second sentence can only see the preceding tokens, including the tokens contained in the first sentence (text + answer).      

These two properties give Bert the seq2seq capability.

## 2.3.2 Label Smoothing

Label smoothing is a regularization method commonly used in classification problems to prevent models from overconfidently predicting labels during training, and improve problems with poor generalization.

Smooth the real labels

$$  
\hat{y}_i = y_i*(1 - \alpha) + \frac{\alpha}{K}
$$  

$y_i$ is the one-hot label vector of the i sample, which dimension is the size of the vocabulary.

$\alpha$ is the smoothing factor, usually setting 0.1. $K$ is the number of categories， $\hat{y}_i$ is smoothed label vector.

### 2.3.3 Adversarial perturbations based embedding layers

Adversarial perturbation is essentially adversarial training, which is to construct some adversarial samples and add them to the original data set to enhance the robustness of the model to adversarial samples and improve the performance of the model. But the input of NLP is text, which is essentially a one-hot vector, so there is no so-called small perturbation. Therefore, we can do adversarial perturbations from the Embedding layer. In our scheme, we directly perturb the weight of the Embedding layer to change the word vector after look-up.

The formula against perturbation:

$$
\mathop{min}\limits_{\theta} \mathbb{E}_{(x,y) \in D}[\mathop{max}\limits_{\Delta x\in \Omega} Loss(x+\Delta x, y; \theta)]
$$

  $\theta$ is the parameters of model， $L(x,y;\theta)$ is the loss of a single model. $\Delta x$ is adversarial perturbation. $\Omega$ is the disturbance space.

（1）Add adversarial perturbation $\Delta x$ to $x$, the purpose is to make the Loss as large as possible, that is, try to make the model prediction error.  
（2） Of course $\Delta x$ is not as big as possible, so it will have a constraint space $\Omega$
（3）After each sample is constructed against the sample $x + \Delta x$, and use it as the input of the model to minimize loss and update the parameters of the model  


Calculated $\Delta x$ using ** FGM **  

Because the purpose is to increase loss, and the method of loss reduction is gradient descent, then the method of loss increase, we can use gradient ascent

So, we can take

$$
\Delta x = \epsilon \triangledown_x Loss(x, y; \theta)
$$

$\epsilon$ is a hyperparameter, generally 0.1.  

In order to prevent the calculated gradient from being too large, we normalize the gradient:

$$
\Delta x = \epsilon \frac{\triangledown_x Loss(x, y; \theta)}{||\triangledown_x Loss(x, y; \theta)||}
$$

Adversarial perturbation for Embedding Weights, dimension:

$x \in \mathbb{R}^{vocab\_size, dim}$  is the weights of the word embedding layer.  

$\triangledown_x Loss(x, y; \theta) \in \mathbb{R}^{vocab\_size, dim}$ is the gradient of the word embedding layer.  

### 2.3.4 Knowledge Distillation

Knowledge distillation is to guide the training of the Student Model by introducing the soft-target related to the Teacher Model as part of the total loss to achieve knowledge transfer.

In our solution, Teacher Model and Student Model are BERT models with the same structure.

**The implementation details of knowledge distillation:**

1. Train a Teacher Model;
2. During the training process of the Student Model, add the label probability (soft target) output by the Teacher Model, and calculate the softmax loss with it;
3. Finally superimposed with the softmax loss of the real label (hard target) as a total loss 

**temperature coefficient：**

It may not be appropriate to directly use the predicted probabilities output by the trained teacher model.

Because, after a network is trained, it has a high degree of confidence in the positive label, and the value of the negative label is very close to 0, and the contribution to the loss function is very small, so small that it can be ignored.

Therefore, a temperature variable can be introduced to make the probability distribution smoother.

$$
\hat{t}_i = softmax(t_i/T)
$$

$t_i$ is the probability vector of the teacher model before $softmax$。 

$T$ is the scaling factor.

When $T$ is higher, the output probability of $softmax$ is smoother, the entropy of its distribution is larger, the information carried by negative labels will be relatively amplified, and model training will pay more attention to negative labels.

Amplifying the probability of negative labels has another benefit: it allows the Student model to learn the relationship between different negative labels and positive labels. For example, a dog may have a probability value of 0.001 in the category of cats, while the probability value in the category of cars may be less than 0.0000001, which can reflect that dogs and cats are more similar than dogs and cars. Scale neural network can obtain more similar information between data structures.

### 2.3.5 Loss Function

1. Teacher Model: The Loss function used for training is the KL divergence between "all the question words predicted by Bert" and "all the original question words" (after label smoothing). 

2. Student Model: The Loss function used in training is the KL divergence of "probability of all question words predicted by Bert" and "probability of all original question words" (after label smoothing) and "probability of all question words predicted by Bert" and "Teacher predicted The KL divergence of all question word probabilities". 

### 2.3.6 Save model：

After each epoch, the Rouge-L score of the model on the validation set is calculated. If the Rouge-L is higher than the previous optimal Rouge-L, the latest model is saved. We use greedy search for question generation on the validation set.  

## 2.4 Experimental details：

(1) Text length settings (mainly based on the distribution of text lengths): 

    + text 的最大长度 (max_t_len) 为 384  
    + answer 的最大长度 (max_a_len) 为 96  
    + question 的最大长度 (max_q_len) 为 32  

(2) Training parameters:  

    + batch_size : 4   
    + gradient_accumulation_steps : 8   
    + EPOCHS: 5 
    + 标签平滑的平滑因子 (label_weight) : 0.1   
    + 对抗训练的 $\epsilon$  (ADV_epsilon) : WoBERT 为 0.3， WoNEZHA 为 0.1   
    + Teache model 在 Student loss 中所占的权重 (teacher_rate) : 0.5  
    + 温度系数 (temperature) : 10  
    
(3) Optimizer settings:  

    + Adam optimizer  
    + Initial learning rate is 3e-5  
    + 使用学习率线性衰减函数，让 学习率 从第 1 个 step 到 最后一个 step ，线性衰减到 初始学习率 的 50% 。

## 2.5 Model prediction： 
加载训练好的 **WoBERT** 和 **WoNEZHA** 的 Student Model   

在进行问题生成时，对于每个 token 的预测，将两个模型预测的 logits 进行平均加权求和，让预测出来的 token 尽量在 **WoBERT** 和 **WoNEZHA** 中得分靠前。  

我们采用 beam search 的方式，来对测试集进行问题生成， beam 的个数为 5 。	
