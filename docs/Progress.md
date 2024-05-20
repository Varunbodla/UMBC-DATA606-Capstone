## $~~~~~~~~~~~$ Cross-Linguistic Communication - A Natural Language Processing Approach
- **GitHub -** *https://github.com/Varunbodla/UMBC-DATA606-Capstone*
- **PPT -** 
- **YouTube -** 

**Objective** :

  The main aim of this study is to develop language translation system leveraging deep learning techniques in NLP to enable accurate translations between English to Hindi. language.

**Need of this study** :
* In a globally interconnected world with diverse linguistic communities, effective communication across language barriers is essential.
* Traditional language translation methods often face challenges in capturing the contextual nuances, handling ambiguity, and providing accurate and contextually relevant translations.
* The need for a robust language translation solution is evident in various domains, including business, diplomacy, education, healthcare, and everyday communication.
* The goal is to develop a state-of-the-art language translation system that outperforms traditional methods, providing accurate, context-aware, and user-friendly translations for a diverse set of languages and applications.
* The success of this project will contribute to breaking down language barriers, fostering global communication, and promoting inclusivity in an increasingly interconnected world.
* Link to data source - *https://www.cfilt.iitb.ac.in/iitb_parallel/*

**Dataset** : 
* The dataset for this study would be The IIT Bombay English-Hindi Corpus compiled by the Indian Institute of Technology Bombay (IIT Bombay) for research purposes. This corpus consists of parallel text data in both English and Hindi languages.
* The dataset is of the size 99.7 MB
* This dataset contains pairs of sentences in English and Hindi, where each English sentence corresponds to its equivalent in Hindi. The corpus likely encompasses a diverse range of topics and genres to capture the variations in language usage across different contexts. It may include texts from various sources such as news articles, literature, websites, and possibly even user-generated content.

**Methodology**:

![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/a5e09ad4-5fe2-43db-ab67-aea6c668740b)

**Data Cleaning**:
- Originally, the dataset has 1.65 million datapoints in both English and Hindi languages. In order to simplify the study considering the computation resources, 5000 points 
  have been randomly sampled from the original dataset.
- The following text cleaning steps have been performed.
   - Apply decontractions for English language text. For example, can’t to cannot and wouldn’t to would not.
   - Lower case the text. 
   - Remove all the non-alphabetic characters from the text string leaving only letters.
   - For Hindi language text, extracting only Hindi word text.
   - Removing single letter words
   - Removing stop words from the English text.
   - Dropping rows with empty strings in the Hindi and English columns.
 - The effective shape of the dataset has become 4859 datapoints.

**Exploratory Data Analysis**:

It is important to analyze the word lengths for hindi and english sentences to check if there is any skewness in its distribution.

![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/25b2ec08-aa74-4453-aec4-ce4c20dbe4d2)

It can be observed from the above histograms that the distribution of word lengths for both Hindi and English are skewed towards right. It can be inferred that there are more sentences with less number of words in text in both English and Hindi languages for the given data.

**Model Architecture**:

![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/7ff9260a-20b0-488b-990a-6264f9f214e4)

The following is the workflow for Encoder - Decoder model.

- We create 2 inputs for the model
- One is fed to the Encoder as the input.
- The Encoder will generate Encoder Output and the Encoder states.
- The Encoder states are then fed to the decoder along with decoder input.
- The output of the decoder is passed through the dense layer.
- The dense layer predicts the output word.

Here, for this model there are 3,055,383 trainable parameters (of size 11.66 MB)

The model is trained with the following parameters.

- Optimizer - Adam
- Initial learning rate - 0.001
- Loss - sparse_categorical_crossentropy
- Metric - sparse_categorical_accuracy
- Batch Size - 64
- Epochs - 50 
- Early Stopping with a patience of 4
- ReduceLROnPlateau with a patience of 2 by a factor of 0.95

**Results**:

The solution converged at 50th Epoch. At Convergence, the following results are obtained.

![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/e64f1e52-4247-4584-ad16-9bd004f91815)

| Train Loss  | Test Loss   | Train Accuracy | Test Accuracy |
|-------------|-------------|----------------|---------------|
| 9.0221e-04  | 8.8506e-04  | 1.00           | 1.00          |

**Conclusion**

The machine translation model from English to Hindi performs exceptionally well with perfect accuracy and very low error rates on the provided data. This excellent performance offers several real-world benefits:

- Better Access - The model makes it easier for Hindi-speaking people to access information and services that were previously available only in English.
- Enhanced Communication - It improves communication in personal, professional, and business contexts, helping people understand each other better.
- Business Expansion - Companies can reach more customers in the Hindi-speaking market, driving business growth.
- Educational Support -  Students and learners can access a wider range of educational materials, aiding their learning and development.
- Healthcare Improvement -  Hindi-speaking patients can get accurate medical information and care, leading to better health outcomes.

In summary, the model's ability to accurately translate between English and Hindi can positively impact various areas, making information and services more accessible and communication more effective.

**Deployment**:

The web application has been built using Streamlit and has been deployed in Streamlit Cloud.

![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/957b1617-15ce-48e0-a0c8-c7b82c005df4)

Link - https://language-translation-application.streamlit.app/

**Future Work**:

- Due to the limited computation resources, the model has been trained with just 5000 datapoints. The model could very well be learned with all the 1.65 million points with   a good neural machine and therefore can make the model generalize well on the unseen data.
- The Attention layer can be added to the encoder decoder model to learn the long sequences effectively.
