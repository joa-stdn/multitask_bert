# Around the BERT model: from a basic implementation to advanced optimizations and multitask learning
## CS224N Final Project

This project is the result of the collaboration between Ines, Yoni, and Joachim for our CS224N Final Project (Winter 2023).

### Description
The goal of this project is to develop a simple and efficient architecture for multitask learning
in natural language processing, based on a pretrained BERT model. Using BERT
contextualized embeddings and making them go through a single additional layer
per task, the **Multitask BERT** consistently achieves decent scores on three target
language problems: sentiment analysis, paraphrase detection and semantic textual
similarity (less than 20% below state-of-the-art performances).

During training,
one could choose to freeze BERT weights and only update additional parameters,
but we found that fine-tuning the BERT block to fit the underlying distribution
yielded better results

The use of PCGrad to get rid of conflicting gradients, as
well as gradient accumulation to boost training and additional datasets to better
understand diverse language styles are a few examples of ideas leveraged by our
model.

Our main contribution is the simplicity of the model, which requires less
that a single specific dense layer per task, contrary to many state-of-the-art papers
in existing literature.

This architecture is thus particularly adapted to end users
with limited resources willing to simultaneously achieve acceptable baseline results
on several downstream language tasks.


### Final Report
The final report is available here: http://web.stanford.edu/class/cs224n/final-reports/final-report-169376110.pdf (CS224N website).

### Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
