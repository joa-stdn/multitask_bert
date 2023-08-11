import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from pcgrad import PCGrad
#from torch.utils.tensorboard import SummaryWriter

from datasets import SentenceClassificationDataset, SentencePairDataset, MultiTaskDataset, load_multitask_data, load_multitask_test_data
from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask
from constants import TQDM_DISABLE, TENSORBOARD_DIR

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
N_CFIMDB_CLASSES = 2
HIDDEN_SIZE = 32
N_SIM_LABELS = 6


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        # self.linear2 = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.linear2(x)
        return x
    
class MLP_simple(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_simple, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class LSTMHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMHead, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        self.lstm.flatten_parameters() 
        _, (x, _)  = self.lstm(x)
        hidden_last_L=x[-2]
        hidden_last_R=x[-1]
        hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
        x = self.dropout(hidden_last_out)
        x = self.fc(x)
        return x

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.MLP_sentiment = MLP(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.MLP_paraphrase = MLP_simple(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.MLP_similarity = MLP_simple(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
    
        self.LSTM_sentiment = LSTMHead(BERT_HIDDEN_SIZE, HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.LSTM_paraphrase = LSTMHead(BERT_HIDDEN_SIZE, HIDDEN_SIZE*4, HIDDEN_SIZE)
        self.LSTM_similarity = LSTMHead(BERT_HIDDEN_SIZE, HIDDEN_SIZE*4, HIDDEN_SIZE)

        self.layer_sentiment = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.layer_cfimdb = torch.nn.Linear(BERT_HIDDEN_SIZE, N_CFIMDB_CLASSES)
        self.sim_relu = nn.ReLU()
        self.sim_sigmoid = torch.nn.Sigmoid()
        # self.final_layer_similarity = torch.nn.Linear(2 * HIDDEN_SIZE, N_SIM_LABELS)
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cosine_sim_para_parameter = nn.Parameter(torch.FloatTensor([0.002]), requires_grad=True)
        self.cosine_sim_sim_parameter = nn.Parameter(torch.FloatTensor([0.002]), requires_grad=True)


        # # predict sentiment
        # #  https://www.geeksforgeeks.org/fine-tuning-bert-model-for-sentiment-analysis/
        # # relu activation function
        # self.relu =  nn.ReLU()
        # # dense layer 1
        # self.fc1 = nn.Linear(768,512) 
        # # dense layer 2 (Output layer)
        # self.fc2 = nn.Linear(512,2)

        # # predict paraphrase

        # # predict similarity
        # # use cosine similarity loss if possible



    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        outputs = self.bert(input_ids, attention_mask)
        return outputs


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        outputs = self.forward(input_ids, attention_mask)
        cls_output = outputs["pooler_output"]
        # embeddings = outputs["last_hidden_state"]
        # sentiment = self.LSTM_sentiment(embeddings)
        sentiment = self.layer_sentiment(cls_output)

        return sentiment
    
    def predict_cfimdb(self, input_ids, attention_mask):
        outputs = self.forward(input_ids, attention_mask)
        cls_output = outputs["pooler_output"]
        # embeddings = outputs["last_hidden_state"]
        # mean_embeddings = torch.mean(embeddings, dim=1)
        sentiment = self.layer_cfimdb(cls_output)
        return sentiment


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        outputs_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_1 = outputs_1["last_hidden_state"]
        mean_embeddings_1 = torch.mean(embeddings_1, dim=1)
        # paraphrase1 = self.MLP_paraphrase(mean_embeddings_1)
        # paraphrase1 = self.LSTM_paraphrase(embeddings_1)

        outputs_2 = self.forward(input_ids_2, attention_mask_2)
        embeddings_2 = outputs_2["last_hidden_state"]
        mean_embeddings_2 = torch.mean(embeddings_2, dim=1)
        # paraphrase2 = self.MLP_paraphrase(mean_embeddings_2)
        # paraphrase2 = self.LSTM_paraphrase(embeddings_2)

        paraphrase = self.cosine_sim(mean_embeddings_1, mean_embeddings_2)
        paraphrase = torch.clip(1000*self.cosine_sim_para_parameter*(paraphrase-(1-1/(1000*self.cosine_sim_para_parameter))), 0, 1)

        return paraphrase

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        
        outputs_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_1 = outputs_1["last_hidden_state"]
        mean_embeddings_1 = torch.mean(embeddings_1, dim=1)
        # similar1 = self.LSTM_similarity(embeddings_1)


        outputs_2 = self.forward(input_ids_2, attention_mask_2)
        embeddings_2 = outputs_2["last_hidden_state"]
        mean_embeddings_2 = torch.mean(embeddings_2, dim=1)
        # similar2 = self.LSTM_similarity(embeddings_2)
        # similar2 = self.layer_similarity(mean_embeddings_2)
        # similar2 = self.sim_relu(similar2)

        # Concatenate similar1 and similar2 on the last dimension
        # similar = torch.cat((similar1, similar2), dim=1)
        # similar = self.final_layer_similarity(similar)
        #similar = self.sim_sigmoid(similar)
        #similar = 5 * (similar)

        similar = self.cosine_sim(mean_embeddings_1, mean_embeddings_2)
        similar = torch.clip(1000*self.cosine_sim_sim_parameter*(similar.squeeze(0)-(1-1/(1000*self.cosine_sim_sim_parameter))), 0, 1)
        return similar


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict() if not args.use_pcgrad else optimizer._optim.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def predict_loss_sentiment(criterion, model, batch, device, args):
    b_ids, b_mask, b_labels = (batch['token_ids'],
                                batch['attention_mask'], batch['labels'])

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    b_labels = b_labels.to(device)

    logits = model.predict_sentiment(b_ids, b_mask)
    sst_pred = logits.argmax(dim=-1).flatten().cpu().detach().numpy()
    sst_true = b_labels.flatten().cpu().detach().numpy()
    
    return criterion(logits, b_labels.view(-1)), sst_pred, sst_true

def predict_loss_cfimdb(criterion, model, batch, device, args):
    b_ids, b_mask, b_labels = (batch['token_ids'],
                                batch['attention_mask'], batch['labels'])

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    b_labels = b_labels.to(device)

    logits = model.predict_cfimdb(b_ids, b_mask)
    cfimdb_pred = logits.argmax(dim=-1).flatten().cpu().detach().numpy()
    cfimdb_true = b_labels.flatten().cpu().detach().numpy()
    
    return criterion(logits, b_labels.view(-1)), cfimdb_pred, cfimdb_true
    

def predict_loss_paraphrase(criterion, model, batch, device, args):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
        batch['token_ids_1'].to(device), 
        batch['attention_mask_1'].to(device),
        batch['token_ids_2'].to(device),
        batch['attention_mask_2'].to(device),
        batch['labels'].to(device)
    )

    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    para_pred = logits.sigmoid().round().flatten().cpu().detach().numpy()
    para_true = b_labels.flatten().cpu().detach().numpy()

    return criterion(logits, b_labels.float()), para_pred, para_true



def predict_loss_similarity(criterion, model, batch, device, args):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
        batch['token_ids_1'].to(device), 
        batch['attention_mask_1'].to(device),
        batch['token_ids_2'].to(device),
        batch['attention_mask_2'].to(device),
        batch['labels'].to(device)
    )

    logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    sim_pred = logits.flatten().cpu().detach().numpy()
    sim_true = b_labels.flatten().cpu().detach().numpy()

    return criterion(logits, b_labels.float()/5), sim_pred, sim_true

## Currently only trains on sst dataset
def train_multitask(args):
    
    

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    #writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data, cfimdb_train_data, num_labels2, yelp_train_data = \
        load_multitask_data(args.sst_train,args.para_train,args.sts_train, args.cfimdb_train, args.yelp_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data, cfimdb_dev_data, num_labels2, yelp_dev_data = \
        load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.cfimdb_dev, args.yelp_dev, split ='train')

    # we won't be using cfimdb test data for now

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    cfimdb_train_data = SentenceClassificationDataset(cfimdb_train_data, args)
    cfimdb_dev_data = SentenceClassificationDataset(cfimdb_dev_data, args)
    yelp_train_data = SentenceClassificationDataset(yelp_train_data, args)
    yelp_dev_data = SentenceClassificationDataset(yelp_dev_data, args)

    train_multi_task_dataset = MultiTaskDataset(sst_train_data, para_train_data, sts_train_data, cfimdb_train_data, yelp_train_data, args)

    train_multi_task_dataloader = DataLoader(train_multi_task_dataset, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=train_multi_task_dataset.collate_fn)

    dev_sst_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                  collate_fn=sst_dev_data.collate_fn)
    dev_para_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                  collate_fn=para_dev_data.collate_fn)
    dev_sts_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                  collate_fn=sts_dev_data.collate_fn)
    dev_cfimdb_dataloader = DataLoader(cfimdb_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=cfimdb_dev_data.collate_fn)
    dev_yelp_dataloader = DataLoader(yelp_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=yelp_dev_data.collate_fn)

    # Init model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
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    if args.load_pretrained is not None:
        saved = torch.load(args.load_pretrained)
        model.load_state_dict(saved['model'])

    lr = args.lr
    
    if args.use_pcgrad:
        optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    else:
        optimizer = AdamW(model.parameters(), lr=lr)

    best_avg_dev_acc = 0
    criterion_sim = nn.MSELoss()
    criterion_sst = nn.CrossEntropyLoss()
    criterion_para = nn.BCEWithLogitsLoss()
    criterion_cfimdb = nn.CrossEntropyLoss() # can change
    
    gradient_accumulation_total_steps = 4

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss_sst = 0
        train_loss_para = 0
        train_loss_sim = 0
        train_loss_cfimdb = 0
        train_loss_yelp = 0
        num_batches = 0
        train_para_y_true, train_para_y_pred = [], []
        train_sts_y_true, train_sts_y_pred = [], []
        train_sst_y_true, train_sst_y_pred = [], []
        train_cfimdb_y_true, train_cfimdb_y_pred = [], []
        train_yelp_y_true, train_yelp_y_pred = [], []

        for batch in tqdm(train_multi_task_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            if TQDM_DISABLE and num_batches % 100 == 0: print(num_batches, "/", len(train_multi_task_dataloader))
            sentiment_batch, paraphrase_batch, similarity_batch, cfimdb_batch, yelp_batch\
                  = batch["sentiment"], batch["paraphrase"], batch["similarity"], batch["cfimdb"], batch["yelp"]
            optimizer.zero_grad()
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    #loss_para, para_pred, para_true = predict_loss_paraphrase(criterion_para, model, paraphrase_batch, device, args)
                    #loss_para = loss_para / gradient_accumulation_total_steps
                    loss_yelp, yelp_pred, yelp_true = predict_loss_sentiment(criterion_cfimdb, model, yelp_batch, device, args)
                    loss_yelp = loss_yelp / gradient_accumulation_total_steps
                    if sentiment_batch is not None:
                        loss_sst, sst_pred, sst_true = predict_loss_sentiment(criterion_sst, model, sentiment_batch, device, args)
                        loss_sst = loss_sst / gradient_accumulation_total_steps
                        train_sst_y_pred.extend(sst_pred)
                        train_sst_y_true.extend(sst_true)
                        train_loss_sst += loss_sst.item()
                    else:
                        loss_sst = None
                    if similarity_batch is not None:
                        loss_sim, sim_pred, sim_true = predict_loss_similarity(criterion_sim, model, similarity_batch, device, args)
                        loss_sim = loss_sim / gradient_accumulation_total_steps
                        train_sts_y_pred.extend(sim_pred)
                        train_sts_y_true.extend(sim_true)
                        train_loss_sim += loss_sim.item()
                    else:
                        loss_sim = None
                    if cfimdb_batch is not None:
                        loss_cfimdb, cfimdb_pred, cfimdb_true = predict_loss_cfimdb(criterion_cfimdb, model, cfimdb_batch, device, args)
                        loss_cfimdb = loss_cfimdb / gradient_accumulation_total_steps
                        train_cfimdb_y_pred.extend(cfimdb_pred)
                        train_cfimdb_y_true.extend(cfimdb_true)
                        train_loss_cfimdb += loss_cfimdb.item()
                    else:
                        loss_cfimdb = None
                    # if yelp_batch is not None:
                    #     loss_yelp, yelp_pred, yelp_true = predict_loss_cfimdb(criterion_cfimdb, model, yelp_batch, device, args)
                    #     train_yelp_y_pred.extend(yelp_pred)
                    #     train_yelp_y_true.extend(yelp_true)
                    #     train_loss_yelp += loss_yelp.item()
                    # else:
                    #     loss_yelp = None
                    if paraphrase_batch is not None:
                        loss_para, para_pred, para_true = predict_loss_paraphrase(criterion_para, model, paraphrase_batch, device, args)
                        loss_para = loss_para / gradient_accumulation_total_steps
                        train_para_y_pred.extend(para_pred)
                        train_para_y_true.extend(para_true)
                        train_loss_para += loss_para.item()
                    else:
                        loss_para = None

            else:
                # loss_para, para_pred, para_true = predict_loss_paraphrase(criterion_para, model, paraphrase_batch, device, args)
                # loss_para = loss_para / gradient_accumulation_total_steps
                loss_yelp, yelp_pred, yelp_true = predict_loss_cfimdb(criterion_cfimdb, model, yelp_batch, device, args)
                loss_yelp = loss_yelp / gradient_accumulation_total_steps
                if sentiment_batch is not None:
                    loss_sst, sst_pred, sst_true = predict_loss_sentiment(criterion_sst, model, sentiment_batch, device, args)
                    loss_sst = loss_sst / gradient_accumulation_total_steps
                    train_sst_y_pred.extend(sst_pred)
                    train_sst_y_true.extend(sst_true)
                    train_loss_sst += loss_sst.item()
                else:
                    loss_sst = None
                if similarity_batch is not None:
                    loss_sim, sim_pred, sim_true = predict_loss_similarity(criterion_sim, model, similarity_batch, device, args)
                    loss_sim = loss_sim / gradient_accumulation_total_steps
                    train_sts_y_pred.extend(sim_pred)
                    train_sts_y_true.extend(sim_true)
                    train_loss_sim += loss_sim.item()
                else:
                    loss_sim = None
                if cfimdb_batch is not None:
                    loss_cfimdb, cfimdb_pred, cfimdb_true = predict_loss_cfimdb(criterion_cfimdb, model, cfimdb_batch, device, args)
                    train_cfimdb_y_pred.extend(cfimdb_pred)
                    train_cfimdb_y_true.extend(cfimdb_true)
                    train_loss_cfimdb += loss_cfimdb.item()
                else:
                    loss_cfimdb = None
                if paraphrase_batch is not None:
                    loss_para, para_pred, para_true = predict_loss_paraphrase(criterion_para, model, paraphrase_batch, device, args)
                    loss_para = loss_para / gradient_accumulation_total_steps
                    train_para_y_pred.extend(para_pred)
                    train_para_y_true.extend(para_true)
                    train_loss_para += loss_para.item()
                else:
                    loss_para = None


            train_yelp_y_pred.extend(yelp_pred)
            train_yelp_y_true.extend(yelp_true)


            if args.use_pcgrad:
                losses = [loss_sst, loss_para, loss_sim, loss_cfimdb, loss_yelp]
                optimizer.pc_backward([loss for loss in losses if loss is not None], scaler=scaler)
            elif args.use_amp:
                #scaler.scale(loss_para).backward()
                scaler.scale(loss_yelp).backward()
                if loss_sst is not None:
                    scaler.scale(loss_sst).backward()
                if loss_sim is not None:
                    scaler.scale(loss_sim).backward()
                if loss_cfimdb is not None:
                    scaler.scale(loss_cfimdb).backward()
                if loss_para is not None:
                    scaler.scale(loss_para).backward()
            else:
                #loss_para.backward()
                loss_yelp.backward()
                if loss_sst is not None:
                    loss_sst.backward()
                if loss_sim is not None:
                    loss_sim.backward()
                if loss_cfimdb is not None:
                    loss_cfimdb.backward()
                if loss_para is not None:
                    loss_para.backward()

            if ((num_batches + 1) % gradient_accumulation_total_steps == 0) or (num_batches + 1 == len(train_multi_task_dataloader)): 
                if args.use_amp:
                    if args.use_pcgrad:
                        scaler.step(optimizer._optim)
                    else:
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            #train_loss_para += loss_para.item()
            train_loss_yelp += loss_yelp.item()

            del loss_sst, loss_para, loss_sim, loss_cfimdb, loss_yelp
            torch.cuda.empty_cache()

            num_batches += 1

        train_loss_sst = train_loss_sst / (num_batches)
        train_loss_para = train_loss_para / (num_batches)
        train_loss_sim = train_loss_sim / (num_batches)
        train_loss_cfimdb = train_loss_cfimdb / (num_batches)
        train_loss_yelp = train_loss_yelp / (num_batches)

        train_acc_para = np.mean(np.array(train_para_y_pred) == np.array(train_para_y_true))
        train_corr_sim = np.corrcoef(train_sts_y_pred,train_sts_y_true)[1][0]
        train_acc_sst = np.mean(np.array(train_sst_y_pred) == np.array(train_sst_y_true))
        train_acc_cfimdb = np.mean(np.array(train_cfimdb_y_pred) == np.array(train_cfimdb_y_true))
        train_acc_yelp = np.mean(np.array(train_yelp_y_pred) == np.array(train_yelp_y_true))

        # train_acc_para, _, _, train_acc_sst, _, _, train_corr_sim, _, _ = \
        #     model_eval_multitask(train_sst_dataloader, train_para_dataloader, \
        #     train_sts_dataloader, model, device)

        # train_acc_sst, _, _, _, _, _ = model_eval_sst(train_sst_dataloader, model, device)
        # dev_acc_sst, _, _, _, _, _ = model_eval_sst(dev_sst_dataloader, model, device)
        
        dev_acc_para, _, _, dev_acc_sst, _, _, dev_corr_sim, _, _, dev_acc_cfimdb,_,_, \
        dev_acc_yelp, _, _= \
            model_eval_multitask(dev_sst_dataloader, dev_para_dataloader, \
                dev_sts_dataloader, dev_cfimdb_dataloader, dev_yelp_dataloader ,model, device)

        if np.isnan(dev_corr_sim):
            dev_corr_sim = 0
        avg_dev_acc = (dev_acc_para + dev_acc_sst + abs(dev_corr_sim)) / 3
        avg_dev_acc2 = (dev_acc_para + dev_acc_sst + dev_acc_cfimdb + abs(dev_corr_sim)) / 4
        avg_dev_acc3 = (dev_acc_para + dev_acc_sst + dev_acc_cfimdb + dev_acc_yelp + abs(dev_corr_sim)) / 5
        if avg_dev_acc3 > best_avg_dev_acc:
            best_avg_dev_acc = avg_dev_acc3
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}:")
        print(f"*** train loss SST :: {train_loss_sst :.3f}, train acc SST :: {train_acc_sst :.3f}, dev acc SST :: {dev_acc_sst :.3f}")
        print(f"*** train loss Para :: {train_loss_para :.3f}, train acc Para :: {train_acc_para :.3f}, dev acc Para :: {dev_acc_para :.3f}")
        print(f"*** train loss Sim :: {train_loss_sim :.3f}, train corr Sim :: {train_corr_sim :.3f}, dev corr Sim :: {dev_corr_sim :.3f}")
        print(f"*** train loss CFIMDB :: {train_loss_cfimdb :.3f}, train acc CFIMDB :: {train_acc_cfimdb :.3f}, dev acc CFIMDB :: {dev_acc_cfimdb :.3f}")
        print(f"*** train loss Yelp :: {train_loss_yelp :.3f}, train acc Yelp :: {train_acc_yelp :.3f}, dev acc Yelp :: {dev_acc_yelp :.3f}")

    #     writer.add_scalar("Loss/train/sst", train_loss_sst, epoch)
    #     writer.add_scalar("Loss/train/para", train_loss_para, epoch)
    #     writer.add_scalar("Loss/train/sim", train_loss_sim, epoch)

    #     writer.add_scalar("Acc/train/sst", train_acc_sst, epoch)
    #     writer.add_scalar("Acc/train/para", train_acc_para, epoch)
    #     writer.add_scalar("Acc/train/sim", train_corr_sim, epoch)

    #     writer.add_scalar("Acc/dev/sst", dev_acc_sst, epoch)
    #     writer.add_scalar("Acc/dev/para", dev_acc_para, epoch)
    #     writer.add_scalar("Acc/dev/sim", dev_corr_sim, epoch)
    
    # writer.close()


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--cfimdb_train", type=str, default="data/ids-cfimdb-train.csv")
    parser.add_argument("--cfimdb_dev", type=str, default="data/ids-cfimdb-dev.csv")
    parser.add_argument("--cfimdb_test", type=str, default="data/ids-cfimdb-test-student.csv")

    parser.add_argument("--yelp_train", type=str, default="data/ids-yelp-train.csv")
    parser.add_argument("--yelp_dev", type=str, default="data/ids-yelp-dev.csv")
    parser.add_argument("--yelp_test", type=str, default="data/ids-yelp-dev.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--cfimdb_dev_out", type=str, default="predictions/cfimdb-dev-output.csv")
    parser.add_argument("--cfimdb_test_out", type=str, default="predictions/cfimdb-test-output.csv")

    parser.add_argument("--yelp_dev_out", type=str, default="predictions/yelp-dev-output.csv")
    parser.add_argument("--yelp_test_out", type=str, default="predictions/yelp-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    # use extensions
    parser.add_argument("--use_pcgrad", action='store_true')
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--use_imbalance", action='store_true')
    parser.add_argument("--load_pretrained", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    if args.option == 'finetune':
        test_model(args)