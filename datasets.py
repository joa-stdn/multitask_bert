#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions to load the 3 datasets we're using.
You should only need to call load_multitask_data to get the training and dev examples
to train your model.
'''


import csv

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer

from constants import IMBALANCE_MULTIPLIER, MAX_SENTIMENT_SAMPLES, MAX_SIMILARITY_SAMPLES, MAX_CFIMDB_SAMPLES, MAX_PARAPHRASE_SAMPLES


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression =True):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])


        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'sent_ids': sent_ids
            }

        return batched_data

class MultiTaskDataset:
    def __init__(self, sentiment_dataset, paraphrase_dataset, similarity_dataset, cfimdb_dataset, yelp_dataset,  args):
        self.p = args
        self.sentiment_dataset = sentiment_dataset
        self.paraphrase_dataset = paraphrase_dataset
        self.similarity_dataset = similarity_dataset
        self.cfimdb_dataset = cfimdb_dataset
        self.yelp_dataset = yelp_dataset


    def __len__(self):
        return max(len(self.sentiment_dataset), len(self.paraphrase_dataset), len(self.similarity_dataset), len(self.cfimdb_dataset), len(self.yelp_dataset))
    
    def __getitem__(self, idx):
            #paraphrase_item = self.paraphrase_dataset.__getitem__(idx%len(self.paraphrase_dataset))
            yelp_item = self.yelp_dataset.__getitem__(idx%len(self.yelp_dataset))
            if self.p.use_imbalance:
                if idx//len(self.sentiment_dataset) <= IMBALANCE_MULTIPLIER:
                    sentiment_item = self.sentiment_dataset.__getitem__(idx%len(self.sentiment_dataset))
                    cfimdb_item = self.cfimdb_dataset.__getitem__(idx%len(self.cfimdb_dataset))
                    paraphrase_item = self.paraphrase_dataset.__getitem__(idx%len(self.paraphrase_dataset))
                else:
                    sentiment_item = None
                    cfimdb_item = None
                    paraphrase_item = None
                if idx//len(self.similarity_dataset) <= IMBALANCE_MULTIPLIER:
                    similarity_item = self.similarity_dataset.__getitem__(idx%len(self.similarity_dataset))
                else:
                    similarity_item = None
            else:
                sentiment_item = self.sentiment_dataset.__getitem__(idx%len(self.sentiment_dataset))
                cfimdb_item = self.cfimdb_dataset.__getitem__(idx%len(self.cfimdb_dataset))
                paraphrase_item = self.paraphrase_dataset.__getitem__(idx%len(self.paraphrase_dataset))
                similarity_item = self.similarity_dataset.__getitem__(idx%len(self.similarity_dataset))
            
            return (paraphrase_item, sentiment_item, similarity_item, cfimdb_item, yelp_item)
    
    def collate_fn(self, all_data):
        yelp_data = [x[4] for x in all_data]
        if self.p.use_imbalance:
            sentiment_data = [x[1] for x in all_data if x[1] is not None][:MAX_SENTIMENT_SAMPLES]
            similarity_data = [x[2] for x in all_data if x[2] is not None][:MAX_SIMILARITY_SAMPLES]
            cfimdb_data = [x[3] for x in all_data if x[3] is not None][:MAX_CFIMDB_SAMPLES]
            #yelp_data = [x[4] for x in all_data if x[4] is not None][:MAX_YELP_SAMPLES]
            paraphrase_data = [x[0] for x in all_data if x[0] is not None][:MAX_PARAPHRASE_SAMPLES]
        else:
            sentiment_data = [x[1] for x in all_data]
            similarity_data = [x[2] for x in all_data]
            cfimdb_data = [x[3] for x in all_data]
            paraphrase_data = [x[0] for x in all_data]

        #paraphrase_collated = self.paraphrase_dataset.collate_fn(paraphrase_data)
        yelp_collated = self.yelp_dataset.collate_fn(yelp_data)
        if len(sentiment_data) > 0:
            sentiment_collated = self.sentiment_dataset.collate_fn(sentiment_data)
        else:
            sentiment_collated = None
        if len(similarity_data) > 0:
            similarity_collated = self.similarity_dataset.collate_fn(similarity_data)
        else:
            similarity_collated = None
        if len(cfimdb_data) > 0:
            cfimdb_collated = self.cfimdb_dataset.collate_fn(cfimdb_data)
        else:
            cfimdb_collated = None
        if len(paraphrase_data) > 0:
            paraphrase_collated = self.paraphrase_dataset.collate_fn(paraphrase_data)
        else:
            paraphrase_collated = None

        return {
            'paraphrase': paraphrase_collated,
            'sentiment': sentiment_collated,
            'similarity': similarity_collated,
            'cfimdb': cfimdb_collated,
            'yelp': yelp_collated
        }



def load_multitask_test_data():
    paraphrase_filename = f'data/quora-test.csv'
    sentiment_filename = f'data/ids-sst-test.txt'
    similarity_filename = f'data/sts-test.csv'

    sentiment_data = []

    with open(sentiment_filename, 'r') as fp:
        #index=0
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if index>100: break
            #index+=1
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r') as fp:
        #index=0
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if index>100: break
            #index+=1
            #if record['split'] != split:
            #    continue
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r') as fp:
        #index=0
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if index>100: break
            #index+=1
            similarity_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data
    


def load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,cfimdb_filename,yelp_filename, split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = '\t'):
                #if index>100: break
                #index+=1
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = '\t'):
                #if index>100: break
                #index+=1
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    sentiment_data2 = []
    num_labels2 = {}
    if split == 'test':
        with open(cfimdb_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = '\t'):
                #if index>100: break
                #index+=1
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data2.append((sent,sent_id))
    else:
        with open(cfimdb_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = '\t'):
                #if index>100: break
                #index+=1
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels2:
                    num_labels2[label] = len(num_labels2)
                sentiment_data2.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data2)} {split} examples from {cfimdb_filename}")

    sentiment_data3 = []
    num_labels3 = num_labels2
    if split == 'test':
        with open(yelp_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = ','):
                #if index>100: break
                #index+=1
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data3.append((sent,sent_id))
    else:
        with open(yelp_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = ','):
                #if index>100: break
                #index+=1
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels3:
                    num_labels3[label] = len(num_labels3)
                sentiment_data3.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data3)} {split} examples from {yelp_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = '\t'):
                #if index>100: break
                #index+=1
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = '\t'):
                #if index>100: break
                #index+=1
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            #index=0 
            for record in csv.DictReader(fp,delimiter = '\t'):
                #if index>100: break
                #index+=1
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            #index=0
            for record in csv.DictReader(fp,delimiter = '\t'):
                #if index>100: break
                #index+=1
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data, sentiment_data2, num_labels2, sentiment_data3
