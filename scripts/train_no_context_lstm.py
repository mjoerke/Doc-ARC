import os,sys,argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import AdamW

import metrics
from data_utils import SequenceLabelingDataset, read_tagset
from torch.utils.data import DataLoader
from datasets import *

import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class BertForSequenceLabeling(nn.Module):

    def __init__(self, model_dir, config, freeze_bert=False, lstm_hidden_dim=128, num_lstm=1):
        super(BertForSequenceLabeling, self).__init__()
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained(model_dir, config=config)

        self.BERT_DIM = config.hidden_size
        self.LSTM_HIDDEN_DIM = lstm_hidden_dim

        self.freeze_bert = freeze_bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.num_lstm = num_lstm

        if num_lstm == 1: #dynamic baseline
            input_dim = self.BERT_DIM
        else:
            input_dum = 4 * self.BERT_DIM

        self.lstm1 = nn.LSTM(input_dim, self.LSTM_HIDDEN_DIM, bidirectional=True, batch_first=True)
        
        if num_lstm == 2:
            self.lstm2 = nn.LSTM(2 * self.LSTM_HIDDEN_DIM, self.LSTM_HIDDEN_DIM, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * self.LSTM_HIDDEN_DIM, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None, labels=None):
        self.train()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        if labels is not None:
            labels = labels.to(device)

        last_state, pooled_states, hidden_states = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask)

        if self.num_lstm == 1:
            out = torch.matmul(transforms, last_state)
        else:
            all_layers = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]), -1)
            out = torch.matmul(transforms, all_layers)

        out, _ = self.lstm1(out)

        if self.num_lstm == 2:
            out, _ = self.lstm2(out)

        out = self.dropout(out)

        logits = self.classifier(out)
        
        if labels is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def evaluate(self, data_loader, batch_size, metric, tagset):

        self.eval()

        pred = []
        true = []

        with torch.no_grad():

            for document_batch in data_loader:

                # unpack batch size of 1
                document = document_batch[0]
                batches = get_batches(document, batch_size, tagset)
                num_batches = len(batches['inputs'])

                for b in range(num_batches):

                    inputs = batches['inputs'][b].to(device)
                    masks = batches['masks'][b].to(device)
                    transforms = batches['transforms'][b].to(device)
                    labels = batches['labels'][b]

                    logits = self.forward(input_ids=inputs, attention_mask=masks, transforms=transforms)

                    batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                    bs, msl = batch_predictions.shape

                    for i in range(bs):
                        tags = batches['labels'][b][i]
                        preds = batch_predictions[i]

                        for j in range(msl):
                            if tags[j] == -100:
                                continue

                            pred.append(int(preds[j]))
                            true.append(int(tags[j]))

        return metric(true, pred, tagset)

    def write_predictions(self, output_file, data_loader, tagset):
        self.eval()

        rev_tagset = {v: k for k,v in tagset.items()}
        pred_file = os.path.join(output_file)

        with open(pred_file, 'w+', encoding='utf8') as f:
            with torch.no_grad():

                for document_batch in data_loader:
                    # unpack batch size of 1
                    document = document_batch[0]
                    doc_id = document.path.split("/")[-1]
                    f.write(doc_id + "\n")
                    batches = get_batches(document, batch_size=32, tagset=tagset)
                    num_batches = len(batches['inputs'])

                    ordering = batches['ordering']
                    seq_idx = 0

                    for b in range(num_batches):
                        inputs = batches['inputs'][b].to(device)
                        masks = batches['masks'][b].to(device)
                        transforms = batches['transforms'][b].to(device)
                        labels = batches['labels'][b]

                        logits = self.forward(input_ids=inputs, attention_mask=masks, transforms=transforms)

                        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                        bs, msl = batch_predictions.shape

                        for i in range(bs):
                            masked_tags = batches['labels'][b][i]
                            masked_preds = batch_predictions[i]
                            seq = document[ordering[seq_idx]]

                            labels = []
                            preds = []
                            words = seq.words[1:-1]

                            for j in range(msl):
                                if masked_tags[j] == -100:
                                    continue

                                preds.append(int(masked_preds[j]))
                                labels.append(int(masked_tags[j]))

                            for word, pred, true in zip(words, preds, labels):
                                out = word + '\t' + rev_tagset[pred] + '\t' + rev_tagset[true] + "\n"
                                f.write(out)

                            seq_idx += 1


def get_batches(document, batch_size, tagset):
    sequences = [s for s in document.sequences if s.is_labeled]
    N = len(sequences)

    if N % batch_size == 0:
        num_batches = N // batch_size
    else:
        num_batches = N // batch_size + 1

    ordering = np.random.permutation(N)

    batched_data = [] # (batch_size x max_seq_len)
    batched_masks = [] # (batch_size x max_seq_len)
    batched_labels = [] # (batch_size x max_sent_len)
    batched_transforms = [] # (batch_size x max_seq_len x max_sent_len)
    # max_sent_len == len in words
    # max_seq_len == len in tokens

    max_sequence_lengths = []
    max_label_lengths = []

    for b in range(num_batches):
        start = b * batch_size
        stop = min((b+1) * batch_size, len(sequences))

        seqs = [sequences[ordering[i]] for i in range(start, stop)]

        max_seq_len = max([len(seq.token_ids) for seq in seqs])
        max_sequence_lengths.append(max_seq_len)

        max_label_len = max([len(seq.get_label_ids(tagset)) for seq in seqs])
        max_label_lengths.append(max_label_len)

        data = np.zeros((stop-start, max_seq_len))
        masks = np.zeros((stop-start, max_seq_len))
        labels = np.zeros((stop-start, max_label_len))
        transforms = np.zeros((stop-start, max_label_len, max_seq_len))

        for i in range(stop-start):
            s = seqs[i]
            n_tokens = len(s.token_ids)
            n_labels = len(s.label_ids)

            data[i, :n_tokens] = s.token_ids
            masks[i, :n_tokens] = [1.0] * n_tokens

            labels[i:, :n_labels] = s.label_ids
            labels[i, n_labels:] = (max_label_len - n_labels) * [-100]

            transforms[i, :n_labels, :n_tokens] = s.transform

        batched_data.append(torch.LongTensor(data))
        batched_masks.append(torch.FloatTensor(masks))
        batched_labels.append(torch.LongTensor(labels))
        batched_transforms.append(torch.FloatTensor(transforms))

    return {"inputs": batched_data,
            "masks": batched_masks,
            "labels": batched_labels,
            "transforms": batched_transforms,
            "max_sequence_lengths": max_sequence_lengths,
            "ordering": ordering, 
            "sequences": sequences}

def load_model(model_type, pretrained_dir, checkpoint_file, num_labels, freeze_bert, lstm_dim, num_lstm):
    assert not (pretrained_dir and checkpoint_file), "Can't load pretrained base model and checkpointed weights"

    if pretrained_dir:
        # preload config and model files from pretrained_dir
        # use this to load domain fine-tuned model before task fine-tuning
        config = AutoConfig.from_pretrained(pretrained_dir)
        config.output_hidden_states = True
        config.num_labels = num_labels

        model = BertForSequenceLabeling(pretrained_dir, config=config, freeze_bert=freeze_bert, 
                                                        lstm_hidden_dim=lstm_dim, num_lstm=num_lstm)
    else:
        # default to pre-trained version
        config = AutoConfig.from_pretrained(model_type)
        config.output_hidden_states = True
        config.num_labels = num_labels

        model = BertForSequenceLabeling(model_type, config=config, freeze_bert=freeze_bert, 
                                                    lstm_hidden_dim=lstm_dim, num_lstm=num_lstm)

        if checkpoint_file:
            model.load_state_dict(torch.load(checkpoint_file))

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='{train,test}', required=True)
    parser.add_argument('--output_dir', help='File to write model checkpoints and log files to', required=False)
    parser.add_argument('--dataset', help='{genia, genia_full, litbank, litbank_full, ontonotes}', required=True)
    parser.add_argument('--metric', help='{accuracy,fscore,span_fscore}', default='span_fscore', required=False)

    parser.add_argument('--pretrained_dir', help='Directory to read MLM pretrained BERT base model weights from', required=False)
    parser.add_argument('--checkpoint_file', help='File to read full checkpointed model weights from (e.g. to resume training or test)', required=False)
    parser.add_argument('--model_type', help='Pretrained BERT configuration checkpoint, e.g. bert-base-cased', required=True)

    parser.add_argument("--batch_size", default=16, type=int, help="The batch size on GPU.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, 
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--lstm_dim", default=128, type=int, help="LSTM hidden dimension size.")
    parser.add_argument("--num_lstm", default=1, type=int, help="How many LSTMs to stack on BERT base.")
    parser.add_argument('--lr', type=float, default=2e-5, required=False)
    parser.add_argument('--num_epochs', type=int, default=100, required=False) 
    
    parser.add_argument('--freeze_bert', help='Whether to freeze BERT base', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 
                "training.log" if args.mode == 'train' else 'test.log'), mode='w+'),
            logging.StreamHandler()
        ]
    )

    logging.info("Running on: {}".format(device))
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    if args.dataset == 'genia':
        dataset = GENIA
    elif args.dataset == 'litbank':
        dataset = LITBANK
    elif args.dataset == 'ontonotes':
        dataset = ONTONOTES
    else:
        raise ValueError("Invalid dataset")

    if 'google' in args.model_type:
        tokenizer_type = 'bert-base-uncased'
    else:
        tokenizer_type = args.model_type

    do_lower_case = 'uncased' in tokenizer_type
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, do_lower_case=do_lower_case, do_basic_tokenize=False)

    metric = None
    if args.metric.lower() == "fscore":
        metric = metrics.check_f1_two_lists
    elif args.metric.lower() == "accuracy":
        metric = metrics.get_accuracy
    elif args.metric.lower() == "span_fscore":
        metric = metrics.check_span_f1_two_lists

    tagset = read_tagset(dataset['tagset'])

    model = load_model(args.model_type, args.pretrained_dir, args.checkpoint_file, len(tagset), args.freeze_bert, args.lstm_dim, args.num_lstm)
    model.to(device)

    mode = args.mode
        
    if mode == 'train':

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-6)
        loss_function = nn.CrossEntropyLoss(ignore_index=-100)

        train_documents = SequenceLabelingDataset(dataset['train_dir'], tokenizer)
        train_data_loader = DataLoader(train_documents, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: x)

        dev_documents = SequenceLabelingDataset(dataset['dev_dir'], tokenizer)
        dev_data_loader = DataLoader(dev_documents, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: x)

        global_steps = 0 # number of backward passes
        steps = 0 # number of forward passes (not divided by accummulation)
        total_loss = 0
        best_val = -1
        best_idx = 0
        patience = 3

        for epoch in range(args.num_epochs):

            logging.info("*** TRAINING ****\n")
            logging.info("Epoch: {}".format(epoch))
            total_loss = 0

            for i, document_batch in enumerate(train_data_loader):

                document = document_batch[0]
                document_loss = 0

                batches = get_batches(document, args.batch_size, tagset)

                num_batches = len(batches['inputs'])

                for b in range(num_batches):
                    inputs = batches['inputs'][b].to(device)
                    transforms = batches['transforms'][b].to(device)
                    masks = batches['masks'][b].to(device)
                    labels = batches['labels'][b].to(device)

                    loss = model.forward(input_ids=inputs, attention_mask=masks, transforms=transforms, labels=labels)

                    loss /= args.gradient_accumulation_steps
                    document_loss += loss.item()
                    steps += 1

                    if steps % args.gradient_accumulation_steps == 0:
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        model.zero_grad()
                        global_steps += 1

                        if global_steps % 100 == 0:
                            logging.info("Global step: {}".format(global_steps))

                num_labeled = sum([len(batches['inputs'][b]) for b in range(num_batches)])
                logging.info("Document {}/{}, (len={}) loss: {}".format(i, len(train_documents), num_labeled, document_loss))
                total_loss += document_loss

            logging.info("Epoch total loss: {}".format(total_loss))
            logging.info("*** EVALUATING ***\n")
            value = model.evaluate(dev_data_loader, args.batch_size, metric, tagset)
            logging.info("DEV {}: {}".format(args.metric, value))

            if value > best_val:
                best_idx = epoch
                best_val = value

                model_dir = os.path.join(args.output_dir, "checkpoint-{}.bin".format(best_idx))
                logging.info("Saving model @ {}".format(model_dir))
                torch.save(model.state_dict(), model_dir)
                
                # model.save_pretrained(args.output_dir)

            elif (epoch - best_idx) > patience:
                logging.info("Aborting training after {} epochs of patience".format(patience))
                logging.info("Best model @ epoch {} with {}={}".format(best_idx, args.metric, best_val))
                break

        del model # allows torch to free gpu memory before loading best model from disk

        logging.info("*** TESTING ***\n")

        best_model_dir = os.path.join(args.output_dir, "checkpoint-{}.bin".format(best_idx))
        logging.info("Loading best model from {}".format(best_model_dir))
        best_model = load_model(args.model_type, None, best_model_dir, len(tagset), args.freeze_bert, args.lstm_dim, args.num_lstm)

        bert_output_dir = os.path.join(args.output_dir, "bert")
        os.makedirs(bert_output_dir, exist_ok=True)

        logging.info("Saving BERT base model weights @ {}".format(bert_output_dir))
        best_model.bert.save_pretrained(bert_output_dir)
        best_model.to(device)

        test_documents = SequenceLabelingDataset(dataset['test_dir'], tokenizer)
        test_data_loader = DataLoader(test_documents, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: x)

        value = best_model.evaluate(test_data_loader, args.batch_size, metric, tagset)
        logging.info("TEST {}: {}".format(args.metric, value))

        return


    elif mode == 'predict':
        prediction_file = os.path.join(args.output_dir, "predictions.txt")

        # TODO
        pass


    elif mode == 'test':
        logging.info("*** TESTING ***\n")

        test_documents = SequenceLabelingDataset(dataset['test_dir'], tokenizer)
        test_data_loader = DataLoader(test_documents, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: x)

        value = model.evaluate(test_data_loader, args.batch_size, metric, tagset)
        logging.info("TEST {}: {}".format(args.metric, value))

if __name__ == "__main__":
    main()
