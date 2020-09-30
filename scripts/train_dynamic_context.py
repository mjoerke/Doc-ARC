import os,sys,argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim

import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer, AdamW

import metrics
from data_utils import DocumentAttentionDataset, read_tagset
from torch.utils.data import DataLoader
from datasets import *

import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

DROPOUT_RATE = 0.25
BERT_DIM = 768

class BertWithAttention(nn.Module):

    def __init__(self, model_dir, config, freeze_bert=False, lstm_hidden_dim=128, vanilla=False):
        super(BertWithAttention, self).__init__()

        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained(model_dir, config=config)

        self.BERT_DIM = config.hidden_size
        self.LSTM_HIDDEN_DIM = lstm_hidden_dim

        self.freeze_bert = freeze_bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.vanilla = vanilla
        self.dist_embeddings = nn.Embedding(9, 20)
        self.dropout = nn.Dropout(DROPOUT_RATE)

        if not vanilla:
            self.context_encoder = nn.LSTM(self.BERT_DIM, self.LSTM_HIDDEN_DIM, bidirectional=True, batch_first=True)
            self.attention_encoder = nn.LSTM(4 * self.LSTM_HIDDEN_DIM + 20, self.LSTM_HIDDEN_DIM, bidirectional=True, batch_first=True)
            self.attention_weights = nn.Linear(2 * self.LSTM_HIDDEN_DIM + 20, 1)
        else:
            self.attention_encoder = nn.LSTM(2 * self.BERT_DIM + 20, self.LSTM_HIDDEN_DIM, bidirectional=True, batch_first=True)
            self.attention_weights = nn.Linear(self.BERT_DIM + 20, 1)

        self.classifier = nn.Linear(2 * self.LSTM_HIDDEN_DIM, self.num_labels)

    def forward_bert(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None):
        last_state, pooled_states, hidden_states = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask)
        # all_layers = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]), -1)
        out = torch.matmul(transforms, last_state)
        return out

    def forward(self, inputs, masks, transforms, context_inputs, context_masks, context_transforms,
                      attn_sentence_idx, attn_word_idx, attn_dists, attn_mask):

        sentence_representations = self.forward_bert(inputs, attention_mask=masks, transforms=transforms)

        if not self.vanilla:
            sentence_representations, _ = self.context_encoder(sentence_representations)
        # sentence_representations = torch.zeros(sentence_representations.shape).to(device)
        B, L, K = attn_sentence_idx.shape

        context_representations = self.forward_bert(context_inputs, attention_mask=context_masks, transforms=context_transforms)

        if not self.vanilla:
            context_representations, _ = self.context_encoder(context_representations)
        # print(context_representations.shape)
        context_representations = context_representations[attn_sentence_idx.view(B * L * K), attn_word_idx.view(B * L * K)]
        context_representations = context_representations.view(B * L, K, -1)

        dist_embeds = self.dist_embeddings(attn_dists.view(B * L, K))
        attention = torch.cat((context_representations, dist_embeds), -1)

        attention_weights = self.attention_weights(attention)
        attention_weights = torch.exp(attention_weights).squeeze()
        attention_weights = attention_weights * attn_mask.view(B * L, K)

        weight_sum = torch.sum(attention_weights, dim=1).unsqueeze(-1) + 1e-10
        attention_weights = attention_weights / weight_sum

        attention = attention.permute((0, 2, 1))

        attention = torch.matmul(attention, attention_weights.unsqueeze(-1)) 

        attention = attention.squeeze()

        merged = torch.cat((sentence_representations, attention.view(B, L, -1)), -1)
        merged, _ = self.attention_encoder(merged)
        # out, _ = self.attention_encoder(attention.view(B, L, -1))
        out = self.dropout(merged)
        logits = self.classifier(out)
        return logits

    def evaluate(self, data_loader, batch_size, metric, tagset, k):

        self.eval()

        pred = []
        true = []

        with torch.no_grad():

            for document_batch in data_loader:

                # unpack batch size of 1
                document = document_batch[0]
                batches = get_batches(document, batch_size, tagset, k)
                num_batches = len(batches['inputs'])

                for b in range(num_batches):

                    inputs = batches['inputs'][b].to(device)
                    transforms = batches['transforms'][b].to(device)
                    masks = batches['masks'][b].to(device)
                    labels = batches['labels'][b].to(device)

                    attn_sentence_idx = batches['attn_sentence_idx'][b].to(device)
                    attn_word_idx = batches['attn_word_idx'][b].to(device)
                    attn_dists = batches['attn_dists'][b].to(device)
                    attn_mask = batches['attn_masks'][b].to(device)

                    context_inputs = batches['context_inputs'][b].to(device)
                    context_masks = batches['context_masks'][b].to(device)
                    context_transforms = batches['context_transforms'][b].to(device)

                    logits = self.forward(inputs=inputs, masks=masks, transforms=transforms,
                                            context_inputs=context_inputs, context_masks=context_masks, 
                                            context_transforms=context_transforms,
                                            attn_sentence_idx=attn_sentence_idx, attn_word_idx=attn_word_idx,
                                            attn_dists=attn_dists, attn_mask=attn_mask)

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

    def write_predictions(self, output_file, data_loader, tagset, k):
        self.eval()

        rev_tagset = {v: k for k,v in tagset.items()}
        pred_file = os.path.join(output_file)

        with open(pred_file, 'w+', encoding='utf8') as f:
            
            with torch.no_grad():

                for document_batch in data_loader:

                    # unpack batch size of 1
                    document = document_batch[0]
                    doc_id = document.path.split("/")[-1]
                    print(doc_id)
                    f.write(doc_id + "\n")

                    batches = get_batches(document, 16, tagset, k)
                    num_batches = len(batches['inputs'])

                    ordering = batches['ordering']
                    seq_idx = 0

                    for b in range(num_batches):

                        inputs = batches['inputs'][b].to(device)
                        transforms = batches['transforms'][b].to(device)
                        masks = batches['masks'][b].to(device)
                        labels = batches['labels'][b].to(device)

                        attn_sentence_idx = batches['attn_sentence_idx'][b].to(device)
                        attn_word_idx = batches['attn_word_idx'][b].to(device)
                        attn_dists = batches['attn_dists'][b].to(device)
                        attn_mask = batches['attn_masks'][b].to(device)

                        context_inputs = batches['context_inputs'][b].to(device)
                        context_masks = batches['context_masks'][b].to(device)
                        context_transforms = batches['context_transforms'][b].to(device)

                        logits = self.forward(inputs=inputs, masks=masks, transforms=transforms,
                                                context_inputs=context_inputs, context_masks=context_masks, 
                                                context_transforms=context_transforms,
                                                attn_sentence_idx=attn_sentence_idx, attn_word_idx=attn_word_idx,
                                                attn_dists=attn_dists, attn_mask=attn_mask)

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

# distance buckets from https://arxiv.org/pdf/1707.07045.pdf
# [0, 1, 2, 3,   4,    5,     6,     7,   8]
# [1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+]
def bucket_dist(dist):
    if dist <= 4:
        return dist
    elif dist >= 5 and dist < 8:
        return 4
    elif dist >= 8 and dist < 16:
        return 5
    elif dist >= 16 and dist < 32:
        return 6
    elif dist >= 32 and dist < 64:
        return 7
    elif dist >= 64:
        return 8
    else:
        print(dist)
        print("problem!")
        sys.exit(1)

def get_batches(document, batch_size, tagset, num_attended):
    sequences = [s for s in document.sequences if s.is_labeled]
    N = len(sequences)

    if N % batch_size == 0:
        num_batches = N // batch_size
    else:
        num_batches = N // batch_size + 1

    ordering = np.random.permutation(N)

    batched_inputs = [] # (batch_size x max_seq_len)
    batched_masks = [] # (batch_size x max_seq_len)
    batched_labels = [] # (batch_size x max_sent_len)
    batched_transforms = [] # (batch_size x max_seq_len x max_sent_len)
    # max_sent_len == len in words
    # max_seq_len == len in tokens
    batched_attn_sentence_idx = [] #(batch_size x max_sent_len x k) 
    batched_attn_word_idx = [] #(batch_size x max_sent_len x k)
    batched_attn_dists = [] #(batch_size x max_sent_len x k)
    batched_attn_masks = [] #(batch_size x max_sent_len x k)

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

        inputs = np.zeros((stop-start, max_seq_len))
        masks = np.zeros((stop-start, max_seq_len))
        labels = np.zeros((stop-start, max_label_len))
        transforms = np.zeros((stop-start, max_label_len, max_seq_len))

        attn_sentence_idx = np.zeros((stop-start, max_label_len, num_attended))
        attn_word_idx = np.zeros((stop-start, max_label_len, num_attended))
        attn_dists = np.zeros((stop-start, max_label_len, num_attended))
        attn_masks = np.zeros((stop-start, max_label_len, num_attended))

        for i in range(stop-start):
            s = seqs[i]
            n_tokens = len(s.token_ids)
            n_labels = len(s.label_ids)

            inputs[i, :n_tokens] = s.token_ids
            masks[i, :n_tokens] = [1.0] * n_tokens

            labels[i:, :n_labels] = s.label_ids
            labels[i, n_labels:] = (max_label_len - n_labels) * [-100]

            transforms[i, :n_labels, :n_tokens] = s.transform

            s_attn =  s.attn_seq_idxs
            w_attn = s.attn_word_idxs
            d_attn = s.attn_dists

            for j in range(len(s_attn)):
                n_context = len(s_attn[j])

                for k in range(n_context):
                    # attn_sentence_idx[i, j, k] = document.attn_index_map[s_attn[j][k]]
                    attn_sentence_idx[i, j, k] = s_attn[j][k]
                    attn_word_idx[i, j, k] = w_attn[j][k]
                    attn_dists[i, j, k] = bucket_dist(d_attn[j][k])
                    attn_masks[i, j, k] = 1

        batched_inputs.append(torch.LongTensor(inputs))
        batched_masks.append(torch.FloatTensor(masks))
        batched_labels.append(torch.LongTensor(labels))
        batched_transforms.append(torch.FloatTensor(transforms))
        # don't make torch.LongTensor until the indices have been mapped
        batched_attn_sentence_idx.append(attn_sentence_idx)
        batched_attn_word_idx.append(torch.LongTensor(attn_word_idx))
        batched_attn_dists.append(torch.LongTensor(attn_dists))
        batched_attn_masks.append(torch.FloatTensor(attn_masks))

    # for each batch, compute which sentences will be attended over and
    # batch them appropriately

    batched_context_inputs = []
    batched_context_masks = []
    batched_context_transforms = []

    for b in range(num_batches):
        attended_sequences = np.unique(batched_attn_sentence_idx[b])
        attended_sequences = [int(s) for s in attended_sequences]
        context_sequences = [document[i] for i in attended_sequences]

        num_attended_sequences = len(attended_sequences)
        max_context_sequence_len = max([len(seq.token_ids) for seq in context_sequences])
        # length in # words after transform
        max_context_sentence_len = max([len(seq.words) for seq in context_sequences])

        context_inputs = np.zeros((num_attended_sequences, max_context_sequence_len))
        context_masks = np.zeros((num_attended_sequences, max_context_sequence_len))
        context_transforms = np.zeros((num_attended_sequences, max_context_sentence_len, max_context_sequence_len))

        # get all context sequences and populate input/mask/transform arrays
        for i in range(num_attended_sequences):
            s = context_sequences[i]
            n_tokens = len(s.token_ids)
            n_words = len(s.words)

            context_inputs[i, :n_tokens] = s.token_ids
            context_masks[i, :n_tokens] = [1.0] * n_tokens
            context_transforms[i, :n_words, :n_tokens] = s.transform

        # re-assigned batched_attn_sentence_idx to from global (document) index to local (batch) index
        B_, L_, K_ = batched_attn_sentence_idx[b].shape
        for b_ in range(B_):
            for l_ in range(L_):
                for k_ in range(K_):
                    og_idx = int(batched_attn_sentence_idx[b][b_, l_, k_])
                    batched_attn_sentence_idx[b][b_, l_, k_] = attended_sequences.index(og_idx)

        batched_attn_sentence_idx[b] = torch.LongTensor(batched_attn_sentence_idx[b])
        batched_context_inputs.append(torch.LongTensor(context_inputs))
        batched_context_masks.append(torch.FloatTensor(context_masks))
        batched_context_transforms.append(torch.FloatTensor(context_transforms))


    return {"inputs": batched_inputs,
            "masks": batched_masks,
            "transforms": batched_transforms,
            "labels": batched_labels,
            "attn_sentence_idx": batched_attn_sentence_idx,
            "attn_word_idx": batched_attn_word_idx,
            "attn_dists": batched_attn_dists,
            "attn_masks": batched_attn_masks,
            "max_sequence_lengths": max_sequence_lengths,
            "max_label_lengths": max_label_lengths,
            "context_inputs": batched_context_inputs,
            "context_masks": batched_context_masks,
            "context_transforms": batched_context_transforms,
            "ordering": ordering}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def load_model(model_type, pretrained_dir, checkpoint_file, num_labels, freeze_bert, lstm_dim, vanilla):
    assert not (pretrained_dir and checkpoint_file), "Can't load pretrained base model and checkpointed weights"


    if pretrained_dir:
        # preload config and model files from pretrained_dir
        # use this to load domain fine-tuned model before task fine-tuning
        config = AutoConfig.from_pretrained(pretrained_dir)
        config.output_hidden_states = True
        config.num_labels = num_labels

        model = BertWithAttention(pretrained_dir, config=config, freeze_bert=freeze_bert, 
                                                  lstm_hidden_dim=lstm_dim, vanilla=vanilla)
    else:
        # default to pre-trained version
        config = AutoConfig.from_pretrained(model_type)
        config.output_hidden_states = True
        config.num_labels = num_labels

        model = BertWithAttention(model_type, config=config, freeze_bert=freeze_bert, 
                                              lstm_hidden_dim=lstm_dim, vanilla=vanilla)

        if checkpoint_file:
            model.load_state_dict(torch.load(checkpoint_file))

    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='{train,test}', required=True)
    parser.add_argument('--output_dir', help='File to write model checkpoints and log files to', required=False)
    parser.add_argument('--dataset', help='{genia, genia_full, litbank, litbank_full, ontonotes}', required=True)
    parser.add_argument('--metric', help='{accuracy,fscore,span_fscore}', default='span_fscore', required=False)

    parser.add_argument("--batch_size", default=16, type=int, help="The batch size on GPU.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, 
                        help="Number of updates steps to accumulate before performing a backward pass.")

    parser.add_argument('--freeze_bert', help='Whether to freeze BERT weights', action='store_true')
    parser.add_argument('--self_attention', help='Whether sequences should be allowed to attention to themsleves', action='store_true')
    parser.add_argument('--vanilla', help='Whether to add LSTM encoders to model', action='store_true')
    parser.add_argument("--lstm_dim", default=128, type=int, help="LSTM hidden dimension size.")

    parser.add_argument('--pretrained_dir', help='Directory to read custom fine-tuned (BERT) base model weights from', required=False)
    parser.add_argument('--checkpoint_file', help='File to read checkpointed model weights from (to resume training or test)', required=False)
    parser.add_argument('--model_type', help='Pretrained BERT configuration checkpoint, e.g. bert-base-cased', required=True)

    parser.add_argument('--lr', type=float, default=1e-4, required=False)
    parser.add_argument('--num_epochs', type=int, default=20, required=False)
    # parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1, required=False)
    # parser.add_argument('--batch_size', help='GPU batch size', type=int, default=16, required=False)
    parser.add_argument('--k', type=int, help='How many context sequences to attend over', default=10, required=False)

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
    elif args.dataset == 'genia_full':
        dataset = GENIA_FULL
    elif args.dataset == 'litbank':
        dataset = LITBANK
    elif args.dataset == 'litbank_full':
        dataset = LITBANK_FULL
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

    model = load_model(args.model_type, args.pretrained_dir, args.checkpoint_file, len(tagset), args.freeze_bert, args.lstm_dim, args.vanilla)
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

        train_documents = DocumentAttentionDataset(dataset['train_dir'], tokenizer, args.k, args.self_attention)
        train_data_loader = DataLoader(train_documents, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: x)

        dev_documents = DocumentAttentionDataset(dataset['dev_dir'], tokenizer, args.k, args.self_attention)
        dev_data_loader = DataLoader(dev_documents, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: x)

        global_steps = 0
        steps = 0
        total_loss = 0
        best_val = -1
        best_idx = 0
        patience = 3

        for epoch in range(args.num_epochs):

            logging.info("*** TRAINING ****\n")
            logging.info("Epoch: {}".format(epoch))
            total_loss = 0

            for i, document_batch in enumerate(train_data_loader):

                # unpack batch size of 1
                document = document_batch[0]
                # logging.info("Training on document {}".format(document.path))
                document_loss = 0

                batches = get_batches(document, args.batch_size, tagset, args.k)

                num_batches = len(batches['inputs'])
                num_labeled = sum([len(batches['inputs'][b]) for b in range(num_batches)])
                num_attn = sum([np.prod(batches['attn_sentence_idx'][b].shape) for b in range(num_batches)])

                logging.info("Document {}/{}: (len={}, attn={}, batches={})".format(i+1, len(train_documents), num_labeled, num_attn, num_batches))

                model.train()

                for b in range(num_batches):

                    inputs = batches['inputs'][b].to(device)
                    transforms = batches['transforms'][b].to(device)
                    masks = batches['masks'][b].to(device)
                    labels = batches['labels'][b].to(device)

                    attn_sentence_idx = batches['attn_sentence_idx'][b].to(device)
                    attn_word_idx = batches['attn_word_idx'][b].to(device)
                    attn_dists = batches['attn_dists'][b].to(device)
                    attn_mask = batches['attn_masks'][b].to(device)

                    context_inputs = batches['context_inputs'][b].to(device)
                    context_masks = batches['context_masks'][b].to(device)
                    context_transforms = batches['context_transforms'][b].to(device)

                    logits = model.forward(inputs=inputs, masks=masks, transforms=transforms,
                                            context_inputs=context_inputs, context_masks=context_masks, 
                                            context_transforms=context_transforms,
                                            attn_sentence_idx=attn_sentence_idx, attn_word_idx=attn_word_idx,
                                            attn_dists=attn_dists, attn_mask=attn_mask)

                    loss = loss_function(logits.view(-1, model.num_labels), labels.view(-1))
                    document_loss += loss.item()

                    loss /= args.gradient_accumulation_steps
                    steps += 1

                    if steps % args.gradient_accumulation_steps == 0:
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        model.zero_grad()
                        global_steps += 1

                        if global_steps % 100 == 0:
                            logging.info("Global step: {}".format(global_steps))

                logging.info("Loss: {}".format(document_loss / num_labeled))
                total_loss += document_loss

            logging.info("Epoch total loss: {}".format(total_loss))
            logging.info("*** EVALUATING ***\n")
            value = model.evaluate(dev_data_loader, args.batch_size, metric, tagset, args.k)
            logging.info("DEV {}: {}".format(args.metric, value))

            if value > best_val:
                best_idx = epoch
                best_val = value

                model_dir = os.path.join(args.output_dir, "checkpoint-{}.bin".format(best_idx))
                logging.info("Saving model @ {}".format(model_dir))
                torch.save(model.state_dict(), model_dir)

            elif (epoch - best_idx) > patience:
                logging.info("Aborting training after {} epochs of patience".format(patience))
                logging.info("Best model @ epoch {} with {}={}".format(best_idx, args.metric, best_val))
                del model # allows torch to free gpu memory before loading best model from disk
                break

        logging.info("*** TESTING ***\n")

        best_model_dir = os.path.join(args.output_dir, "checkpoint-{}.bin".format(best_idx))
        logging.info("Loading best model from {}".format(best_model_dir))
        best_model = load_model(args.model_type, None, best_model_dir, len(tagset), args.freeze_bert, args.lstm_dim, args.vanilla)
        best_model.to(device)
        
        test_documents = DocumentAttentionDataset(dataset['test_dir'], tokenizer, args.k, args.self_attention)
        test_data_loader = DataLoader(test_documents, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: x)

        value = best_model.evaluate(test_data_loader, args.batch_size, metric, tagset, args.k)
        logging.info("TEST {}: {}".format(args.metric, value))

        return


    elif mode == 'predict':
        prediction_file = os.path.join(args.output_dir, "predictions.txt")

        # TODO
        pass


    elif mode == 'test':
        logging.info("*** TESTING ***\n")

        test_documents = DocumentAttentionDataset(dataset['test_dir'], tokenizer, args.k, args.self_attention)
        test_data_loader = DataLoader(test_documents, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: x)

        value = model.evaluate(test_data_loader, args.batch_size, metric, tagset, args.k)
        logging.info("TEST {}: {}".format(args.metric, value))

if __name__ == "__main__":
    main()
