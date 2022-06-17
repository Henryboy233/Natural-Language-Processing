import numpy as np
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from bokeh.plotting import figure, show
from bokeh.io import push_notebook, output_notebook
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show
from bokeh.io import push_notebook, output_notebook
from bokeh.models import ColumnDataSource, LabelSet
import pandas as pd
import torch
import collections
import numpy as np
from tqdm import tqdm
import os
import time
import logging

def display_closestwords_tsnescatterplot(model, word): 
    arr = np.empty((0,100), dtype='f')
    word_labels = [word] 
    # get close words
    close_words = model.similar_by_word(word)
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0) 
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]] 
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0) 
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    for label, x, y in zip(word_labels, x_coords, y_coords):
        if label == word: 
            plt.scatter(x, y, c='r')
        else:
            plt.scatter(x, y, c='g')
        plt.annotate(label, xy=(x, y), xytext=(5, -2), textcoords='offset points')
    plt.title(f'Words closest to: {word}')
    plt.show()


'''makes an interactive scatter plot with text labels for each point'''
def interactive_tsne(text_labels, tsne_array):
    # Define a dataframe to be used by bokeh context
    bokeh_df = pd.DataFrame(tsne_array, text_labels, columns=['x','y']) 
    bokeh_df['text_labels'] = bokeh_df.index
    # interactive controls to include to the plot

    TOOLS="hover, zoom_in, zoom_out, box_zoom, undo, redo, reset, box_select"
    p = figure(tools=TOOLS, plot_width=700, plot_height=700)
    # define data source for the plot
    source = ColumnDataSource(bokeh_df)
    # scatter plot
    p.scatter('x', 'y', source=source, fill_alpha=0.6, fill_color="#8724B5", line_color=None)
    # text labels
    labels = LabelSet(x='x', y='y', text='text_labels', y_offset=8, text_font_size="8pt", text_color="#555555", source=source, text_align='center')
    p.add_layout(labels)
    # show plot inline
    output_notebook()
    show(p)


def visual_wv(model, word_nums=200):
    tsne = TSNE(n_components=2, random_state=0)
    input_vocab = list(model.wv.key_to_index.keys())[:word_nums]
    points = len(input_vocab)
    X = model.wv[input_vocab]
    X_tsne = tsne.fit_transform(X[:points])
    interactive_tsne(list(input_vocab)[:points], X_tsne)




# Create split data
def split_by_jobtype(df, proportion):
    final_list = [] 
    np.random.seed(1234)

    by_jobtype = collections.defaultdict(list) 
    for _, row in df.iterrows():
        by_jobtype[row['job_type']].append(row.to_dict())
        
    for _, item_list in sorted(by_jobtype.items()): 
        np.random.shuffle(item_list)
        n_total = len(item_list)
        n_train = int(proportion[0] * n_total) 
        n_val = int(proportion[1] * n_total) 

        
        # Give data point a split attribute
        for item in item_list[:n_train]: 
            item['split'] = 'train'
        for item in item_list[n_train:n_train+n_val]: 
            item['split'] = 'val'
        for item in item_list[n_train+n_val:]: 
            item['split'] = 'test'
        final_list.extend(item_list)

    res = pd.DataFrame(final_list)
    return res

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100



def train_engin(args, model, train_dataloader, val_dataloader, train_state):
    
    optimizer = args.optimizer
    loss_func = args.loss_func
    model.to(args.device)

    for epoch_index in range(args.num_epochs):
        train_state['epoch_index'] = epoch_index

        running_loss, running_acc = 0.0 , 0.0 
        model.train()
        args.logger.info("TRAIN: {}|{}".format(epoch_index, args.num_epochs))
        for batch_index, batch_dict in tqdm(enumerate(train_dataloader)):
            text_vectors = batch_dict["batch_text_vectors"].to(args.device)
            labels = batch_dict["batch_labels"].to(args.device)

            optimizer.zero_grad()
            y_pred = model(text_vectors)
            loss = loss_func(y_pred, labels) 
            loss_batch = loss.item()

            running_loss += (loss_batch-running_loss) / (batch_index + 1) 
            loss.backward()
            optimizer.step()

            acc_batch = compute_accuracy(y_pred, labels) 
            running_acc += (acc_batch - running_acc) / (batch_index + 1)
        train_state['train_loss'].append(running_loss) 
        train_state['train_acc'].append(running_acc)
        args.logger.info("TRAIN loss: {}, acc: {}".format(running_loss, running_acc))

        # val
        running_loss, running_acc = 0.0 , 0.0 
        model.eval()
        for batch_index, batch_dict in tqdm(enumerate(val_dataloader)): 
            text_vectors = batch_dict["batch_text_vectors"].to(args.device)
            labels = batch_dict["batch_labels"].to(args.device)

            y_pred = model(text_vectors) 
            # step 2. compute the loss
            loss = loss_func(y_pred, labels) 
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1) 
            # step 3. compute the accuracy
            acc_batch = compute_accuracy(y_pred, labels)
            running_acc += (acc_batch - running_acc) / (batch_index + 1) 
        args.logger.info("VAL loss: {}, acc: {}".format(running_loss, running_acc))

        train_state['val_loss'].append(running_loss) 
        train_state['val_acc'].append(running_acc)



def make_train_state(): 
    return {'epoch_index': 0, 'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': 1,
            'test_acc': 1}


def test_engine(args, model, test_dataloader, train_state):
    running_loss, running_acc = 0.0 , 0.0 
    loss_func = args.loss_func
    model.to(args.device)
    model.eval()

    for batch_index, batch_dict in tqdm(enumerate(test_dataloader)):

        text_vectors = batch_dict["batch_text_vectors"].to(args.device)
        labels = batch_dict["batch_labels"].to(args.device)

        y_pred = model(text_vectors)

        # compute the loss
        loss = loss_func(y_pred, labels)

        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, labels)
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        train_state['test_loss'] = running_loss
        train_state['test_acc'] = running_acc

    args.logger.info("\n\nTEST loss: {}, acc: {}".format(running_loss, running_acc))


def get_logger(dir, tile):
    os.makedirs(dir, exist_ok=True)
    log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(dir, "{}_{}.log".format(log_file, tile))

    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(levelname)s:%(message)s"
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    
    fhlr = logging.FileHandler(log_file) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger