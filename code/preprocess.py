from zutils.datasets.twitter.tweet_database import TweetDatabase
from zutils.algo.utils import format_list_to_string
from zutils.algo.utils import ensure_directory_exist
from paras import load_params
from random import shuffle
import sys

class Tweet:
    def load_tweet(self, line):
        self.line = line
        items = line.split('\x01')
        self.id = long(items[0])
        self.uid = long(items[1])
        self.lat = float(items[2])
        self.lng = float(items[3])
        self.datetime = items[4]
        self.ts = int(float(items[5]))%(3600*24)
        self.text = items[6]
        self.words = self.text.split(' ')
        self.raw = items[7]
        if len(items)>8:
            self.poi_id = items[8]
            self.poi_lat = float(items[9]) if items[9] else items[9]
            self.poi_lng = float(items[10]) if items[10] else items[10]
            self.category = items[11]
            self.poi_name = items[12]
        else:
            self.category = ''


def load_tweets(clean_tweet_file):
    td = TweetDatabase()
    td.load_clean_tweets_from_file(clean_tweet_file)
    return td


def split_data(td, train_ratio, valid_ratio):
    td.shuffle_tweets()
    train_db, valid_test_db = td.split(train_ratio)
    valid_db, test_db = valid_test_db.split(valid_ratio / (1.0 - train_ratio))
    return train_db, valid_db, test_db


def write_data(td, word_dict, spatial_grid, output_file):
    ensure_directory_exist(output_file)
    with open(output_file, 'w') as fout:
        for tweet in td.tweets:
            word_ids = word_dict.convert_to_ids(tweet.message.words)
            location_id = spatial_grid.get_cell_id((tweet.location.lat, tweet.location.lng))
            # ignore the out-of-vocab location and words
            if location_id is not None and len(word_ids) > 0:
                fout.write(format_list_to_string([location_id, word_ids]) + '\n')


def load_labeled_tweets(tweet_file):
    tweets = []
    with open(tweet_file, 'r') as fin:
        for line in fin:
            items = line.strip().split('\x01')
            if len(items) > 8:
                t = Tweet()
                t.load_tweet(line)
                tweets.append(t)
    return tweets


def build_category_vocab(tweets, cat_vocab_file):
    cat_to_id = {}
    cats = list(set([t.category for t in tweets]))
    with open(cat_vocab_file, 'w') as fout:
        for idx, cat in enumerate(cats):
            cat_to_id[cat] = idx
            fout.write(str(idx) + '\t' + cat + '\n')
    return cat_to_id


def convert_tweet_cat(tweet, word_dict, cat_dict):
    words = tweet.words
    cat = tweet.category
    word_idx = word_dict.convert_to_ids(words)
    cat_idx = cat_dict[cat]
    return word_idx, cat_idx

def write_labeled_tweets(tweets, train_ratio, word_dict, cat_dict, train_file, test_file):
    shuffle(tweets)
    n_train = int(len(tweets) * train_ratio)
    ensure_directory_exist(train_file)
    ensure_directory_exist(test_file)
    with open(train_file, 'w') as fout:
        for tweet in tweets[:n_train]:
            word_idx, cat_idx = convert_tweet_cat(tweet, word_dict, cat_dict)
            if len(word_idx) > 0:
                t_string = format_list_to_string([cat_idx, word_idx])
                fout.write(t_string + '\n')
    with open(test_file, 'w') as fout:
        for tweet in tweets[n_train:]:
            word_idx, cat_idx = convert_tweet_cat(tweet, word_dict, cat_dict)
            if len(word_idx) > 0:
                t_string = format_list_to_string([cat_idx, word_idx])
                fout.write(t_string + '\n')

def run(pd):
    tweet_file = pd['raw_data_file']
    # load tweets
    td = load_tweets(tweet_file)
    # # build word vocab
    word_dict = td.gen_word_dict(min_freq=pd['min_token_freq'])
    word_dict.write_to_file(pd['x_vocab_file'])
    # build spatial vocab
    # spatial_grid = td.gen_spatial_grid(pd['grid_list'], min_freq=pd['min_token_freq'])
    # spatial_grid.write_to_file(pd['y_vocab_file'])
    # # split data
    # train_db, valid_db, test_db = split_data(td, pd['train_ratio'], pd['valid_ratio'])
    # # write train and test data into file
    # write_data(train_db, word_dict, spatial_grid, pd['train_data_file'])
    # write_data(valid_db, word_dict, spatial_grid, pd['valid_data_file'])
    # write_data(test_db, word_dict, spatial_grid, pd['test_data_file'])
    train_ratio = pd['classify_train_ratio']
    train_file = pd['classify_train_file']
    test_file = pd['classify_test_file']
    cat_vocab_file = pd['classify_cat_file']
    tweets = load_labeled_tweets(tweet_file)
    cat_dict = build_category_vocab(tweets, cat_vocab_file)
    write_labeled_tweets(tweets, train_ratio, word_dict, cat_dict, train_file, test_file)


if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file) # load parameters as a dict
    run(pd)
