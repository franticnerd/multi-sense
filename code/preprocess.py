from zutils.datasets.twitter.tweet_database import TweetDatabase
from zutils.algo.utils import format_list_to_string
from zutils.algo.utils import ensure_directory_exist
from paras import load_params
import sys


def load_tweets(clean_tweet_file):
    td = TweetDatabase()
    td.load_clean_tweets_from_file(clean_tweet_file)
    return td


def split_data(td, train_ratio):
    td.shuffle_tweets()
    train_db, test_db = td.split(train_ratio)
    return train_db, test_db


def write_data(td, word_dict, spatial_grid, output_file):
    ensure_directory_exist(output_file)
    with open(output_file, 'w') as fout:
        for tweet in td.tweets:
            word_ids = word_dict.convert_to_ids(tweet.message.words)
            location_id = spatial_grid.get_cell_id((tweet.location.lat, tweet.location.lng))
            # ignore the out-of-vocab location and words
            if location_id is not None and len(word_ids) > 0:
                fout.write(format_list_to_string([location_id, word_ids]) + '\n')


def run(pd):
    # load tweets
    td = load_tweets(pd['raw_data_file'])
    # build word vocab
    word_dict = td.gen_word_dict(min_freq=pd['min_token_freq'])
    word_dict.write_to_file(pd['x_vocab_file'])
    # build spatial vocab
    spatial_grid = td.gen_spatial_grid(pd['grid_list'], min_freq=pd['min_token_freq'])
    spatial_grid.write_to_file(pd['y_vocab_file'])
    # split data
    train_db, test_db = split_data(td, pd['train_ratio'])
    # write train and test data into file
    write_data(train_db, word_dict, spatial_grid, pd['train_data_file'])
    write_data(test_db, word_dict, spatial_grid, pd['test_data_file'])


if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file) # load parameters as a dict
    run(pd)
