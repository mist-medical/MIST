import os
import datetime
import tensorflow as tf

from runtime.utils import create_empty_dir


class Logger:

    def __init__(self, args, fold, train_loss, val_loss):
        self.args = args
        self.fold = fold
        self.train_loss = train_loss
        self.val_loss = val_loss

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_dir = os.path.join(args.results, 'logs',
                                     'fold_{}'.format(self.fold),
                                     '{}'.format(current_time))
        create_empty_dir(self.base_dir)

        self.train_log_dir = os.path.join(self.base_dir, 'train')
        self.test_log_dir = os.path.join(self.base_dir, 'val')

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

    def update(self, epoch):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', self.val_loss.result(), step=epoch)
