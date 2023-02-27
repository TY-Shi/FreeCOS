r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """

    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.nclass = 1
        self.f1_buf = []
        self.precision_buf = []
        self.recall_buf = []

        self.f1_thinbuf = []
        self.precision_thinbuf = []
        self.recall_thinbuf = []

        self.f1_thickbuf = []
        self.precision_thickbuf = []
        self.recall_thickbuf = []

        self.quality_buf = []
        self.cor_buf = []
        self.com_buf = []

        self.quality_thinbuf = []
        self.cor_thinbuf = []
        self.com_thinbuf = []

        self.quality_thickbuf = []
        self.cor_thickbuf = []
        self.com_thickbuf = []

        self.loss_buf = dict()

    def update(self, f1, precision, recall, quality, cor, com, loss_dict):
        self.f1_buf.append(f1)
        self.precision_buf.append(precision)
        self.recall_buf.append(recall)
        self.quality_buf.append(quality)
        self.com_buf.append(com)
        self.cor_buf.append(cor)
        if loss_dict is not None:
            for key in loss_dict.keys():
                if key not in self.loss_buf.keys():
                    self.loss_buf[key] = []
                loss = loss_dict[key].detach().clone()
                if loss is None:
                    loss = torch.tensor(0.0)
                self.loss_buf[key].append(loss)

    def update_thinthick(self, f1, precision, recall, f1_thin, precision_thin, recall_thin, f1_thick, precision_thick,
                         recall_thick, quality, cor, com, quality_thin, cor_thin, com_thin, quality_thick, cor_thick, com_thick, loss_dict):
        self.f1_buf.append(f1)
        self.precision_buf.append(precision)
        self.recall_buf.append(recall)

        self.f1_thinbuf.append(f1_thin)
        self.precision_thinbuf.append(precision_thin)
        self.recall_thinbuf.append(recall_thin)

        self.f1_thickbuf.append(f1_thick)
        self.precision_thickbuf.append(precision_thick)
        self.recall_thickbuf.append(recall_thick)

        self.quality_buf.append(quality)
        self.com_buf.append(com)
        self.cor_buf.append(cor)

        self.quality_thinbuf.append(quality_thin)
        self.com_thinbuf.append(com_thin)
        self.cor_thinbuf.append(cor_thin)

        self.quality_thickbuf.append(quality_thick)
        self.com_thickbuf.append(com_thick)
        self.cor_thickbuf.append(cor_thick)

        if loss_dict is not None:
            for key in loss_dict.keys():
                if key not in self.loss_buf.keys():
                    self.loss_buf[key] = []
                loss = loss_dict[key].detach().clone()
                if loss is None:
                    loss = torch.tensor(0.0)
                self.loss_buf[key].append(loss)

    def compute_f1(self):
        f1 = torch.stack(self.f1_buf)
        f1 = f1.mean()
        return f1

    def compute_precision(self):
        precision = torch.stack(self.precision_buf)
        precision = precision.mean()
        return precision

    def compute_recall(self):
        recall = torch.stack(self.recall_buf)
        recall = recall.mean()
        return recall

    def compute_f1thin(self):
        f1 = torch.stack(self.f1_thinbuf)
        f1 = f1.mean()
        return f1

    def compute_precisionthin(self):
        precision = torch.stack(self.precision_thinbuf)
        precision = precision.mean()
        return precision

    def compute_recallthin(self):
        recall = torch.stack(self.recall_thinbuf)
        recall = recall.mean()
        return recall

    def compute_f1thick(self):
        f1 = torch.stack(self.f1_thickbuf)
        f1 = f1.mean()
        return f1

    def compute_precisionthick(self):
        precision = torch.stack(self.precision_thickbuf)
        precision = precision.mean()
        return precision

    def compute_recallthick(self):
        recall = torch.stack(self.recall_thickbuf)
        recall = recall.mean()
        return recall

    def compute_quality(self):
        quality = torch.stack(self.quality_buf)
        quality = quality.mean()
        return quality

    def compute_correctness(self):
        correctness = torch.stack(self.cor_buf)
        correctness = correctness.mean()
        return correctness

    def compute_completeness(self):
        completeness = torch.stack(self.com_buf)
        completeness = completeness.mean()
        return completeness

    def compute_qualitythin(self):
        quality = torch.stack(self.quality_thinbuf)
        quality = quality.mean()
        return quality

    def compute_correctnessthin(self):
        correctness = torch.stack(self.cor_thinbuf)
        correctness = correctness.mean()
        return correctness

    def compute_completenessthin(self):
        completeness = torch.stack(self.com_thinbuf)
        completeness = completeness.mean()
        return completeness

    def compute_qualitythick(self):
        quality = torch.stack(self.quality_thickbuf)
        quality = quality.mean()
        return quality

    def compute_correctnessthick(self):
        correctness = torch.stack(self.cor_thickbuf)
        correctness = correctness.mean()
        return correctness

    def compute_completenessthick(self):
        completeness = torch.stack(self.com_thickbuf)
        completeness = completeness.mean()
        return completeness

    def write_result(self, split, epoch):
        f1 = self.compute_f1()
        precision = self.compute_precision()
        recall = self.compute_recall()
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        for key in self.loss_buf.keys():
            loss_buf = torch.stack(self.loss_buf[key])
            msg += 'Avg ' + str(key) + ' :  %6.5f  ' % loss_buf.mean()
        msg += 'F1: %5.2f   ' % f1
        msg += 'Pr: %5.2f   ' % precision
        msg += 'R: %5.2f   ' % recall
        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            msg = '[Time: ' + dt_ms + '] '
            msg += '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx, datalen)
            f1 = self.compute_f1()
            if epoch != -1:
                for key in self.loss_buf.keys():
                    loss_buf = torch.stack(self.loss_buf[key])
                    msg += str(key) + ' :  %6.5f  ' % loss_buf[-1]
                    msg += 'Avg ' + str(key) + ' :  %6.5f  ' % loss_buf.mean()
            msg += 'F1: %5.2f  |  ' % f1
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """

    @classmethod
    def initialize(cls, config,training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        #logname = args.logname if training else '_TEST_' + args.weight.split('/')[-2].split('.')[0]  # + logtime
        logname = config.logname
        if logname == '': logname = logtime

        cls.logpath = os.path.join('logs', logname + '.log')
        #cls.benchmark = args.benchmark
        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        # logging.info('\n:=========== Curvilinear Segmentation. with JTFN ===========')
        # for arg_key in args.__dict__:
        #     logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        # logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_f1(cls, model, epoch, F1, optimizer_l,optimizer_r):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_l': optimizer_l.state_dict(), 'optimizer_r':optimizer_r},
                   os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f.\n' % (epoch, F1))

    @classmethod
    def save_model_f1_S(cls, model, epoch, F1, optimizer_l):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_l': optimizer_l.state_dict()},
                   os.path.join(cls.logpath, 'best_Segment.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f.\n' % (epoch, F1))

    @classmethod
    def save_model_f1_T(cls, model, epoch, F1, optimizer_l):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_l': optimizer_l.state_dict()},
                   os.path.join(cls.logpath, 'best_Dis.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f.\n' % (epoch, F1))

    @classmethod
    def save_model_f1_singlecps(cls, model, epoch, F1, optimizer_l):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_l': optimizer_l.state_dict()},
                   os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f.\n' % (epoch, F1))

    @classmethod
    def save_model_f1_single(cls, model, epoch, F1, optimizer):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f.\n' % (epoch, F1))

    @classmethod
    def save_model_f1_high(cls, model, epoch, F1, optimizer_l,optimizer_r,idx):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_l': optimizer_l.state_dict(), 'optimizer_r':optimizer_r},
                   os.path.join(cls.logpath, 'best_model_f1_{}.pt'.format(idx)))
        cls.info('Model saved @%d w/ val. F1: %5.2f.\n' % (epoch, F1))

    @classmethod
    def save_model_all(cls, model, epoch, F1, Pr, R, optimizer):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(cls.logpath, 'best_model_all.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f Pr: %5.2f R: %5.2f.\n' % (epoch, F1, Pr, R))

    @classmethod
    def save_model_sota(cls, model, epoch, F1, Pr, R, optimizer):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(cls.logpath, 'best_model_sota.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f Pr: %5.2f R: %5.2f.\n' % (epoch, F1, Pr, R))

    @classmethod
    def log_params(cls, model):
        # caluate the parameter
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))
