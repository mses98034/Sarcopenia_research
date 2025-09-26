from configparser import ConfigParser
import configparser
import sys, os

sys.path.append('..')


class Configurable(object):
    def __init__(self, args, extra_args):
        config = ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(args.config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        if args.train:
            config.set('Run', 'gpu', args.gpu)
            config.set('Run', 'run_num', args.run_num)
            config.set('Run', 'gpu_count', args.gpu_count)
            config.set('Network', 'ema_decay', args.ema_decay)
            config.set('Network', 'model', args.model)
            config.set('Network', 'backbone', args.backbone)

            # Add timestamp to save_dir to avoid conflicts between different training runs
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            original_save_dir = self._config.get('Save', 'save_dir')
            timestamped_save_dir = f"{original_save_dir}_{timestamp}"
            config.set('Save', 'save_dir', timestamped_save_dir)

            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    def set_attr(self, section, name, value):
        return self._config.set(section, name, value)

    @property
    def patch_x(self):
        return self._config.getint('Data', 'patch_x')

    @property
    def patch_y(self):
        return self._config.getint('Data', 'patch_y')

    @property
    def aug_num(self): # ???
        return self._config.getint('Data', 'aug_num')

    @property
    def data_name(self):
        return self._config.get('Data', 'data_name')

    @property
    def data_path(self):
        return self._config.get('Data', 'data_path')

    @property
    def train_filename(self):
        return self._config.get('Data', 'train_filename')

    @property
    def test_filename(self):
        return self._config.get('Data', 'test_filename')

    # ------------save path config reader--------------------

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def tmp_dir(self):
        return self._config.get('Save', 'tmp_dir')

    @property
    def tensorboard_dir(self):
        return self._config.get('Save', 'tensorboard_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def log_file(self):
        return self._config.get('Save', 'log_file')

    # ------------Network path config reader--------------------

    @property
    def model(self):
        return self._config.get('Network', 'model')

    @property
    def anchors_size(self):
        anchors_size = self._config.get('Network', 'anchors_size')
        anchors_size = anchors_size.split(',')
        return [int(x) for x in anchors_size]

    @property
    def n_channels(self):
        return self._config.getint('Network', 'n_channels')

    @property
    def backbone(self):
        return self._config.get('Network', 'backbone')

    @property
    def ema_decay(self):
        return self._config.getfloat('Network', 'ema_decay')

    @property
    def use_pretrained(self):
        # Default to True for backward compatibility
        try:
            return self._config.getboolean('Network', 'use_pretrained')
        except:
            return True

    # ------------Network path config reader--------------------

    @property
    def epochs(self):
        return self._config.getint('Run', 'n_epochs')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def patience(self):
        return self._config.getint('Run', 'patience')

    @property
    def gpu(self):
        return self._config.getint('Run', 'gpu')

    @property
    def printfreq(self):
        return self._config.getint('Run', 'printfreq')

    @property
    def gpu_count(self):
        gpus = self._config.get('Run', 'gpu_count')
        gpus = gpus.split(',')
        return [int(x) for x in gpus]

    @property
    def workers(self):
        return self._config.getint('Run', 'workers')

    @property
    def run_num(self):
        return self._config.getint('Run', 'run_num')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def load_best_epoch(self):
        return self._config.getboolean('Run', 'load_best_epoch')

    @property
    def fp16(self):
        return self._config.getboolean('Run', 'fp16')


    @property
    def test_det_batch_size(self):
        return self._config.getint('Run', 'test_det_batch_size')

    @property
    def nfold(self):
        return self._config.getint('Run', 'nfold')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')
    
    @property
    def early_stopping(self):
        return self._config.getboolean('Run', 'early_stopping') 
        
    # ------------Optimizer path config reader--------------------
    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def min_lrate(self):
        return self._config.getfloat('Optimizer', 'min_lrate')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')

    @property
    def loss_function(self):
        return self._config.get('Loss', 'loss_function') 

    @property
    def weight_decay(self):
        return self._config.getfloat('Optimizer', 'weight_decay') 

    @property
    def beta1(self):
        return self._config.getfloat('Optimizer', 'beta1') 

    @property
    def beta2(self):
        return self._config.getfloat('Optimizer', 'beta2') 


    # ------------Scheduler path config reader--------------------
    @property
    def scheduler_patience(self):
        return self._config.getint('Scheduler', 'scheduler_patience') 

    @property
    def scheduler_factor(self):
        return self._config.getfloat('Scheduler', 'scheduler_factor') 
    @property
    def huber_delta(self):
        return self._config.getfloat('Loss', 'huber_delta') 

    # ------------Contrastive Learning path config reader--------------------
    @property
    def contrastive_learning(self):
        return self._config.getboolean('Contrastive', 'contrastive_learning')

    @property
    def cam_enhancement(self):
        return self._config.getboolean('Contrastive', 'cam_enhancement')

    @property
    def contrastive_beta(self):
        return self._config.getfloat('Contrastive', 'contrastive_beta')

    @property
    def contrastive_temperature(self):
        return self._config.getfloat('Contrastive', 'contrastive_temperature')

    # ------------Implant Removal config reader--------------------
    @property
    def remove_implants(self):
        try:
            return self._config.getboolean('Data', 'remove_implants')
        except:
            return False  # Default to False for backward compatibility

    @property
    def implant_threshold(self):
        try:
            return self._config.getint('Data', 'implant_threshold')
        except:
            return 240  # Default threshold

    @property
    def removal_strategy(self):
        try:
            return self._config.get('Data', 'removal_strategy')
        except:
            return 'gaussian_noise'  # Default strategy