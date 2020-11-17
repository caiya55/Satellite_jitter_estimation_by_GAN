class Config(object):
  pass

def get_config(is_train):
  config = Config()
  if is_train:
    config.batch_size = 5
    config.im_size = [256, 256]
    config.lr = 1e-4
    config.iteration = 95

    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
  else:
    config.batch_size = 5
    config.im_size = [256, 256]

    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
  return config

def get_config2D(is_train):
  config = Config()
  if is_train:
    config.batch_size = 5
    config.im_size = [256, 256]
    config.lr = 1e-4
    config.iteration = 3000

    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt2D"
  else:
    config.batch_size = 5
    config.im_size = [256, 256]

    config.result_dir = "result"
    config.ckpt_dir = "ckpt2D"
  return config
