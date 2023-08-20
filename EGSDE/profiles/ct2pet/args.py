import argparse
argsall = argparse.Namespace(
    ckpt = 'pretrained_model/pet_diffusion.pt',
    dsepath = 'pretrained_model/ct2pet.pt',
    config_path = 'profiles/ct2pet/ct2pet.yml',
    t = 40,
    ls = 400,
    li = 0,
    s1 = 'cosine',
    s2 = 'neg_l2',
    phase = 'test',
    root = 'runs/',
    sample_step = 1,
    batch_size = 4,
    diffusionmodel = 'ADM',
    down_N = 16,
    seed=1234)