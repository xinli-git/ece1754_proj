import torch.nn.functional as F
import torch
import numba

torch.backends.cudnn.benchmark = True


def eval_task(args):

    do_profile = False
    device = torch.device('cuda')
    # Use the last layer in ResNet-50
    N, H, W, CO, CI, KH, KW, strides, padding, groups= args

    data = torch.randn((N, CI * groups, H, W)).to(device)
    weight = torch.randn((CO * groups, CI, KH, KW)).to(device)
    # warm up
    for _ in range(10):
        out = F.conv2d(data,
                       weight,
                       stride=strides,
                       padding=padding,
                       groups=groups)

    tot = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(1000):

        out = F.conv2d(data,
                       weight,
                       stride=strides,
                       padding=padding,
                       groups=groups)
    end.record()
    end.synchronize()

    tot += start.elapsed_time(end)

    return tot / 1000


if __name__ == "__main__":
    niters = 1000
    filtre_size_opionts = [\
            (224, 224, 64, 3, 7, 7, (2, 2), (3, 3), 1),
            (7, 7, 512, 512, 3, 3, (1, 1), (1, 1), 1),
            ]
    for batch_size in [1, 8, 64]:
        for idx, filter_size in enumerate(filtre_size_opionts):
            args = (batch_size, ) + filter_size
            prefix = "batch_{}_filter_id_{}".format(batch_size, idx)
            print(prefix, ":", eval_task(args))
