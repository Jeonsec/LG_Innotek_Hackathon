""" Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .poly_lr import PolyLRScheduler
from .step_lr import StepLRScheduler


def create_scheduler(args, optimizer):
    num_epochs = args.epochs

    if getattr(args, "lr_noise", None) is not None:
        lr_noise = getattr(args, "lr_noise")
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(args, "lr_noise_pct", 0.67),
        noise_std=getattr(args, "lr_noise_std", 1.0),
        noise_seed=getattr(args, "seed", 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(args, "lr_cycle_mul", 1.0),
        cycle_decay=getattr(args, "lr_cycle_decay", 1.0),
        cycle_limit=getattr(args, "lr_cycle_limit", 10),
    )

    lr_scheduler = None
    if args.sched == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            k_decay=getattr(args, "lr_k_decay", 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            **noise_args,
        )
    elif args.sched == "poly":
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=args.power,  # overloading 'decay_rate' as polynomial power
            t_initial=num_epochs,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            k_decay=getattr(args, "lr_k_decay", 1.0),
#            **cycle_args,
#            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs

    return lr_scheduler, num_epochs
