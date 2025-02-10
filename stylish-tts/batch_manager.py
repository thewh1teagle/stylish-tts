import json
import os.path as osp

from meldataset import FilePathDataset, build_dataloader, get_frame_count, get_time_bin
import utils


class BatchManager:
    def __init__(
        self,
        train_path,
        log_dir,
        probe_batch_max=None,
        root_path="",
        OOD_data=[],
        min_length=50,
        device="cpu",
        accelerator=None,
        log_print=None,
        multispeaker=False,
        text_cleaner=None,
        stage="",
    ):
        self.train_path = train_path
        self.probe_batch_max = probe_batch_max
        self.log_dir = log_dir
        self.log_print = log_print
        self.device = device
        self.multispeaker = multispeaker
        self.stage = stage
        self.batch_dict = {}
        self.load_batch_dict()

        train_list = utils.get_data_path_list(self.train_path)
        if len(train_list) == 0:
            print("Could not open train_list", self.train_path)
            exit()
        self.dataset = FilePathDataset(
            train_list,
            root_path,
            OOD_data=OOD_data,
            min_length=min_length,
            validation=False,
            multispeaker=multispeaker,
            text_cleaner=text_cleaner,
        )
        self.time_bins = self.dataset.time_bins()
        self.process_count = 1
        if accelerator is not None:
            self.process_count = accelerator.num_processes
            accelerator.even_batches = False
        loader = build_dataloader(
            self.dataset,
            self.time_bins,
            batch_size=self.batch_dict,
            num_workers=32,
            device=device,
            drop_last=True,
            multispeaker=multispeaker,
        )
        self.epoch_step_count = len(loader.batch_sampler)

    def get_step_count(self):
        return self.epoch_step_count // self.process_count

    def get_batch_size(self, i):
        batch_size = 1
        if str(i) in self.batch_dict:
            batch_size = self.batch_dict[str(i)]
        return batch_size

    def set_batch_size(self, i, batch_size):
        self.batch_dict[str(i)] = batch_size

    def load_batch_dict(self):
        batch_file = osp.join(self.log_dir, f"{self.stage}_batch_sizes.json")
        if osp.isfile(batch_file):
            with open(batch_file, "r") as batch_input:
                self.batch_dict = json.load(batch_input)

    def save_batch_dict(self):
        batch_file = osp.join(self.log_dir, f"{self.stage}_batch_sizes.json")
        with open(batch_file, "w") as o:
            json.dump(self.batch_dict, o)

    def epoch_loop(self, train, debug=False) -> bool:
        if not self.batch_dict:
            self.probe_loop(train)
            # return true here so we know we probed instead of trained
            return True
        else:
            self.train_loop(train=train, debug=debug)
            return False

    def probe_loop(self, train):
        if self.process_count > 1:
            exit(
                "--probe_batch must be run with accelerator num_processes set to 1. After running it, distribute the batch_sizes.json files to the log directories and run in DDP"
            )

        self.batch_dict = {}
        batch_size = self.probe_batch_max
        time_keys = sorted(list(self.time_bins.keys()))
        max_frame_size = get_frame_count(time_keys[-1])
        for key in time_keys:
            frame_count = get_frame_count(key)
            last_bin = key
            done = False
            while not done:
                try:
                    if batch_size == 1:
                        self.set_batch_size(key, 1)
                        done = True
                    elif batch_size > 0:
                        print(
                            "Attempting %d/%d @ %d"
                            % (frame_count, max_frame_size, batch_size)
                        )
                        loader = build_dataloader(
                            self.dataset,
                            self.time_bins,
                            batch_size=self.batch_dict,
                            num_workers=1,
                            device=self.device,
                            drop_last=True,
                            multispeaker=self.multispeaker,
                            probe_bin=key,
                            probe_batch_size=batch_size,
                        )

                        loader = train.accelerator.prepare(loader)
                        for _, batch in enumerate(loader):
                            _, _ = train.train_batch(
                                i=0, batch=batch, running_loss=0, iters=0, train=train
                            )
                            break
                        self.set_batch_size(key, batch_size)
                    done = True
                except Exception as e:
                    if "out of memory" in str(e):
                        audio_length = (last_bin * 0.25) + 0.25
                        self.log_print(
                            f"TRAIN_BATCH OOM ({last_bin}) @ batch_size {batch_size}: audio_length {audio_length} total audio length {audio_length * batch_size}"
                        )
                        print("Probe saw OOM -- backing off")
                        import gc

                        train.optimizer.zero_grad()
                        gc.collect()
                        torch.cuda.empty_cache()
                        counting_up = False
                        if batch_size > 1:
                            batch_size -= 1
                    else:
                        print("UNKNOWN EXCEPTION")
                        raise e
        self.save_batch_dict()

    def train_loop(self, train, debug=False):
        running_loss = 0
        iters = 0
        last_oom = -1
        max_attempts = 3
        loader = build_dataloader(
            self.dataset,
            self.time_bins,
            batch_size=self.batch_dict,
            num_workers=32,
            device=self.device,
            drop_last=True,
            multispeaker=self.multispeaker,
            epoch=train.manifest.current_epoch,
        )
        self.epoch_step_count = len(loader.batch_sampler)
        loader = train.accelerator.prepare(loader)
        for i, batch in enumerate(loader):
            last_bin = get_time_bin(batch[0].shape[-1])
            for attempt in range(1, max_attempts + 1):
                try:
                    if debug:
                        batch_size = self.get_batch_size(last_bin)
                        audio_length = (last_bin * 0.25) + 0.25
                        self.log_print(
                            f"train_batch(i={i}, batch={batch_size}, running_loss={running_loss}, iters={iters}), segment_bin_length={audio_length}, total_audio_in_batch={batch_size * audio_length}"
                        )
                    running_loss, iters = train.train_batch(
                        i, batch, running_loss, iters, train
                    )
                    break
                except Exception as e:
                    batch_size = self.get_batch_size(last_bin)
                    audio_length = (last_bin * 0.25) + 0.25
                    if "CUDA out of memory" in str(e):
                        self.log_print(
                            f"{attempt * ('*' if attempt < max_attempts else 'X')}\n"
                            f"TRAIN_BATCH OOM ({last_bin}) @ batch_size {batch_size}: audio_length {audio_length} total audio length {audio_length * batch_size}"
                        )
                        self.log_print(e)
                        train.optimizer.zero_grad()
                        if last_oom != last_bin:
                            last_oom = last_bin
                            if batch_size > 1:
                                batch_size -= 1
                            self.set_batch_size(last_bin, batch_size)
                            self.save_batch_dict()
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        raise e
