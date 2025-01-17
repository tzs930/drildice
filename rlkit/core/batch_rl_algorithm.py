import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from rlkit.core import eval_util
from rlkit.torch.core import np_to_pytorch_batch

# class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
#     def __init__(
#             self,
#             trainer,
#             exploration_env,
#             evaluation_env,
#             exploration_data_collector: PathCollector,
#             evaluation_data_collector: PathCollector,
#             replay_buffer: ReplayBuffer,
#             batch_size,
#             max_path_length,
#             num_epochs,
#             num_eval_steps_per_epoch,
#             num_expl_steps_per_train_loop,
#             num_trains_per_train_loop,
#             num_train_loops_per_epoch=1,
#             min_num_steps_before_training=0,
#             num_trains_mine=10,
#             start_epoch=0, # negative epochs are offline, positive epochs are online
#             wandb=None
#     ):
#         super().__init__(
#             trainer,
#             exploration_env,
#             evaluation_env,
#             exploration_data_collector,
#             evaluation_data_collector,
#             replay_buffer,            
#         )
#         self.batch_size = batch_size
#         self.max_path_length = max_path_length
#         self.num_epochs = num_epochs
#         self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
#         self.num_trains_per_train_loop = num_trains_per_train_loop
#         self.num_train_loops_per_epoch = num_train_loops_per_epoch
#         self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
#         self.min_num_steps_before_training = min_num_steps_before_training
#         self._start_epoch = start_epoch
#         self.num_trains_mine = num_trains_mine
        
#         self.wandb = wandb
        

#     def train(self):
#         """Negative epochs are offline, positive epochs are online"""
#         for self.epoch in gt.timed_for(
#                 range(self._start_epoch, self.num_epochs),
#                 save_itrs=True,
#         ):
#             self.offline_rl = self.epoch < 0
#             self._begin_epoch(self.epoch)
#             self._train()            
            
#             if self.wandb:
#                eval_paths = self.eval_data_collector.get_epoch_paths()
#                eval_dict = eval_util.get_generic_path_information(eval_paths)
#                step = self.trainer._n_train_steps_total

#                self.wandb.log(eval_dict, step=step)

#             self._end_epoch(self.epoch)

#     def _train(self):
#         if self.epoch == 0 and self.min_num_steps_before_training > 0:
#             init_expl_paths = self.expl_data_collector.collect_new_paths(
#                 self.max_path_length,
#                 self.min_num_steps_before_training,
#                 discard_incomplete_paths=False,
#             )
#             if not self.offline_rl:
#                 self.replay_buffer.add_paths(init_expl_paths)
#             self.expl_data_collector.end_epoch(-1)

#         self.eval_data_collector.collect_new_paths(
#             self.max_path_length,
#             self.num_eval_steps_per_epoch,
#             discard_incomplete_paths=True,
#         )
#         gt.stamp('evaluation sampling')

#         for _ in range(self.num_train_loops_per_epoch):
#             new_expl_paths = self.expl_data_collector.collect_new_paths(
#                 self.max_path_length,
#                 self.num_expl_steps_per_train_loop,
#                 discard_incomplete_paths=False,
#             )
#             gt.stamp('exploration sampling', unique=False)

#             if not self.offline_rl:
#                 self.replay_buffer.add_paths(new_expl_paths)
#             gt.stamp('data storing', unique=False)

#             self.training_mode(True)
#             for it in range(self.num_trains_per_train_loop):
#                 for _ in range(self.num_trains_mine):
#                     train_data = self.replay_buffer.random_batch(self.batch_size)
#                     train_data = np_to_pytorch_batch(train_data)
#                     mine_stats = self.trainer.train_mi(train_data)

#                 if it % 1000 == 0:
#                     if self.wandb:
#                         step = self.trainer._n_train_steps_total
#                     # if self.wandb and step % 5000 == 0:
#                         print(step, mine_stats)
#                         self.wandb.log(mine_stats, step=step)
                    
#                 train_data = self.replay_buffer.random_batch(self.batch_size)
#                 self.trainer.train(train_data)

            
#             gt.stamp('training', unique=False)
#             self.training_mode(False)

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0, # negative epochs are offline, positive epochs are online
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not self.offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')

        for _ in range(self.num_train_loops_per_epoch):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            gt.stamp('exploration sampling', unique=False)

            if not self.offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)            
