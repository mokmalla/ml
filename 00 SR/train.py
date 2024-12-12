import time
import tensorflow as tf
import functools
import os

from model import evaluate
from model import srgan

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/edsr'):
        self.now = None
        self.loss = loss
        self.checkpoint_dir = checkpoint_dir

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)
        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue
                
                ckpt.psnr = psnr_value
                # ckpt -> weights
                # ckpt_mgr.save()
                self.save_weights(weights_dir=self.checkpoint_dir, step=step)
                
                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        try:
            step_filepath = self.checkpoint_dir+'/step.txt'
            with open(step_filepath, 'r') as file:
                lines = file.readlines()
                if not lines:
                    raise 
                step = int(lines[-1].strip())

            self.load_weights(self.checkpoint_dir, step=step)
            print(f'Model restored from checkpoint at step {step}.')
        except Exception as e:
            print(e)
            print('No Checkpoint')
        # if self.checkpoint_manager.latest_checkpoint:
        #     self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        #     print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


    def save_weights(self, weights_dir='./weights/edsr', step=None):
        try:
            self.checkpoint.model.save_weights(weights_dir + '/weights.h5')
            if step:
                step_filepath = weights_dir+'/step.txt'
                if not os.path.exists(step_filepath):
                    with open(step_filepath, 'w') as file:
                        pass
                with open(step_filepath, 'a') as file:
                    file.write(f'{step}' + '\n')
        
            print(f'save weights in {weights_dir} (step: {step})')
        except Exception as e:
            print(e)
            print('save failed')

        return
    
    def load_weights(self, weights_dir='./weights/edsr', step=None):
        self.checkpoint.model.load_weights(weights_dir + '/weights.h5')
        if step:
            self.set_step(step)
        return
    
    def set_step(self, step=0):
        self.checkpoint.step.assign(tf.Variable(step))
        print(f"setting step {step}")
    

class EdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class WdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganGeneratorTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganTrainer:
    #
    # TODO: model and optimizer checkpoints
    #
    def __init__(self,
                 generator,
                 discriminator,
                 checkpoint_dir='./.ckpt/srgan',
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        if content_loss == 'VGG22':
            self.vgg = srgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = srgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.content_loss = content_loss

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

        self.checkpoint1 = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate=learning_rate),
                                              model=generator)
        self.checkpoint_manager1 = tf.train.CheckpointManager(checkpoint=self.checkpoint1,
                                                             directory=checkpoint_dir+'/g',
                                                             max_to_keep=3)
        self.checkpoint2 = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate=learning_rate),
                                              model=discriminator)
        self.checkpoint_manager2 = tf.train.CheckpointManager(checkpoint=self.checkpoint2,
                                                             directory=checkpoint_dir+'/d',
                                                             max_to_keep=3)
        self.restore()
        

    def train(self, train_dataset,valid_dataset, steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()

        ckpt1 = self.checkpoint1
        ckpt2 = self.checkpoint2
        ckpt_mgr1 = self.checkpoint_manager1
        ckpt_mgr2 = self.checkpoint_manager2

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps-ckpt1.step.numpy()):
            ckpt1.step.assign_add(1)
            ckpt2.step.assign_add(1)
            step = ckpt1.step.numpy()

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 50 == 0:
                duration = time.perf_counter() - self.now
                psnr_value = self.evaluate(valid_dataset)

                print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                print(f'\tPSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')
                pls_metric.reset_states()
                dls_metric.reset_states()

                ckpt_mgr1.save()
                ckpt_mgr2.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint1.model(lr, training=True)

            hr_output = self.checkpoint2.model(hr, training=True)
            sr_output = self.checkpoint2.model(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.checkpoint1.model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.checkpoint2.model.trainable_variables)

        self.checkpoint1.optimizer.apply_gradients(zip(gradients_of_generator, self.checkpoint1.model.trainable_variables))
        self.checkpoint2.optimizer.apply_gradients(zip(gradients_of_discriminator, self.checkpoint2.model.trainable_variables))

        return perc_loss, disc_loss

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
    
    def evaluate(self, dataset):
        return evaluate(self.checkpoint1.model, dataset)
    
    def restore(self): 
        if self.checkpoint_manager1.latest_checkpoint:
            self.checkpoint1.restore(self.checkpoint_manager1.latest_checkpoint)
            print(f'Generator restored from checkpoint at step {self.checkpoint1.step.numpy()}.')        
        if self.checkpoint_manager2.latest_checkpoint:
            self.checkpoint2.restore(self.checkpoint_manager2.latest_checkpoint)
            print(f'Discriminater restored from checkpoint at step {self.checkpoint2.step.numpy()}.')
        
import tf2gan as gan

class CycleganTrainer:
    #
    # TODO: model and optimizer checkpoints
    #
    def __init__(self,
                 generator_H,
                 generator_L,
                 discriminator,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]),
                 checkpoint_dir='./.ckpt/cyclegan'):
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)
        self.MSE = tf.losses.MeanSquaredError()
        self.MAE = tf.losses.MeanAbsoluteError()
        self.checkpoint_dir = checkpoint_dir

        self.ckpt_G_L2H = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=generator_H)
        self.ckpt_G_H2L = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=generator_L)
        self.ckpt_D_H = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=discriminator)
        self.ckpt_D_L = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=discriminator)
        self.ckpt_mgr_G_L2H = tf.train.CheckpointManager(checkpoint=self.ckpt_G_L2H,
                                                             directory=checkpoint_dir+'/1',
                                                             max_to_keep=3)
        self.ckpt_mgr_G_H2L = tf.train.CheckpointManager(checkpoint=self.ckpt_G_H2L,
                                                             directory=checkpoint_dir+'/2',
                                                             max_to_keep=3)
        self.ckpt_mgr_D_H = tf.train.CheckpointManager(checkpoint=self.ckpt_D_H,
                                                             directory=checkpoint_dir+'/3',
                                                             max_to_keep=3)
        self.ckpt_mgr_D_L = tf.train.CheckpointManager(checkpoint=self.ckpt_D_L,
                                                             directory=checkpoint_dir+'/4',
                                                             max_to_keep=3)
        self.restore()

    @property
    def model(self):
        return self.ckpt_G_L2H.model

    def train(self, train_dataset, valid_dataset, steps=200000, evaluate_every=1000, save_best_only=False):
        pls_metric = Mean()
        dls_metric = Mean()
        
        ckpt1 = self.ckpt_G_L2H
        ckpt2 = self.ckpt_G_H2L
        ckpt3 = self.ckpt_D_H
        ckpt4 = self.ckpt_D_L
        ckpt_mgr1 = self.ckpt_mgr_G_L2H
        ckpt_mgr2 = self.ckpt_mgr_G_H2L
        ckpt_mgr3 = self.ckpt_mgr_D_H
        ckpt_mgr4 = self.ckpt_mgr_D_L

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps-ckpt1.step.numpy()):
            ckpt1.step.assign_add(1)
            ckpt2.step.assign_add(1)
            ckpt3.step.assign_add(1)
            ckpt4.step.assign_add(1)
            step = ckpt1.step.numpy()

            pl, dl = self.train_step(lr, hr)

            if step % evaluate_every == 0:
                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}', end='')
                print(f', PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')
                pls_metric.reset_states()
                dls_metric.reset_states()

                if save_best_only and psnr_value <= self.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                self.psnr = psnr_value

                # ckpt_mgr1.save()
                # ckpt_mgr2.save()
                # ckpt_mgr3.save()
                # ckpt_mgr4.save()
                self.save_weights_all(weights_dir=self.checkpoint_dir, step=step)

                self.now = time.perf_counter()
                

    @tf.function
    def train_G(self, lr, hr):
        with tf.GradientTape() as t:
            L2H = self.ckpt_G_L2H.model(lr, training=True)
            H2L = self.ckpt_G_H2L.model(hr, training=True)
            L2H2L = self.ckpt_G_H2L.model(L2H, training=True)
            H2L2H = self.ckpt_G_L2H.model(H2L, training=True)

            H_L1 = self.MAE(hr, L2H)
            L_L1 = self.MAE(lr, H2L)
            H_L2 = self.MSE(hr, H2L2H)
            L_L2 = self.MSE(lr, L2H2L)

            G_loss = (H_L1 + H_L2) + (L_L1 + L_L2)

        G_grad = t.gradient(G_loss, self.ckpt_G_L2H.model.trainable_variables + self.ckpt_G_H2L.model.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(G_grad, self.ckpt_G_L2H.model.trainable_variables + self.ckpt_G_H2L.model.trainable_variables))

        return L2H, H2L, {'H_L1': H_L1,
                        'L_L1': L_L1,
                        'H_L2': H_L2,
                        'L_L2': L_L2,
                        }
    
    @tf.function
    def train_D(self, A, B, A2B, B2A):
        with tf.GradientTape() as t:
            A_d_logits = self.ckpt_D_L.model(A, training=True)
            B2A_d_logits = self.ckpt_D_L.model(B2A, training=True)
            B_d_logits = self.ckpt_D_H.model(B, training=True)
            A2B_d_logits = self.ckpt_D_H.model(A2B, training=True)

            A_d_loss = self.MAE(A_d_logits, B2A_d_logits)
            B_d_loss = self.MAE(B_d_logits, A2B_d_logits)

            D_loss = (A_d_loss ) + (B_d_loss )

        D_grad = t.gradient(D_loss, self.ckpt_D_L.model.trainable_variables + self.ckpt_D_H.model.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(D_grad, self.ckpt_D_L.model.trainable_variables + self.ckpt_D_H.model.trainable_variables))

        return {'A_d_loss': A_d_loss,
                'B_d_loss': B_d_loss,
                }
    
        # cannot autograph `A2B_pool`
        # A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
        # B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    def train_step(self, lr, hr):
        A2B, B2A, G_loss_dict = self.train_G(lr, hr)
        D_loss_dict = self.train_D(lr, hr, A2B, B2A)

        return G_loss_dict, D_loss_dict

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
    
    def evaluate(self, dataset):
        return evaluate(self.ckpt_G_L2H.model, dataset)

    def restore(self): 
        try:
            step_filepath = self.checkpoint_dir+'/step.txt'
            with open(step_filepath, 'r') as file:
                lines = file.readlines()
                if not lines:
                    raise 
                step = int(lines[-1].strip())

            self.load_weights_all(self.checkpoint_dir, step=step)
            print(f'Model restored from checkpoint at step {step}.')
        except Exception as e:
            print(e)
            print('No Checkpoint')

        # if self.ckpt_mgr_G_L2H.latest_checkpoint:
        #     self.ckpt_G_L2H.restore(self.ckpt_mgr_G_L2H.latest_checkpoint)
        #     print(f'Model(G_L2H) restored from checkpoint at step {self.ckpt_G_L2H.step.numpy()}.')        
        # if self.ckpt_mgr_G_H2L.latest_checkpoint:
        #     self.ckpt_G_H2L.restore(self.ckpt_mgr_G_H2L.latest_checkpoint)
        #     print(f'Model(G_H2L) restored from checkpoint at step {self.ckpt_G_H2L.step.numpy()}.')
        # if self.ckpt_mgr_D_H.latest_checkpoint:
        #     self.ckpt_D_H.restore(self.ckpt_mgr_D_H.latest_checkpoint)
        #     print(f'Model(D_H) restored from checkpoint at step {self.ckpt_D_H.step.numpy()}.')
        # if self.ckpt_mgr_D_L.latest_checkpoint:
        #     self.ckpt_D_L.restore(self.ckpt_mgr_D_L.latest_checkpoint)
        #     print(f'Model(D_L) restored from checkpoint at step {self.ckpt_D_L.step.numpy()}.')
    
    def save_weights_all(self, weights_dir='./weights/cyclegan', step=None):
        try:
            self.ckpt_G_L2H.model.save_weights(weights_dir + '/weights_1.h5')
            self.ckpt_G_H2L.model.save_weights(weights_dir + '/weights_2.h5')
            self.ckpt_D_H.model.save_weights(weights_dir + '/weights_3.h5')
            self.ckpt_D_L.model.save_weights(weights_dir + '/weights_4.h5')
            if step:
                step_filepath = weights_dir+'/step.txt'
                if not os.path.exists(step_filepath):
                    with open(step_filepath, 'w') as file:
                        pass
                with open(step_filepath, 'a') as file:
                    file.write(f'{step}' + '\n')
            print(f'save weights in {weights_dir} (step: {step})')
        except Exception as e:
            print(e)
            print('save failed')

        return
    
    def load_weights_all(self, weights_dir='./weights/cyclegan', step=None):
        self.ckpt_G_L2H.model.load_weights(weights_dir + '/weights_1.h5')
        self.ckpt_G_H2L.model.load_weights(weights_dir + '/weights_2.h5')
        self.ckpt_D_H.model.load_weights(weights_dir + '/weights_3.h5')
        self.ckpt_D_L.model.load_weights(weights_dir + '/weights_4.h5')
        if step:
            self.set_step(step)
        return
    
    def set_step(self, step=0):
        self.ckpt_G_L2H.step.assign(tf.Variable(step))
        self.ckpt_G_H2L.step.assign(tf.Variable(step))
        self.ckpt_D_H.step.assign(tf.Variable(step))
        self.ckpt_D_L.step.assign(tf.Variable(step))
        print(f"setting step {step}")
