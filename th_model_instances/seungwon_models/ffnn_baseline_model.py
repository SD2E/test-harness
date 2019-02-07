import tensorflow as tf

from protstab_test_harness_and_leaderboard.model_runner_instances.seungwon_models.utils_nn import *
from protstab_test_harness_and_leaderboard.model_runner_instances.seungwon_models.utils_train import train_mtl_neural_net

########################################################
####    Feedforward Net for Single-task Learning    ####
########################################################
#### FFNN3 model for MTL
class MTL_several_FFNN_minibatch():
    def __init__(self, num_tasks, dim_layers, batch_size, learning_rate, learning_rate_decay=-1, l1_reg_scale=0.0, max_epochs=10000, patience=1000, patience_multiplier=2.0, improvement_threshold=0.99, train_validation_ratio=[0.8, 0.2]):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers) - 1
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.l1_reg_scale = l1_reg_scale
        self.batch_size = batch_size

        self.max_epochs = max_epochs
        self.patience = patience
        self.patience_multiplier = patience_multiplier
        self.improvement_threshold = improvement_threshold
        self.train_validation_ratio = train_validation_ratio

        #### session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)

        #### placeholder of model
        self.model_input = [tf.placeholder(tf.float32, [None, self.layers_size[0]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(tf.float32, [None, self.layers_size[-1]]) for _ in range(self.num_tasks)]
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model for train data
        self.models, self.param, reg_param = [], [], []
        for task_cnt in range(self.num_tasks):
            model_tmp, param_tmp = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=None)
            self.models.append(model_tmp)
            self.param.append(param_tmp)
            reg_param.append(param_tmp[0::2])

        self.param_in_list = sum(self.param, [])
        self.param_placeholders = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in self.param_in_list]
        self.param_update_ops = [p.assign(p_placeholder) for p, p_placeholder in zip(self.param_in_list, self.param_placeholders)]

        #### functions of model
        self.eval, self.loss, self.accuracy, _ = mtl_model_output_functions(self.models, self.true_output, self.num_tasks)

        with tf.name_scope('L1_regularization'):
            reg_loss = []
            for param_list in reg_param:
                reg_term = 0.0
                for p in param_list:
                    reg_term = reg_term + tf.reduce_sum(tf.abs(p))
                reg_loss.append(reg_term)

        if learning_rate_decay <= 0:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss[x] + self.l1_reg_scale * reg_loss[x]) for x in range(self.num_tasks)]
        else:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate / (1.0 + self.epoch*self.learn_rate_decay)).minimize(self.loss[x] + self.l1_reg_scale * reg_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()
        self.sess.run(tf.global_variables_initializer())

    def get_param(self):
        return self.sess.run(self.param_in_list)

    def set_param(self, param_val):
        assert (len(param_val) == len(self.param_update_ops)), "The number of given parameters' value doesn't match with the number of parameters in the network."
        for param_cnt in range(len(self.param_in_list)):
            self.sess.run(self.param_update_ops[param_cnt], feed_dict={self.param_placeholders[param_cnt]: param_val[param_cnt]})

    def fit(self, X, y):
        train_mtl_neural_net(self.sess, [self.update, self.loss, self.get_param, self.set_param], [self.model_input, self.true_output, self.epoch], self.num_tasks, self.batch_size, self.max_epochs, self.patience, self.patience_multiplier, self.improvement_threshold, X, y, self.train_validation_ratio)

    def predict(self, X, k):
        assert (k < self.num_tasks and k > -1), "Given task exceed the index of tasks"
        return self.sess.run(self.eval[k], feed_dict={self.model_input[k]: X})


########################################################
#### Single Feedforward Net for Multi-task Learning ####
########################################################
#### FFNN3 model for MTL
class MTL_FFNN_minibatch():
    def __init__(self, num_tasks, dim_layers, batch_size, learning_rate, learning_rate_decay=-1, l1_reg_scale=0.0, max_epochs=10000, patience=1000, patience_multiplier=2.0, improvement_threshold=0.99, train_validation_ratio=[0.8, 0.2]):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.l1_reg_scale = l1_reg_scale
        self.batch_size = batch_size

        self.max_epochs = max_epochs
        self.patience = patience
        self.patience_multiplier = patience_multiplier
        self.improvement_threshold = improvement_threshold
        self.train_validation_ratio = train_validation_ratio

        #### session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)

        #### placeholder of model
        self.model_input = [tf.placeholder(tf.float32, [None, self.layers_size[0]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(tf.float32, [None, self.layers_size[-1]]) for _ in range(self.num_tasks)]
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model
        self.models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.param = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=None)
            else:
                model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=self.param)
            self.models.append(model_tmp)
        reg_param = self.param[0::2]

        self.param_placeholders = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in self.param]
        self.param_update_ops = [p.assign(p_placeholder) for p, p_placeholder in zip(self.param, self.param_placeholders)]

        #### functions of model
        self.eval, self.loss, self.accuracy, _ = mtl_model_output_functions(self.models, self.true_output, self.num_tasks)

        with tf.name_scope('L1_regularization'):
            reg_loss = 0.0
            for p in reg_param:
                reg_loss = reg_loss + tf.reduce_sum(tf.abs(p))

        if learning_rate_decay <= 0:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]
        else:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()
        self.sess.run(tf.global_variables_initializer())

    def get_param(self):
        return self.sess.run(self.param)

    def set_param(self, param_val):
        assert (len(param_val) == len(self.param_update_ops)), "The number of given parameters' value doesn't match with the number of parameters in the network."
        for param_cnt in range(len(self.param)):
            self.sess.run(self.param_update_ops[param_cnt], feed_dict={self.param_placeholders[param_cnt]: param_val[param_cnt]})

    def fit(self, X, y):
        train_mtl_neural_net(self.sess, [self.update, self.loss, self.get_param, self.set_param], [self.model_input, self.true_output, self.epoch], self.num_tasks, self.batch_size, self.max_epochs, self.patience, self.patience_multiplier, self.improvement_threshold, X, y, self.train_validation_ratio)

    def predict(self, X, k):
        assert (k < self.num_tasks and k > -1), "Given task exceed the index of tasks"
        return self.sess.run(self.eval[k], feed_dict={self.model_input[k]: X})


########################################################
#### Hard Parameter Sharing for Multi-task Learning ####
########################################################
class MTL_FFNN_HPS_minibatch():
    def __init__(self, num_tasks, dim_shared_layers, dim_task_specific_layers, batch_size, learning_rate, learning_rate_decay=-1, l1_reg_scale=0.0, max_epochs=10000, patience=1000, patience_multiplier=2.0, improvement_threshold=0.99, train_validation_ratio=[0.8, 0.2]):
        #### parameters
        self.num_tasks = num_tasks
        self.shared_layers_size = dim_shared_layers
        self.task_specific_layers_size = dim_task_specific_layers
        self.num_layers = [len(self.shared_layers_size)-1] + [len(self.task_specific_layers_size[x]) for x in range(self.num_tasks)]

        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.l1_reg_scale = l1_reg_scale
        self.batch_size = batch_size

        self.max_epochs = max_epochs
        self.patience = patience
        self.patience_multiplier = patience_multiplier
        self.improvement_threshold = improvement_threshold
        self.train_validation_ratio = train_validation_ratio

        #### session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)

        #### placeholder of model
        self.model_input = [tf.placeholder(tf.float32, [None, self.shared_layers_size[0]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(tf.float32, [None, self.task_specific_layers_size[0][-1]]) for _ in range(self.num_tasks)]
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model
        self.models, self.specific_param = [], []
        for task_cnt in range(self.num_tasks):
            #### generate network common to tasks
            if task_cnt == 0:
                shared_model_tmp, self.shared_param = new_fc_net(self.model_input[task_cnt], self.shared_layers_size[1:], params=None)
            else:
                shared_model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.shared_layers_size[1:], params=self.shared_param)

            #### generate task-dependent network
            specific_model_tmp, ts_params = new_fc_net(shared_model_tmp[-1], self.task_specific_layers_size[task_cnt], params=None)

            self.models.append(shared_model_tmp + specific_model_tmp)
            self.specific_param.append(ts_params)
        self.param = self.shared_param + sum(self.specific_param, [])

        self.param_placeholders = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in self.param]
        self.param_update_ops = [p.assign(p_placeholder) for p, p_placeholder in zip(self.param, self.param_placeholders)]

        reg_param = self.shared_param[0::2]

        #### functions of model
        self.eval, self.loss, self.accuracy, _ = mtl_model_output_functions(self.models, self.true_output, self.num_tasks)

        with tf.name_scope('L1_regularization'):
            reg_loss = 0.0
            for p in reg_param:
                reg_loss = reg_loss + tf.reduce_sum(tf.abs(p))

        if learning_rate_decay <= 0:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]
        else:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()
        self.sess.run(tf.global_variables_initializer())

    def get_param(self):
        return self.sess.run(self.param)

    def set_param(self, param_val):
        assert (len(param_val) == len(self.param_update_ops)), "The number of given parameters' value doesn't match with the number of parameters in the network."
        for param_cnt in range(len(self.param)):
            self.sess.run(self.param_update_ops[param_cnt], feed_dict={self.param_placeholders[param_cnt]: param_val[param_cnt]})

    def fit(self, X, y):
        train_mtl_neural_net(self.sess, [self.update, self.loss, self.get_param, self.set_param], [self.model_input, self.true_output, self.epoch], self.num_tasks, self.batch_size, self.max_epochs, self.patience, self.patience_multiplier, self.improvement_threshold, X, y, self.train_validation_ratio)

    def predict(self, X, k):
        assert (k < self.num_tasks and k > -1), "Given task exceed the index of tasks"
        return self.sess.run(self.eval[k], feed_dict={self.model_input[k]: X})


########################################################
####  Tensor Factorization for Multi-task Learning  ####
########################################################
class MTL_FFNN_Tensor_Factor_minibatch():
    def __init__(self, num_tasks, dim_shared_layers, dim_task_specific_layers, batch_size, learning_rate, learning_rate_decay=-1, l1_reg_scale=0.0, factor_type='Tucker', factor_eps_or_k=0.01, max_epochs=10000, patience=1000, patience_multiplier=2.0, improvement_threshold=0.99, train_validation_ratio=[0.8, 0.2]):
        self.num_tasks = num_tasks
        self.shared_layers_size = dim_shared_layers
        self.task_specific_layers_size = dim_task_specific_layers
        self.num_layers = [len(self.shared_layers_size)-1] + [len(self.task_specific_layers_size[x]) for x in range(self.num_tasks)]

        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.l1_reg_scale = l1_reg_scale
        self.batch_size = batch_size

        self.max_epochs = max_epochs
        self.patience = patience
        self.patience_multiplier = patience_multiplier
        self.improvement_threshold = improvement_threshold
        self.train_validation_ratio = train_validation_ratio

        #### session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)

        #### placeholder of model
        self.model_input = [tf.placeholder(tf.float32, [None, self.shared_layers_size[0]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(tf.float32, [None, self.task_specific_layers_size[0][-1]]) for _ in range(self.num_tasks)]
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model
        self.models, self.shared_param, self.specific_param = new_tensorfactored_fc_fc_nets(self.model_input, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, activation_fn=tf.nn.relu, shared_params=None, specific_params=None, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)
        self.param = self.shared_param + self.specific_param

        self.param_placeholders = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in self.param]
        self.param_update_ops = [p.assign(p_placeholder) for p, p_placeholder in zip(self.param, self.param_placeholders)]

        reg_param = self.shared_param[0::2]

        #### functions of model
        self.eval, self.loss, self.accuracy, _ = mtl_model_output_functions(self.models, self.true_output, self.num_tasks)

        with tf.name_scope('L1_regularization'):
            reg_loss = 0.0
            for p in reg_param:
                reg_loss = reg_loss + tf.reduce_sum(tf.abs(p))

        if learning_rate_decay <= 0:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]
        else:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()
        self.sess.run(tf.global_variables_initializer())

    def get_param(self):
        return self.sess.run(self.param)

    def set_param(self, param_val):
        assert (len(param_val) == len(self.param_update_ops)), "The number of given parameters' value doesn't match with the number of parameters in the network."
        for param_cnt in range(len(self.param)):
            self.sess.run(self.param_update_ops[param_cnt], feed_dict={self.param_placeholders[param_cnt]: param_val[param_cnt]})

    def fit(self, X, y):
        train_mtl_neural_net(self.sess, [self.update, self.loss, self.get_param, self.set_param], [self.model_input, self.true_output, self.epoch], self.num_tasks, self.batch_size, self.max_epochs, self.patience, self.patience_multiplier, self.improvement_threshold, X, y, self.train_validation_ratio)

    def predict(self, X, k):
        assert (k < self.num_tasks and k > -1), "Given task exceed the index of tasks"
        return self.sess.run(self.eval[k], feed_dict={self.model_input[k]: X})