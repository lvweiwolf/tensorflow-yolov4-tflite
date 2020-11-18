import os
import sys
import tensorflow as tf



model_dir ='./checkpoints/yolov4-tiny-416/frozen_models'
freezefile = 'simple_frozen_graph.pb'

gpuConfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True


def display():
    # 重置计算图
    tf.compat.v1.reset_default_graph()
    model_path = os.path.join(model_dir, freezefile)
    with tf.compat.v1.Session(config=gpuConfig) as sess:
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        output_graph_def = tf.compat.v1.GraphDef()
        # 获得默认的图
        graph = tf.compat.v1.get_default_graph()
        with open(model_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.compat.v1.import_graph_def(output_graph_def, name="")
            # 当前计算图中有多少个操作节点
            print('%d ops in the final graph.' % len(output_graph_def.node))
            
            tensor_name = [tensor.name for tensor in output_graph_def.node]
            print(tensor_name)
            print('---------------------------')
            # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
            summaryWriter = tf.compat.v1.summary.FileWriter(os.path.join(model_dir, 'log_graph'), graph)
            # for op in graph.get_operations():
            #     # print出tensor的name和值
            #     print(op.name, op.values())
            

 
if __name__ == '__main__':
    display()