# Recommended tensorflow version is 1.15.x for the CovidNet models dated before Aug 2020
import argparse
import os
import tensorflow as tf
from tensorflow import keras


def freeze_graph():
    with tf.compat.v1.Session() as sess:
        # Restore the graph
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(root_folder, model_folder_name, meta_name))

        print(os.path.join(root_folder, model_folder_name, checkpoint_filename))

        # To get node names run:
        print(sess.graph.get_operations())

        # Load weights
        saver.restore(sess, os.path.join(root_folder, model_folder_name, checkpoint_filename))

        # Freeze the graph
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph creating a Frozen-Graph folder (depends on input, if user gives a Frozen-Graph folder as input)
        if not os.path.exists(os.path.dirname(os.path.join(root_folder, output_frozen_graph_name))):
            try:
                os.makedirs(os.path.dirname(os.path.join(root_folder, output_frozen_graph_name)))
            except OSError as exc:  # Guard against race condition
                print("Couldn't make folders for output frozen graph file")
                exit()

        with open(os.path.join(root_folder, output_frozen_graph_name), 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())


def list_nodes_in_graph():
    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(os.path.join(root_folder, model_folder_name, meta_name))

        print(os.path.join(root_folder, model_folder_name, checkpoint_filename))

        # To get node names run
        name_array = []
        for operation in sess.graph.get_operations():
            name_array.append(operation.name)

        with open(os.path.join(root_folder, 'node_names.txt'), 'w') as f:
            f.write(str(name_array))

        print('Wrote array to file.')


def convert_model():
    graph_file_path = os.path.join(root_folder, output_frozen_graph_name)
    # Input_1 is the input nodes from the documentation
    input_arrays = ['input_1']
    output_arrays = output_node_names
    # Netron software can show you what the input signature/parameters are:
    input_shapes = {'input_1': [None, 480, 480, 3]}

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=graph_file_path,
        input_arrays=input_arrays,
        output_arrays=output_arrays,
        input_shapes=input_shapes
    )

    # Optimise
    if str(optimisation).lower() == 'default':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif str(optimisation).lower() == '16float':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_model_size = open(os.path.join(root_folder, tflite_converted_model_name), "wb").write(tflite_model)
    print('TFLite Model is %d bytes' % tflite_model_size)



parser = argparse.ArgumentParser(description='CovidNetModel To TFlite Converter')
parser.add_argument('--root_folder', default='', type=str, help='Path to root folder enclosing all the models on Mac or Linux: /Users/<username>/Desktop/AndroidCovidNet/')
parser.add_argument('--model_folder_name', default='COVIDNet-CXR4-A/', type=str, help='Path To Model Folder from Root')
parser.add_argument('--meta_name', default='model.meta', type=str, help='File Name of the .meta covidnet model file')
parser.add_argument('--checkpoint_filename', default='model-18540', type=str, help='File Name of the .meta covidnet model file')
parser.add_argument('--output_node_name', default='norm_dense_1/Softmax', type=str, help='Name of the output node')
parser.add_argument('--output_frozen_graph_name', default='Frozen-Graph/output_frozen_graph_A.pb', type=str, help='Name of the output frozen graph')
parser.add_argument('--tflite_converted_model_name', default='converted_model_a_16optim.tflite', type=str, help='File name of the output converted model')
parser.add_argument('--optimisation', default='Default', type=str, help='Optimisation is either: None, Default, 16Float')
args = parser.parse_args()


root_folder = args.root_folder
model_folder_name = args.model_folder_name
meta_name = args.meta_name  # e.g. Your .meta file
checkpoint_name = args.checkpoint_filename  # e.g. model-18540.data-00000-of-00001
output_node_names = [args.output_node_name]  # Output nodes
output_frozen_graph_name = args.output_frozen_graph_name
tflite_converted_model_name = args.tflite_converted_model_name
optimisation = args.optimisation

# First freeze the graph.
freeze_graph()

#list_nodes_in_graph()

# Second Convert the Graph tflite
convert_model()