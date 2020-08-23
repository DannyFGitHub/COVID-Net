# Usage python3 inference-tflite.py --model_file
# /Users/<username>/Desktop/AndroidCovidNet/TFLITEModels/covidnet_b.tflite --label_file
# /Users/<username>/Desktop/AndroidCovidNet/TFLITEModels/labels.txt --image
# /Users/<username>/Desktop/AndroidCovidNet/images/pneu1115.jpg

# Inference On test folder of images labelled as class_size.jpeg python3 inference-tflite.py --model_file
# /Users/<username>/Desktop/AndroidCovidNet/TFLITEModels/covidnet_b.tflite --label_file
# /Users/<username>/Desktop/AndroidCovidNet/TFLITEModels/labels.txt --run_tests_on_folder
# /Users/<username>/Desktop/AndroidCovidNet/images

import argparse
import os
import time

import numpy as np
import tflite_runtime.interpreter as tflite

from data import process_image_file


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def run_inference_tflite(model_file_path, image, top_percent, input_size, label_file, verbose):
    if os.path.exists(model_file_path) and os.path.exists(image):
        interpreter = tflite.Interpreter(model_path=model_file_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        img = process_image_file(image, top_percent, input_size)
        img = img.astype('float32') / 255.0

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        # The following was suggested by an example script, but it was throwing off the results compared to the
        # non-tflite model after commenting out, the results become the same as the non-tflite model when classifying
        # the same photos with both models.
        # if floating_model:
        # input_data = (np.float32(input_data) - args.input_mean) / args.input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

        print("Prediction: " + labels[top_k[0]])

        # Print Further details
        if verbose:
            for i in top_k:
                if floating_model:
                    print('{:.3f}: {}'.format(float(results[i]), labels[i]))
                else:
                    print('{:.3f}: {}'.format(float(results[i] / 255.0), labels[i]))

            print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

        return labels[top_k[0]]

    else:
        print("One of the input files does not exist or couldn't be found.")
        return None


parser = argparse.ArgumentParser(description='COVID-Net Inference')
parser.add_argument('--model_file', default='models/COVIDNet-CXR3-S', type=str, help='Path to output folder')
parser.add_argument('--label_file', default='labels.txt', type=str, help='path to labels file')
parser.add_argument('--run_tests_on_folder', default=None, type=str, help='Path to model folder, '
                                                                          'where live test images can '
                                                                          'be found with format '
                                                                          'type_size.jpg')
parser.add_argument('--image', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
parser.add_argument('--input_mean', default=127.5, type=float, help='input_mean')
parser.add_argument('--input_std', default=127.5, type=float, help='input standard deviation')
parser.add_argument('--num_threads', default=None, type=int, help='number of threads')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
args = parser.parse_args()



test_folder = args.run_tests_on_folder


if test_folder is None:
    run_inference_tflite(args.model_file, args.image, args.top_percent, args.input_size, args.label_file, verbose=True)
else:
    if os.path.exists(test_folder):
        correct_count = 0
        incorrect_count = 0
        for file in os.listdir(test_folder):
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                prediction = run_inference_tflite(args.model_file, os.path.join(test_folder, file), args.top_percent, args.input_size, args.label_file, verbose=False)
                prediction_abbrev = prediction[:5].lower()
                if file.lower().startswith(prediction_abbrev):
                    correct_count += 1
                else:
                    incorrect_count += 1
        print("Correct Count: " + str(correct_count))
        print("Incorrect Count: " + str(incorrect_count))
        print("Total: " + str(correct_count + incorrect_count))
    else:
        print('No tests run as test images folder could not be found.')
