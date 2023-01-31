import argparse, sys, json
import queue, time, threading

import numpy as np
import tkinter as tk
from itertools import count

import matplotlib.pyplot as plt
from PIL import Image, ImageTk

import sounddevice as sd
from vosk import Model, KaldiRecognizer

class SpeechText:
    def __init__(self, sentence_queue):
        self.init()
        self.__q = queue.Queue()
        self.running = True
        self.sentence_queue = sentence_queue

    def getParser(self):
        return self.__parser

    def int_or_str(self, text):
        try:
            return int(text)
        except ValueError:
            return text

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.__q.put(bytes(indata))

    def init(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-l", "--list-devices", action="store_true", help="show list of audio devices and exit")
        args, remaining = parser.parse_known_args()

        if args.list_devices:
            print(sd.query_devices())
            parser.exit(0)

        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[parser])

        parser.add_argument("-f", "--filename", type=str, metavar="FILENAME", help="audio file to store recording to")
        parser.add_argument("-d", "--device", type=self.int_or_str, help="input device (numeric ID or substring)")
        parser.add_argument("-r", "--samplerate", type=int, help="sampling rate")
        parser.add_argument("-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
        args = parser.parse_args(remaining)

        self.__args = args
        self.__parser = parser
    
    def speech_to_text(self):

        results = []
        textResults = []

        if self.__args.samplerate is None:
            device_info = sd.query_devices(self.__args.device, "input")
            self.__args.samplerate = int(device_info["default_samplerate"])
                
        if self.__args.model is None:
            model = Model(lang="en-us")
        else:
            model = Model(lang=self.__args.model)

        if self.__args.filename:
            dump_fn = open(self.__args.filename, "wb")
        else:
            dump_fn = None

        with sd.RawInputStream(samplerate=self.__args.samplerate, blocksize = 8000, device=self.__args.device, dtype="int16", channels=1, callback=self.callback):
            print("#" * 80)
            print("Press Ctrl+C to stop the recording")
            print("#" * 80)

            rec = KaldiRecognizer(model, self.__args.samplerate)
            while self.running:
                data = self.__q.get()
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    resultDict = json.loads(result)
                    results.append(resultDict)
                    text = resultDict.get("text", "")
                    if(text!=''):
                        textResults.append(text)
                        sentence_queue.put(text)
                else:
                    print(rec.PartialResult())
                if dump_fn is not None:
                    dump_fn.write(data)
            
            print('\n\n\nFinished:', textResults)


images_map = {}
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def load_images():
    for letter in letters:
        image = Image.open('letters/' + letter + '.jpg')
        imageArr = np.asarray(image)
        images_map[letter] = imageArr

def show_images(sq):
    fig, ax = plt.subplots(1,1)
    image = np.array([[1,1,1], [2,2,2], [3,3,3]])
    im_show_method = ax.imshow(image)
    
    while(True):
        time.sleep(1)
        text = sq.get()
        text = text.lower().replace(" ", "")
        try:
            for c in text:
                im_show_method.set_data(images_map[c])
                fig.canvas.draw_idle()
                plt.pause(0.2)
        except(Exception):
            pass

def speech_text_callback(sto):
    sto.speech_to_text()

if __name__ == '__main__':

    sentence_queue = queue.Queue()
    load_images()

    speech_text_object = SpeechText(sentence_queue)
    speech_text_thread = threading.Thread(target=speech_text_callback, args=([speech_text_object]))

    try:
        speech_text_thread.start()
        show_images(sentence_queue)

    except(KeyboardInterrupt):
        speech_text_object.running = False
        speech_text_thread.join()
        exit(0)