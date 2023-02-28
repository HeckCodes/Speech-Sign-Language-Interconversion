import argparse, sys, json
import queue, time, threading

import os
import cv2

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

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

gifs_path = "D://sixthsemester//tarp//project//Speech-Sign-Language-Interconversion//ISL_Gifs"
isl_gifs = os.listdir(gifs_path)

def convert_gif_to_frames(gif):

    frame_num = 0
    frame_list = []

    while True:
        try:
            okay, frame = gif.read()
            frame_list.append(frame)
            if not okay:
                break
            frame_num += 1
        except KeyboardInterrupt:
            break

    return frame_list

isl_gif_reverse_lookup = {x: convert_gif_to_frames(cv2.VideoCapture("ISL_Gifs/" + x)) for x in isl_gifs}
images_map = {letter: cv2.imread("letters/" + letter + ".jpg", 0) for letter in letters}

def show_isl_loop(sq):
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 360, 240)
    
    while(True):
        cv2.waitKey(1000)
        text = sq.get()
        try:
            if(isl_gif_reverse_lookup.get(text + ".gif") != None):
                gif_frames = isl_gif_reverse_lookup.get(text + ".gif")
                for frame in gif_frames:
                    cv2.imshow('Resized_Window', frame)
                    cv2.waitKey(50)
            else:
                text = text.lower().replace(" ", "")
                for c in text:
                    cv2.imshow('Resized_Window', images_map[c])
                    cv2.waitKey(350)
        except(Exception):
            pass

def speech_text_callback(sto):
    sto.speech_to_text()

if __name__ == '__main__':

    sentence_queue = queue.Queue()
    speech_text_object = SpeechText(sentence_queue)
    speech_text_thread = threading.Thread(target=speech_text_callback, args=([speech_text_object]))

    try:
        speech_text_thread.start()
        show_isl_loop(sentence_queue)

    except(KeyboardInterrupt):
        speech_text_object.running = False
        speech_text_thread.join()
        exit(0)