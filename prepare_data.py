import json
import numpy as np
from tqdm import tqdm, trange
import pickle
import random
import glob
import pprint
from collections import Counter

#SLIDE=False
CHUNKSIZE=5 #3 #5
BUFF=100


def chunks(lst, n, SLIDE=False):
    """Yield successive n-sized chunks from lst."""
    if SLIDE:
        increment = 1
    else:
        increment = n
    for i in range(0, len(lst), increment):
        yield lst[i:i + n]


def dic2sentence(dic):
    try:
        utt = dic["content"]
        speaker = dic["speaker"]

        if speaker == "N":
            spk = "T"
        else:
            spk = "C"
        return spk+": "+utt.strip()
    except:
        print("error")
        print(dic)
        return None

def prep_train_data(train_path):
    pp = pprint.PrettyPrinter(indent=4)

    paths = glob.glob(train_path+"/"+"*.json")

    data = []
    for fp in paths:
        session_utts = []
        with open(fp, "r") as fh:
            session = json.load(fh)
        for utt_id in session:
            utt = session[utt_id]

            utt["uttID"] = utt_id

            session_utts.append(utt)
        chunked = chunks(session_utts, CHUNKSIZE, SLIDE=True)
        data += list(chunked)


    processed = {}
    for idx, datum in enumerate(tqdm(data)):
        field = datum
        processed[idx] = field

    datapath = f"./train_data/GRADE{CHUNKSIZE}.pkl"
    with open(datapath, "wb") as fh:
        pickle.dump(processed, fh)

    return datapath



def prep_test_data(test_path, data_path=None):
    if data_path != None:
        return data_path
    pp = pprint.PrettyPrinter(indent=4)
    transcribed_paths = glob.glob(test_path+"/**/*.json")


    data = []
    for sid, fp in enumerate(tqdm(transcribed_paths)):
        session_utts = []
        with open(fp, "r") as fh:
            session = json.load(fh)
        try:
            words = session['results'][-1]['alternatives'][0]["words"]
        except:
            # amazon handling (skip for just now)
            continue
        if True:
            utts = [ x['alternatives'][0]['transcript'] for x in session['results'][:-1] ]

            uwords = [x['speakerTag'] for x in words[0:0 + len(utts[0].split())]]
            first_spk = Counter(uwords).most_common(1)[0][0]
            idx2spk = {first_spk: "N", int(3 - first_spk): "P"}

            wdx = 0
            for uid, utt in enumerate(utts):
                uwords = [ x['speakerTag'] for x in words[wdx:wdx+len(utt.split())] ]
                spk = Counter(uwords).most_common(1)[0][0]
                uwords = [x for x in words[wdx:wdx + len(utt.split())]]
                field = {
                    "sessionID": sid + BUFF,
                    "start": float(uwords[0]["startTime"][:-1]) * 1000.0,
                    "end": float(uwords[-1]["endTime"][:-1]) * 1000.0,
                    "content": utt,
                    "speaker": idx2spk[spk],
                    "file": fp,
                    "uttID": len(session_utts)
                }
                session_utts.append(field)
                wdx += len(utt.split())
                #print()
                #print(field)

            #exit()

        if False:
            curr_spk = words[0]["speakerTag"]
            idx2spk = { curr_spk:"N", int(3-curr_spk):"P"}
            curr_start = words[0]["startTime"]
            curr_end = words[0]["endTime"]
            curr_utt = ""
            for wid, word in enumerate(words):
                if curr_spk != word["speakerTag"]:
                    session_utts.append({
                        "sessionID": sid + BUFF,
                        "start": float(curr_start[:-1])*1000.0,
                        "end": float(curr_end[:-1])*1000.0,
                        "content": curr_utt,
                        "speaker":idx2spk[curr_spk],
                        "file":fp,
                        "uttID":len(session_utts)
                    })
                    curr_spk = word["speakerTag"]
                    curr_utt = word["word"]
                    curr_start = word["startTime"]
                else:
                    curr_end = word["endTime"]
                    curr_utt += " " + word["word"]

        chunked = chunks(session_utts, CHUNKSIZE, SLIDE=True)
        data += list(chunked)


    processed = {}
    for idx, datum in enumerate(tqdm(data)):
        field = datum
        processed[idx] = field

    datapath = f"./test_data/GRADE{CHUNKSIZE}.pkl"
    with open(datapath, "wb") as fh:
        pickle.dump(processed, fh)

    return datapath
