import json
from sklearn.metrics import accuracy_score, f1_score
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)
code2idx = {'D':0, 'M':1, 'S':2, 'H':3, 'F':4, 'O':5, 'E':6, 'NA':7}
idx2code = { code2idx[code]:code for code in code2idx }

def save_piechart(counts, meta=""):


    labels = list(code2idx.keys())
    sizes = counts
    #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',  shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.legend()
    plt.savefig(f"fig_pie_{meta}.png")

def get_session_stats(session, cond=lambda x: True):
    #[pp.pprint(session)

    preds = { code: 0.0 for code in code2idx.keys() }
    times = { code: 0.0 for code in code2idx.keys() }
    words = { code: 0.0 for code in code2idx.keys() }

    for utt_id in session:

        utt = session[utt_id]
        try:
            pred = idx2code[utt[0]]
        except:
            #pp.pprint(session)
            #pp.pprint(utt)
            pass

        speaker = utt[1]
        if not cond(speaker):
            continue
        time = utt[2]
        preds[pred] += 1
        times[pred] += time
        words[pred] += len(utt[3].split())

    stats = {
        "predictions": preds,
        "times": times,
        "words": words
    }
    return stats

def report_stat(results, stats_save_prefix="stats"):
    #print(results)
    # TODO1: gather by session
    sessions = results.keys()
    #print(sessions)

    session_results = {} # session_id: { utt_id: (prediction, length in ms) }

    # collect data into structure
    for session in sessions:
        session_results[session] = {}
        for window in results[session]:
            utt = window[0]
            pred = window[1]

            uttID = utt["uttID"]
            speaker = utt["speaker"]
            file = utt["file"]
            utt_content = utt["content"]
            session_results[session][uttID] = [pred, speaker, utt["end"]-utt["start"], utt_content, file]

    #pp.pprint(session_results)

    session_stats = {}
    for session in sessions:
        p_session_stats = get_session_stats(session_results[session], cond=lambda spk:spk=="P")
        n_session_stats = get_session_stats(session_results[session], cond=lambda spk:spk=="N")
        full_session_stats = get_session_stats(session_results[session])
        #pp.pprint(session_stats)
        session_stats[session] = {
            "P": p_session_stats,
            "N": n_session_stats,
            "Full": full_session_stats
        }

    fname = stats_save_prefix + "_sessions.json"
    with open(fname,"w") as fh:
        json.dump(session_stats, fh, indent=4)

    as_one_session_results = {}
    count = 0
    for session in sessions:
        for utt in session_results[session].values():
            as_one_session_results[count] = utt
            count += 1
    #pp.pprint(as_one_session_results)

    p_session_stats = get_session_stats(as_one_session_results, cond=lambda spk:spk=="P")
    n_session_stats = get_session_stats(as_one_session_results, cond=lambda spk:spk=="N")
    full_session_stats = get_session_stats(as_one_session_results)
    as_one_session_stats = {
        "full": {
        "P": p_session_stats,
        "N": n_session_stats,
        "Full": full_session_stats
        }
    }

    fname = stats_save_prefix + "_full.json"
    with open(fname,"w") as fh:
        json.dump(as_one_session_stats, fh, indent=4)

    save_piechart(p_session_stats["predictions"].values(), meta="pred_P")
    save_piechart(n_session_stats["predictions"].values(), meta="pred_N")
    save_piechart(full_session_stats["predictions"].values(), meta="pred_Full")

    return session_results