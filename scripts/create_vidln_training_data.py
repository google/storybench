# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os.path
from pathlib import Path
import sys

import cv2
import numpy as np
import tqdm
import glob

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex


LLM_KEYS = ['llm_split_1', 'llm_split_2']


# ============================================================================ #
#                                     Utils                                    #
# ============================================================================ #
def custom_tokenizer(nlp):
    # Default infixes
    inf = list(nlp.Defaults.infixes)
    # Remove the generic op between numbers or between a number and a -
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")
    # Convert inf to tuple
    inf = tuple(inf)
    # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])
    # Remove - between letters rule
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x]
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)


def frame_number_from_filename(filename: str) -> int:
    stem = Path(filename).stem.removeprefix('img_')
    return int(stem)


def _match_subseq(l1, l2):
    remain1 = set([i for i in range(len(l1)+1)])
    remain2 = set([i for i in range(len(l2)+1)])
    res1 = []
    res2 = []
    for w in range(len(l1), 0, -1):
        for st1 in range(0, len(l1)-w+1):
            # iterate over all possible subseq of length w
            for st2 in range(0, len(l2)-w+1):
                if (st1 in remain1 and st1+w-1 in remain1 and st2 in remain2 and
                    st2+w-1 in remain2 and l1[st1:st1+w] == l2[st2:st2+w]):
                    res1.append((st1, st1+w))
                    res2.append((st2, st2+w))
                    for e in range(st1, st1+w):
                        remain1.remove(e)
                    for e in range(st2, st2+w):
                        remain2.remove(e)
    argsort = sorted(range(len(res1)), key=lambda x: res1[x][0])
    res1 = [res1[ix] for ix in argsort]
    res2 = [res2[ix] for ix in argsort]
    return res1, res2


def _merge_keyframes_captions(maj_kfs, captions):
    sents = []
    kfs = []
    cur_sent = ''
    cur_kf = ''
    for sent, kf in zip(captions, maj_kfs):
        if kf == cur_kf:
            # append
            cur_sent += ' ' + sent.strip()
        else:
            if len(cur_sent):
                # add to new sents and maj kfs 
                sents.append(cur_sent)
                kfs.append(cur_kf)
            # new
            cur_sent = sent.strip()
            cur_kf = kf
    # add last sent/kf
    sents.append(cur_sent)
    kfs.append(cur_kf)
    return sents, kfs


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# ============================================================================ #
#                            Preprocessing pipeline                            #
# ============================================================================ #
#0 Load dataset
def load_data(vidln_fn, llm_fn):
    with open(llm_fn) as f:
        llm_annos = json.load(f)
    llm_annos = {e['vidln_id']: e for e in llm_annos}

    data = []
    with open(vidln_fn) as jsonl:
        for line in tqdm.tqdm(jsonl):
            j = json.loads(line)
            vidln_id = j['vidln_id']

            j2 = llm_annos[vidln_id]

            for ix in range(len(j['actor_narratives'])):
                actor_name = j['actor_narratives'][ix]['actor_name']
                actor_name2 = j2['actor_narratives'][ix]['actor_name']
                assert actor_name == actor_name2
                
                for llm_key in LLM_KEYS:
                    llm_caption = j2['actor_narratives'][ix][llm_key]
                    # fix LLM issues
                    if len(llm_caption):
                        if llm_caption[0] == '.':
                            llm_caption = llm_caption[1:]
                        llm_caption = llm_caption.replace('. .', '.')
                    j['actor_narratives'][ix][llm_key] = llm_caption

            data.append(j)

    return data


#1 Map each word to all its keyframes
def map_word_keyframes(data):
    for j in tqdm.tqdm(data):
        for actor_idx in range(len(j['actor_narratives'])):

            actor_d = j['actor_narratives'][actor_idx]

            actor_d['actor_name'] = actor_d['actor_name'].strip()

            # get timestamp to keyframe mapping
            timestamp_to_kf = {}
            for tr_list in actor_d['traces']:
                for tr_e in tr_list:
                    timestamp_to_kf[tr_e['time_ms_since_epoch']] = (
                        j['keyframe_names'][tr_e['kf_idx']])

            # add keyframe(s) to each word
            for ix, ta_e in enumerate(actor_d['time_alignment']):
                ta_e['keyframes'] = set()
                start = ta_e['start_ms'] + (
                    actor_d['recording_start_time_ms_since_epoch'])
                end = actor_d['recording_start_time_ms_since_epoch']
                if ix+1 == len(actor_d['time_alignment']):
                    end += ta_e['end_ms']
                else:
                    end += actor_d['time_alignment'][ix+1]['start_ms']
                for t, kf in timestamp_to_kf.items():
                    if start <= t <= end:
                        ta_e['keyframes'].add(kf)
                ta_e['keyframes'] = sorted(
                    ta_e['keyframes'], key=frame_number_from_filename)


#2 Map keyframes from word in transcript to word in split transcript
def map_keyframes_to_llm(data, nlp):
    """
    Choose LLM output with largest number of carried lemmas.
    """
    for narr_e in tqdm.tqdm(data):
        for actor_e in narr_e['actor_narratives']:

            # Map each lemma in the original caption to its keyframe(s)
            words_ta = [word_d['referenced_word']
                        for word_d in actor_e['time_alignment']]
            kfs = [word_d['keyframes'] for word_d in actor_e['time_alignment']]
            doc_ta = nlp(' '.join(words_ta))
            lemma_kfs = []
            tok_word = ''
            word_ix = 0
            for token in doc_ta:
                lemma_kfs.append(kfs[word_ix])
                tok_word += token.text
                if tok_word == words_ta[word_ix]:
                    tok_word = ''
                    word_ix += 1
            assert len(doc_ta) == len(lemma_kfs)
            lemma_kfs = [(token.lemma_, lemma_kfs[ix])
                         for ix, token in enumerate(doc_ta)]
            lem_ta = [token.lemma_ for token in doc_ta]
            
            # match words through lemmas (allowing to match eg putting -> puts)
            lm_docs_dict = {k: nlp(actor_e[k]) for k in LLM_KEYS}
            lm_lemmas_dict = {k: [token.lemma_ for token in v]
                              for k, v in lm_docs_dict.items()}

            # choose LLM output with highest number of carried lemmas
            max_len_matched_ixs = -1
            lm_matched_ixs, doc_lm, rng_ta, rng_lm = None, None, None, None
            for k, lem_k in lm_lemmas_dict.items():
                rng_t, rng_k = _match_subseq(lem_ta, lem_k)
                k_matched_ixs = []
                for (s, e) in rng_t:
                    k_matched_ixs += list(range(s, e))
                if len(k_matched_ixs) > max_len_matched_ixs:
                    max_len_matched_ixs = len(k_matched_ixs)
                    doc_lm = lm_docs_dict[k]
                    rng_ta = rng_t
                    rng_lm = rng_k
                    lm_matched_ixs = k_matched_ixs
                    actor_e['caption_lm'] = actor_e[k]
            assert len(lm_matched_ixs)

            # map keyframes from original caption to LM caption
            mapped_kfs = 0
            kfs_tr = [[] for _ in range(len(doc_lm))]
            for ((st, et), (sf, ef)) in zip(rng_ta, rng_lm):
                t_ixs = list(range(st, et))
                f_ixs = list(range(sf, ef))
                for t_ix, f_ix in zip(t_ixs, f_ixs):
                    kfs_tr[f_ix] = lemma_kfs[t_ix][1]
                    mapped_kfs += (len(lemma_kfs[t_ix][1]) > 0)
            tot_kfs = sum(len(t[1]) > 0 for t in lemma_kfs) or 1

            actor_e['caption_lm_kfs'] = kfs_tr
            actor_e['caption_lm_lemma_perc'] = len(lm_matched_ixs) / len(lem_ta)
            actor_e['caption_lm_kfs_perc'] = mapped_kfs / tot_kfs
            actor_e['caption_lm_tok'] = []
            actor_e['caption_lm_pos'] = []
            actor_e['caption_lm_tag'] = []
            actor_e['caption_lm_dep'] = []
            for token in doc_lm:
                actor_e['caption_lm_tok'].append(token.text)
                actor_e['caption_lm_pos'].append(token.pos_)
                actor_e['caption_lm_tag'].append(token.tag_)
                actor_e['caption_lm_dep'].append(token.dep_)


#3 Split into sentences and keyframe majority voting
def map_sentences_to_keyframes(data):
    """
    Choose smaller keyframe when more available (action already visible early)
    Majority of min keyframes for verbs if avail, otherwise all words.
    If no keyframe, use None.
    """
    for narr_e in tqdm.tqdm(data):
        for actor_e in narr_e['actor_narratives']:
            sentences = []
            sentences_kf = []
            sentence, verbs_kfs, words_kfs, all_kfs = [], [], [], []
            sent_ix = 0
            for ix, (tok, pos) in enumerate(zip(actor_e['caption_lm_tok'],
                                                actor_e['caption_lm_pos'])):
                if len(actor_e['caption_lm_kfs'][ix]):
                    words_kfs.append(min(actor_e['caption_lm_kfs'][ix]))
                    all_kfs += actor_e['caption_lm_kfs'][ix]
                    if pos == 'VERB':
                        verbs_kfs.append(min(actor_e['caption_lm_kfs'][ix]))
                if tok == '.':
                    sentences.append(' '.join(sentence) + '.')

                    sent_ix += 1
                    # use maj voting over min kfs per token
                    if len(verbs_kfs):
                        sentences_kf.append(max(set(verbs_kfs), key=verbs_kfs.count))
                    elif len(words_kfs):
                        sentences_kf.append(max(set(words_kfs), key=words_kfs.count))
                    else:
                        sentences_kf.append(None)   
                    sentence, verbs_kfs, words_kfs = [], [], []
                else:
                    sentence.append(tok)

            if len(sentence):
                sentences.append(' '.join(sentence))
                # use maj voting over min kfs per token
                if len(verbs_kfs):
                    sentences_kf.append(max(set(verbs_kfs), key=verbs_kfs.count))
                elif len(words_kfs):
                    sentences_kf.append(max(set(words_kfs), key=words_kfs.count))
                else:
                    sentences_kf.append(None)
            assert len(sentences)
            if all_kfs:
                sentences_kf[0] = min(all_kfs)
                sentences_kf[-1] = max(all_kfs)

            actor_e['captions_lm_tok'] = sentences
            actor_e['captions_lm_kfs'] = sentences_kf


#4 Merge captions with same majority keyframe
def merge_same_keyframe_captions(data):
    for narr_e in tqdm.tqdm(data):
        for actor_e in narr_e['actor_narratives']:
            # combine frames and captions
            sents, kfs = _merge_keyframes_captions(actor_e['captions_lm_kfs'],
                                                   actor_e['captions_lm_tok'])

            if None in kfs:
                # interpolate keyframes
                kf2idx = {narr_e['keyframe_names'][v]: v
                          for v in actor_e['keyframe_selection_indices']}
                idx2kf = {v: k for k, v in kf2idx.items()}
                last_ix = kf2idx[idx2kf[min(idx2kf.keys())]]
                new_kfs = [idx2kf[last_ix]]
                for ix, kf in enumerate(kfs[1:-1]):
                    if kf is None:
                        kf_ix = (kf2idx[kfs[ix]] + kf2idx[kfs[ix+1+1]]) // 2
                        kf_ix = _find_nearest(list(idx2kf.keys()), kf_ix)
                        new_kfs.append(idx2kf[kf_ix])
                new_kfs.append(kfs[-1])
                # combine frames and captions
                sents, kfs = _merge_keyframes_captions(new_kfs, sents)
            
            if None in kfs:
                import pdb; pdb.set_trace()
            assert None not in kfs
            actor_e['captions_lm_merge_tok'] = sents
            actor_e['captions_lm_merge_kfs'] = kfs


#5 Map sentences to video frames
def map_sentences_to_video_frames(data, video_path):
    """
    Split in the middle between two majority keyframes.
    If keyframes in sentence are not sorted, use None.
    """
    for narr_e in tqdm.tqdm(data):
        video_fn = glob.glob(
            os.path.join(video_path, f"{narr_e['video_id']}*"))[0]
        cap = cv2.VideoCapture(video_fn)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = [i for i in range(frame_count)]
        for actor_e in narr_e['actor_narratives']:
            kfs = actor_e['captions_lm_merge_kfs']

            kfs_int = [frame_number_from_filename(k) for k in kfs]
            mids = [(k1+k2)//2 for k1, k2 in zip(kfs_int[:-1], kfs_int[1:])]
            sentences_frames = []
            sent_frames = []
            cur_mid_ix = 0
            for frame in frames:
                sent_frames.append(frame)
                if len(mids) and len(mids) > cur_mid_ix and (
                    frame == mids[cur_mid_ix]):
                    cur_mid_ix += 1
                    sentences_frames.append(sent_frames)
                    sent_frames = []
            sentences_frames.append(sent_frames)
            actor_e['captions_lm_merge_frames'] = sentences_frames

            if sorted(kfs, key=frame_number_from_filename) != kfs:
                actor_e['captions_lm_merge_tok'] = [actor_e['caption']]
                actor_e['captions_lm_merge_kfs'] = [narr_e['keyframe_names'][
                    actor_e['keyframe_selection_indices'][0]]]
                actor_e['captions_lm_merge_frames'] = [frames]

            assert len(actor_e['captions_lm_merge_frames']) == (
                len(actor_e['captions_lm_merge_tok']))


def pipeline(vidln_fn, llm_fn, video_path, nlp):
    print("Loading data...")
    data = load_data(vidln_fn, llm_fn)
    print("Mapping words to keyframes...")
    map_word_keyframes(data)
    print("Mapping keyframes to LLM outputs...")
    map_keyframes_to_llm(data, nlp)
    print("Aggregating keyframes per sentence...")
    map_sentences_to_keyframes(data)
    print("Merging sentences with the same keyframes...")
    merge_same_keyframe_captions(data)
    print("Mapping sentences to video frames...")
    map_sentences_to_video_frames(data, video_path)
    return data


# ============================================================================ #
#                                     MAIN                                     #
# ============================================================================ #
def main(vidln_fn, llm_fn, video_path, fps_path, out_path):
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = custom_tokenizer(nlp)

    with open(os.path.join(fps_path, 'video_to_fps.json')) as f:
        video_to_fps = json.load(f)

    full_data = pipeline(vidln_fn, llm_fn, video_path, nlp)
    with open(out_path, 'w') as f:
        json.dump(full_data, f)

    data = []
    for j in full_data:
        elem = {k: j[k] for k in ['vidln_id', 'dataset_id', 'video_id',
                                  'keyframe_names', 'annotator_id']}

        actors = []
        for actor_e in j['actor_narratives']:

            if actor_e['actor_name'].lower() == 'background':
                actors.append(actor_e)
                continue
            
            # Make sure caption is not empty
            if len(actor_e['captions_lm_merge_tok']) < 1: continue

            # Keep captions that carried at least 70% of the original lemmas
            if actor_e['caption_lm_lemma_perc'] < 0.7: continue

            kfs = actor_e['captions_lm_merge_kfs']
            assert sorted(kfs, key=frame_number_from_filename) == kfs

            for fs in actor_e['captions_lm_merge_frames']:
                assert None not in fs

            # Keep actors whose first caption is at least 1 second long
            if len(actor_e['captions_lm_merge_frames'][0]) < (
                1. * video_to_fps[j['video_id']]):
                continue
            
            # If only one caption after merging, use the original caption
            if len(actor_e['captions_lm_merge_tok']) == 1:
                actor_e['captions_lm_merge_tok'] = [actor_e['caption']]

            actors.append(actor_e)
        
        if actors:
            elem['actor_narratives'] = actors
            data.append(elem)

    with open(out_path, 'w') as f:
        json.dump(data, f)
    

if __name__ == "__main__":
    vidln_fn = sys.argv[1]
    llm_fn = sys.argv[2]
    video_path = sys.argv[3]
    fps_path = sys.argv[4]
    out_path = sys.argv[5]
    main(vidln_fn, llm_fn, video_path, fps_path, out_path)
