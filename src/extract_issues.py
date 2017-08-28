# Read reviews and extract issues, including memory, CPU, battery, and traffic.
# Use word2vec and kmeans to find the similar words
# -*- coding: utf-8 -*-
# __author__  = "Cuiyun Gao"
# __version__ = "1.0"


import os
import logging
import itertools
import json
import re
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import Word2Vec, Phrases, LdaMulticore, TfidfModel
from sklearn.cluster import KMeans, SpectralClustering
from extractSentenceWords import *
from collections import defaultdict

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

keys = ["ad", "ads"]
issues = ["memory", "cpu", "slow", "network", "battery"]
add_issues = ["notification", "download", "charge", "update", "functionality"]
similar_num = 50

bigram = None
trigram = None

# Get cleaned mongodb reviews, including 1547 apps
def get_all_reviews():
    doc_reviews = {}
    sent_reivews = {}
    # reviews_sents = []
    # reviews_docs = []
    rate_sents = []
    rate_docs = []
    num_docs = 0
    num_words = 0
    apk_path = os.path.join("..", "data", "raw")
    apk_names = os.listdir(os.path.join(apk_path, "mongodb"))
    apk_review_paths = [os.path.join(apk_path, "mongodb", apk_name, "review.txt") for apk_name in apk_names]
    apk_rate_paths = [os.path.join(apk_path, "mongodb", apk_name, "info.txt") for apk_name in apk_names]
    for root, dirs, files in os.walk(apk_path, "mysql"):
        for name in files:
            filename = os.path.join(root, name)
            if re.match(r'.*clean_review\.txt', filename):
                apk_review_paths.append(filename)
                apk_names.append(name)
            if re.match(r'.*clean_data\.txt', filename):
                apk_rate_paths.append(filename)

    # =============== validation =================
    # apk_review_paths = ["/home/jczeng/workspace/AppReview/data/raw/mysql/Specific_apps/Skype___free_IM___video_calls_gp/Skype___free_IM___video_calls_gp_clean_review.txt"]
    # apk_rate_paths = ["/home/jczeng/workspace/AppReview/data/raw/mysql/Specific_apps/Skype___free_IM___video_calls_gp/Skype___free_IM___video_calls_gp_clean_data.txt"]

    # =============== validation =================

    for idx, item in enumerate(apk_review_paths):
        logging.info(item)
        with open(item) as fin, open(apk_rate_paths[idx]) as frin:
            apk_name = apk_names[idx]
            review_lines = fin.readlines()
            rate_lines = frin.readlines()
            reviews_sents = []
            reviews_docs = []
            rate_sents = []
            rate_docs = []
            if len(review_lines) != len(rate_lines):
                logging.error("length not equal at %s"%item)
            for j, line in enumerate(review_lines):
                words_sents, wc = extractSentenceWords(line)
                reviews_sents.append(words_sents)
                reviews_docs.append(list(itertools.chain.from_iterable(words_sents)))
                num_docs += 1
                num_words += wc
                r_line = rate_lines[j].strip()
                r_line = r_line.split("******")
                if len(r_line) > 6:
                    if not re.match(r'\d*\.?\d+', r_line[5]):
                        logging.error("rate not float at %s in %s line: %d"%(r_line[5], item, j))
                        rate = 2.0
                    else:
                        rate = float(r_line[5])
                else:
                    rate = float(r_line[0])
                rate_sents.append([rate for w in words_sents])
                rate_docs.append(rate)
            sent_reivews[apk_name] = (trigram[bigram[reviews_sents]], rate_docs)
            doc_reviews[apk_name] = (trigram[bigram[reviews_docs]], rate_docs)
    logging.info("Read %d docs, %d words!" % (num_docs, num_words))
    # reviews_sents = list(itertools.chain.from_iterable(reviews_sents))
    # rate_sents = list(itertools.chain.from_iterable(rate_sents))
    return sent_reivews, doc_reviews, rate_sents, rate_docs

def get_test_reviews():
    doc_reviews = {}
    sent_reivews = {}
    num_docs = 0
    num_words = 0
    apk_path = os.path.join("..", "data", "raw")
    apk_lst_path = os.path.join(apk_path, "package_names.txt")
    # load phrases
    bigram = Phrases.load(os.path.join("..", "model", "bigram.model"))
    trigram = Phrases.load(os.path.join("..", "model", "trigram.model"))
    with open(apk_lst_path) as fin:
        apk_lst = [apk_name.strip() for apk_name in fin.readlines()]
    for apk_name in apk_lst:
        file = os.path.join(apk_path, "mongodb", apk_name, "review.txt")
        with open(file) as fin:
            reviews_sent = []
            reviews_doc = []
            for line in fin.readlines():
                words_sents, wc = extractSentenceWords(line)
                reviews_sent.append(words_sents)
                reviews_doc.append(list(itertools.chain.from_iterable(words_sents)))
                num_docs += 1
                num_words += wc
            sent_reivews[apk_name] = trigram[bigram[reviews_sent]]
            doc_reviews[apk_name] = trigram[bigram[reviews_doc]]

    logging.info("Read %d docs, %d words!" % (num_docs, num_words))
    return sent_reivews, doc_reviews


def extract_phrases(reviews_sents, reviews_docs, save=False):
    logging.info("Extracting phrases...")
    bigram = Phrases(reviews_sents, threshold=5, min_count=5)
    trigram = Phrases(bigram[reviews_sents], threshold=3, min_count=3)
    if save:
        with open('../data/phrase/phrases_%d_%s' % (3, 'app_review'), 'wb') as fout:
            ph_dic = {}
            for phrase, score in bigram.export_phrases(reviews_sents):
                ph_dic[phrase] = score
            for phrase, score in trigram.export_phrases(bigram[reviews_sents]):
                ph_dic[phrase] = score
            for phrase, score in ph_dic.items():
                if re.search(r'\d+', phrase):  # remove digits
                    continue
                phrase = b"_".join(phrase.split(b' '))
                fout.write(phrase + b'\n')
        bigram.save("../model/bigram.model")
        trigram.save("../model/trigram.model")

    return trigram[bigram[reviews_docs]]


# Input "reviews" is a list of sentences, whose words are split.
def training(reviews):
    # convert glove format to word2vec, we use the twitter model of 200 dimensions from https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models
    # glove_model_path = os.path.join("..", "model", "glove.twitter.27B", "glove.twitter.27B.200d.txt")
    # word2vec_pre_model = os.path.join("..", "data", "pre_twitter_word2vec.model")
    # glove2word2vec.glove2word2vec(glove_model_path, word2vec_pre_model)


    # laoding the pre-trained model and retrain with the reviews
    logging.info("Training word2vec...")
    model = Word2Vec(reviews, size=200, min_count=3, workers=8)
    logging.info("Saving word2vec model...")
    model.save(os.path.join("..", "model", "appreviews_word2vec.model"))
    # model = Word2Vec.load_word2vec_format(word2vec_pre_model, binary=False)
    # model.build_vocab(sentences, update=True)
    # model.train(sentences)
    #
    # # model = Word2Vec(bigram_transformer[reviews], size=128, window=5, min_count=3)
    # output_path = os.path.join("..", "model", "reviews.model.bin")
    # model.save(output_path, binary=True)
    return model

def load_model():
    bigram = Phrases.load(os.path.join("..", "model", "bigram.model"))
    trigram = Phrases.load(os.path.join("..", "model", "trigram.model"))
    wv_model = Word2Vec.load(os.path.join("..", "model", "appreviews_word2vec.model"))
    logging.info("Load word2vec model finished")
    return bigram, trigram, wv_model

# find similar word to each issue
def get_similar_word(model, keywords, similar_num):
    issue_dict = {}
    for issue in keywords:
        origi_words = model.most_similar(issue, topn=similar_num)
        origi_words = [word[0] for word in origi_words]
        issue_dict[issue] = origi_words
    return issue_dict

def save_rst(filename, rst):
    with open(filename, 'w') as fout:
        json.dump(rst, fout)

def save_obj(filename, rst):
    import cPickle
    with open(filename, 'w') as fout:
        cPickle.dump(rst, fout)

def filter_reviews(reviews_sents, rate_sents):
    with open("../result/relevant_ad_issues.json.bak") as fin:
        issues_dict = json.load(fin)
    issues_reviews = []
    issues_rates = []
    assert(len(issues_dict) == 1)
    dist = {word: 0 for word in set(issues_dict["ad"])}

    num_review_sents = 0

    for idx, sent in enumerate(reviews_sents):
        for issue, words in issues_dict.items():
            for word in set(words):
                dist[word] += sent.count(word)
            if bool(set(words) & set(sent)):
                issues_reviews.append(sent)
                issues_rates.append(rate_sents[idx])
                num_review_sents += 1
                break

    logging.info("Retrieve %d review sents." % num_review_sents)
    return issues_reviews, issues_rates, dist

def filter_test_reviews(review_sents):
    with open("../result/relevant_ad_issues.json.bak") as fin:
        issues_dict = json.load(fin)
    review_doc = {}
    for apk, docs in review_sents.items():
        issue_docs = []
        for doc in docs:
            issue_sents = []
            for sent in doc:
                for issue, words in issues_dict.items():
                    if bool(set(sent) & set(words)):
                        issue_sents.append(sent)
                        break
            if not issue_sents == []:
                issue_docs.append(list(itertools.chain.from_iterable(issue_sents)))
        review_doc[apk] = issue_docs
    return review_doc
    # for apk, reviews in doc_reviews.items():
    #     for idx, review in enumerate(reviews):
    #         for issue, words in issues_dict.items():
    #             if apk not in issues_reviews:
    #                 issues_reviews[apk] = []
    #             for word in set(words):
    #                 dist[word] += review.count(word)
    #             if bool(set(words) & set(review)):
    #                 issues_reviews[apk].append(review)
    #                 num_reviews += 1
    #                 break
    # logging.info("Retrieve %d reviews."%num_reviews)
    # return issues_reviews, dist

def classify_reviews(review_sents):
    issues_dict = {}
    issues_dict["battery"] = ["battery", "drain", "usage", "consumption", "overheat", "drainer", "consume", "power",
                              "hog", "electricity", "drainage", "charger", "batter", "standby", "discharge", "energy"]
    issues_dict["crash"] = ["crash", "freeze", "foreclose", "lag", "crush", "stall", "close", "shut", "laggy", "glitch",
                            "hang", "load", "stuck", "startup", "buffer", "open", "laggs", "freez", "glitchy", "buggy"]
    issues_dict["memory"] = ["memory", "storage", "space", "gb", "internal", "gigabyte", "ram", "6gb", "occupy", "4gb",
                             "mb", "300mb", "8gb", "500mb", "16gb", "byte", "5gb", "gig", "2gb", "1gb", "1g"]
    issues_dict["network"] = ["network", "connectivity", "internet", "consumption", "wifi", "connection", "reception",
                              "conection", "connect", "signal", "4g", "wi", "3g", "broadband", "fibre", "lte",
                              "reconnecting", "fi", "wireless", "reconnect", "disconnect"]
    issues_dict["privacy"] = ["privacy", "security", "invade", "safety", "personal", "policy", "invasion", "breach",
                              "protection", "protect", "private", "disclosure", "secure", "unsafe", "insecure",
                              "permission", "fingerprint", "encryption", "violation", "encrypt"]
    issues_dict["spam"] = ["spam", "spammer", "scammer", "unsolicited", "harassment", "unwanted", "bot", "bombard",
                           "junk", "scam", "advertisement", "popups", "scraper", "hacker"]
    issues_dict["ui"] = ["ui", "interface", "design", "layout", "gui", "ux", "clunky", "redesign", "aesthetic",
                         "navigation", "usability", "desing", "sleek", "appearance", "aesthetically", "intuitive",
                         "minimalistic", "ugly", "slick", "graphic", "unintuitive", "material_design", "gui",
                         "user_interface"]
    issues_dict["notification"] = ["notification", "push_notification", "notif", "alert", "notifs",
                                   "badge_notification",
                                   "toast_notification", "banner_notification", "vibration", "notification_badge",
                                   "badge", "notification_banner", "notify", "incoming_call", "noti",
                                   "receive_notification",
                                   "message", "incoming_message", "outgo_message", "notification_bar",
                                   "double_notification",
                                   "banner", "unread_message", "reminder", "red_dot", "notification_center",
                                   "badge_icon",
                                   "popup", "vibrate_alert", "vibrate", "lock_screen", "msg", "voice_prompt", "alarm",
                                   "popups", "notifcations", "notification_centre", "notification_panel", "notify_me",
                                   "beep", "live_tile", "weather_alert", "notifaction", "message_arrive",
                                   "sound_vibration",
                                   "birthday_notification", "toast", "notice", "location_service", "an_incoming_call",
                                   "dms"]
    issues_dict["download"] = ["download", "install", "instal", "dl", "dowload", "redownload", "re_download",
                               "re_install",
                               "installation", "reinstall", "downlaod", "downlod", "dwnload", "intall",
                               "delete", "dowloaded", "uninstall", "use", "re_instal", "accidentally_delete",
                               "uninstal", "reinstal", "initialize",
                               "send_msg", "uninstall_reinstall", "register", "upgrade", "redownloaded", "resend",
                               "redownloading", "launch", "unistall", "recover", "re_installation",
                               "verify", "unstall", "dwnld"]
    issues_dict["charge"] = ["charge", "purchase", "buy", "buy_premium", "cost", "recharge", "rob", "pay", "withdrawal",
                             "debit", "cheat", "reimburse", "discharge", "double_charge", "deposit",
                             "refill", "ruble", "tax", "credit", "bill", "owe", "final_value_fee", "deduct_from",
                             "99_cent", "peso", "charge_rs", "10_per_month", "confiscate", "rupee", "hound",
                             "fully_charge", "rs_50", "deduct", "automatically_deduct", "renew", "euro", "30_day",
                             "month_trial", "steal", "payable", "overcharge", "redeem", "discount", "trade",
                             "annual_fee"]
    issues_dict["update"] = ["update", "upgrade", "latest_update", "uodate", "last_update", "updat", "recent_update",
                             "newest_update", "updation", "up_grade", "release", "uptade", "lastest_update", "patch",
                             "version", "redesign", "upadate", "revision", "last_renovation", "newer_version", "upd",
                             "nougat", "instal", "episode", "upate",
                             "newest_version", "udate", "upgrade_ios9", "android_7", "upgrade_ios8",
                             "refresh", "latest_version", "0_nougat", "maj", "doze_feature",
                             "updte", "come_out", "current_version"]
    issues_dict["functionality"] = ["log_in", "log_into", "load", "save", "play", "sign_up", "reload", "upload",
                                    "feature", "function", "capability", "basic_functionality", "usability", "utility",
                                    "flexibility", "fluidity", "basic_function", "customization_option", "functionally",
                                    "additional_feature", "key_feature", "integration", "usefulness", "smoothness",
                                    "component", "full_functionality", "customisation", "volume_slider", "facility",
                                    "consistency", "possibility", "performance", "customization", "core_functionality",
                                    "complexity", "simplicity", "user_experience", "hierarchy", "stability", "option",
                                    "aesthetic", "volume_control", "compatibility",
                                    "additional_functionality", "intuitiveness", "interface",
                                    "keyboard_shortcut", "integration_with", "element", "implementation",
                                    "refinement", "content", "extension"]
    issue_reviews = {}
    issue_reviews['unk'] = []
    for apk, item in review_sents.items():
        docs, rates = item
        for i, doc in enumerate(docs):
            for sent in doc:
                is_issue = False
                for issue, words in issues_dict.items():
                    if issue not in issue_reviews:
                        issue_reviews[issue] = []
                    if bool(set(sent) & set(words)):
                        issue_reviews[issue].append((apk, sent, rates[i]))
                        is_issue = True
                        # issue_sents.append(sent)
                        break
                if is_issue == False:
                    issue_reviews['unk'].append((apk, sent, rates[i]))
        print("%s done..."%apk)
    for key, item in issue_reviews.items():
        print("%s, size: %d" % (key, len(item)))
    return issue_reviews

def sample_reviews_labeling(issue_reviews, num4cat=0):
    """
    sample reviews for labeling
    :param issue_reviews:
    :param num4cat: number of reviews for each category
    :return:
    """
    if num4cat == 0:
        # mass all reviews and random sample
        review_lst = []
        for key, item in issue_reviews.items():
            for apk, sent, rate in item:
                review_lst.append((apk, sent, key, rate))
        # random sample
        sample_ids = np.random.choice(len(review_lst), 500000, replace=False)
        sample_reviews = [review_lst[id] for id in sample_ids if len(review_lst[id][1]) > 3]
    else:
        pass
    # write down sample reviews
    with open("../result/reveiws4label.txt", 'w') as fout:
        for apk, sent, issue, rate in sample_reviews:
            sent_str = ' '.join(sent)
            fout.write("%s\t%s\t%s\t%s\n" % (apk, sent_str, rate, issue))

def count_all_reviews(review_docs, rate_docs):
    issues_dict = {}
    # ============== validation ===============
    issues_dict["battery"] = ["battery", "drain", "usage", "consumption", "overheat", "drainer", "consume", "power",
                              "hog", "electricity", "drainage", "charger", "batter", "standby", "discharge", "energy"]
    issues_dict["crash"] = ["crash", "freeze", "foreclose", "lag", "crush", "stall", "close", "shut", "laggy", "glitch",
                            "hang", "load", "stuck", "startup", "buffer", "open", "laggs", "freez", "glitchy", "buggy"]
    issues_dict["memory"] = ["memory", "storage", "space", "gb", "internal", "gigabyte", "ram", "6gb", "occupy", "4gb",
                             "mb", "300mb", "8gb", "500mb", "16gb", "byte", "5gb", "gig", "2gb", "1gb", "1g"]
    issues_dict["network"] = ["network", "connectivity", "internet", "consumption", "wifi", "connection", "reception",
                              "conection", "connect", "signal", "4g", "wi", "3g", "broadband", "fibre", "lte",
                              "reconnecting", "fi", "wireless", "reconnect", "disconnect"]
    issues_dict["privacy"] = ["privacy", "security", "invade", "safety", "personal", "policy", "invasion", "breach",
                              "protection", "protect", "private", "disclosure", "secure", "unsafe", "insecure",
                              "permission", "fingerprint", "encryption", "violation", "encrypt"]
    issues_dict["spam"] = ["spam", "spammer", "scammer", "unsolicited", "harassment", "unwanted", "bot", "bombard",
                           "junk", "scam", "advertisement", "popups", "scraper", "hacker"]
    issues_dict["ui"] = ["ui", "interface", "design", "layout", "gui", "ux", "clunky", "redesign", "aesthetic",
                         "navigation", "usability", "desing", "sleek", "appearance", "aesthetically", "intuitive",
                         "minimalistic", "ugly", "slick", "graphic", "unintuitive", "material_design", "gui", "user_interface"]
    issues_dict["notification"] = ["notification", "push_notification", "notif", "alert", "notifs", "badge_notification",
                                   "toast_notification", "banner_notification", "vibration", "notification_badge",
                                   "badge", "notification_banner", "notify", "incoming_call", "noti", "receive_notification",
                                   "message", "incoming_message", "outgo_message", "notification_bar", "double_notification",
                                   "banner", "unread_message", "reminder", "red_dot", "notification_center", "badge_icon",
                                   "popup", "vibrate_alert", "vibrate", "lock_screen", "msg", "voice_prompt", "alarm",
                                   "popups", "notifcations", "notification_centre", "notification_panel", "notify_me",
                                   "beep", "live_tile", "weather_alert", "notifaction", "message_arrive", "sound_vibration",
                                   "birthday_notification", "toast", "notice", "location_service", "an_incoming_call", "dms"]
    issues_dict["download"] = ["download", "install", "instal", "dl", "dowload", "redownload", "re_download", "re_install",
                               "installation", "reinstall", "downlaod", "downlod", "dwnload", "intall",
                               "delete", "dowloaded", "uninstall", "use", "re_instal", "accidentally_delete",
                               "uninstal", "reinstal", "initialize",
                               "send_msg", "uninstall_reinstall", "register", "upgrade", "redownloaded", "resend",
                               "redownloading", "launch", "unistall", "recover", "re_installation",
                               "verify", "unstall", "dwnld"]
    issues_dict["charge"] = ["charge", "purchase", "buy", "buy_premium", "cost", "recharge", "rob", "pay", "withdrawal",
                             "debit", "cheat", "reimburse", "discharge", "double_charge", "deposit",
                             "refill", "ruble", "tax", "credit", "bill", "owe", "final_value_fee", "deduct_from",
                             "99_cent", "peso", "charge_rs", "10_per_month", "confiscate", "rupee", "hound",
                             "fully_charge", "rs_50", "deduct", "automatically_deduct", "renew", "euro", "30_day",
                             "month_trial", "steal", "payable", "overcharge", "redeem", "discount", "trade",
                             "annual_fee"]
    issues_dict["update"] = ["update", "upgrade", "latest_update", "uodate", "last_update", "updat", "recent_update",
                             "newest_update", "updation", "up_grade", "release", "uptade", "lastest_update", "patch",
                             "version", "redesign", "upadate", "revision", "last_renovation", "newer_version", "upd",
                             "nougat", "instal", "episode", "upate",
                             "newest_version", "udate", "upgrade_ios9", "android_7", "upgrade_ios8",
                             "refresh", "latest_version", "0_nougat", "maj", "doze_feature",
                             "updte", "come_out", "current_version"]
    issues_dict["functionality"] = ["log_in", "log_into", "load", "save", "play", "sign_up", "reload", "upload",
                                    "feature", "function", "capability", "basic_functionality", "usability", "utility",
                                    "flexibility", "fluidity", "basic_function", "customization_option", "functionally",
                                    "additional_feature", "key_feature", "integration", "usefulness", "smoothness",
                                    "component", "full_functionality", "customisation", "volume_slider", "facility",
                                    "consistency", "possibility", "performance", "customization", "core_functionality",
                                    "complexity", "simplicity", "user_experience", "hierarchy", "stability", "option",
                                    "aesthetic", "volume_control", "compatibility",
                                    "additional_functionality", "intuitiveness", "interface",
                                    "keyboard_shortcut", "integration_with", "element", "implementation",
                                    "refinement", "content", "extension"]
    # ============== validation ===============
    issues = {}
    issues_rates = {}
    scores = {}
    num_issues = 0
    for idx, review in enumerate(review_docs):
        for key, words in issues_dict.items():
            if key not in issues:
                issues[key] = 0
                issues_rates[key] = []
            if bool(set(words) & set(review)):
                issues[key] += 1
                issues_rates[key].append(rate_docs[idx])
                num_issues += 1
    for issue, rates in issues_rates.items():
        issues_rates[issue] = np.mean(rates) if not rates == [] else None
        scores[issue] = my_n_weight(float(issues[issue])/len(review_docs), issues_rates[issue])
    issue_count = {"number_reviews": len(review_docs), "num_issue_reviews": num_issues, "issues": issues, "issues_rates": issues_rates, "scores": scores}

    return issue_count

# count the number of user complaints for each issue
def count_reviews(doc_reviews):
    with open("../result/relevant_issues.json") as fin:
        issues_dict = json.load(fin)

    issue_count = []
    for apk, reviews in doc_reviews.items():
        issues = {}
        issues_rates = {}
        issues_reviews = {}
        num_issues = 0
        with open(os.path.join("../data/raw/mongodb", apk, "info.txt")) as fin:
            rates = [float(line[:3]) for line in fin.readlines()]
        for dix, review in enumerate(reviews):
            for issue, words in issues_dict.items():
                if issue not in issues:
                    issues[issue] = 0
                    issues_rates[issue] = []
                    issues_reviews[issue] = []
                if bool(set(words) & set(review)):
                    issues[issue] += 1
                    issues_rates[issue].append(rates[dix])
                    issues_reviews[issue].append(review)
                    num_issues += 1
        issues_rate = {}
        for issue, rates in issues_rates.items():
            issues_rate[issue] = np.mean(rates) if not rates == [] else None
        issue_count.append({"app_name": apk, "rate": np.mean(rates), "num_reviews": len(reviews), "num_issue_review": num_issues,
                            "issues_portion": float(num_issues)/len(reviews), "issues": issues, "issues_rate": issues_rate, "issues_review": issues_reviews})
    return issue_count

def clustering(wv_model, n_clusters=8, method="Kmeans"):
    with open("../result/ad_issue_reviews") as fin:
        reviews = json.load(fin)
    review_words = set(list(itertools.chain.from_iterable(reviews)))
    word_centers = []
    data = np.array([wv_model[word] for word in review_words if word in wv_model])
    if method == "Kmeans":
        clusters = KMeans(n_clusters=n_clusters).fit(data)
    elif method == "SpectralClustering":
        clusters = SpectralClustering(n_clusters=n_clusters).fit(data)
    for center in clusters.cluster_centers_:
        # pick the most similar 8 words as cluster center
        word_centers.append(wv_model.similar_by_vector(center, topn=8))
    return word_centers

def train_lda(n_topics=10):
    with open("../result/ad_issue_reviews") as fin:
        reviews = json.load(fin)
    # build bag-of-words, corpus
    reviews = [[word for word in review if word not in stopwords.words('english')] for review in reviews]
    from collections import defaultdict
    freq = defaultdict(int)
    for review in reviews:
        for token in review:
            freq[token] += 1
    reviews = [[token for token in review if freq[token] > 1] for review in reviews]
    # dictionary = corpora.Dictionary(reviews)
    # only select ad related word
    with open("../result/relevant_ad_issues.json") as fin:
        ad_words = json.load(fin)
    ad_words = ad_words["ad"]
    dictionary = corpora.Dictionary([ad_words])

    corpus = [dictionary.doc2bow(review) for review in reviews]
    logging.info("LDA start training...")
    lda = LdaMulticore(corpus, num_topics=n_topics)

    lda.save("../model/lda_ad_%d.model"%n_topics)
    return lda

def visual_lda():
    lda = LdaMulticore.load("../model/lda.model")
    with open("../result/ad_issue_reviews") as fin:
        reviews = json.load(fin)
    # build bag-of-words, corpus
    reviews = [[word for word in review if word not in stopwords.words('english')] for review in reviews]
    from collections import defaultdict
    freq = defaultdict(int)
    for review in reviews:
        for token in review:
            freq[token] += 1
    reviews = [[token for token in review if freq[token] > 1] for review in reviews]
    dictionary = corpora.Dictionary(reviews)
    corpus = [dictionary.doc2bow(review) for review in reviews]
    import pyLDAvis.gensim as gensimvis
    import pyLDAvis
    vis_data = gensimvis.prepare(lda, corpus, dictionary)
    pyLDAvis.display(vis_data)

def build_ad_issue_vis(wv_model, review_sents="", rate_sents=""):
    with open("../result/relevant_ad_issues.json") as fin:
        ad_words = json.load(fin)
    ad_words = list(set(ad_words['ad']))

    if review_sents == "" and rate_sents == "":
        with open("../result/ad_issue_reviews") as fin, open("../result/ad_issue_rates") as frin:
            reviews = json.load(fin)
            rates = json.load(frin)
    elif review_sents != "" and rate_sents!= "":
        reviews = review_sents
        rates = rate_sents
    # build bag-of-words, corpus
    dictionary = corpora.Dictionary(reviews)
    stopword_ids = map(dictionary.token2id.get, stopwords.words('english'))
    dictionary.filter_tokens(stopword_ids)
    dictionary.compactify()
    dictionary.filter_extremes(no_below=2, keep_n=None)
    dictionary.compactify()

    # filter ad words
    ad_words = [word for word in ad_words if word in dictionary.token2id]

    corpus = [dictionary.doc2bow(review) for review in reviews]
    tfidf = TfidfModel(corpus)
    ad_ids = map(dictionary.token2id.get, ad_words)
    ad_id2token = dict(zip(ad_ids, ad_words))
    n_corpus = dictionary.doc2bow(list(itertools.chain.from_iterable(reviews)))
    ad_corpus = []
    print("ad review num: %s"%len(reviews))
    for (key, value) in n_corpus:
        if key in ad_ids:
            ad_corpus.append((key, value))
            print("%s\t%s"%(ad_id2token[key], float(value)/len(reviews)))

    #============================
    ad_word_tfidf = tfidf[ad_corpus]

    # compute the mean rate of ad words
    ad_rate_dict = {}
    for idx, review in enumerate(reviews):
        for word in ad_words:
            if word not in ad_rate_dict:
                ad_rate_dict[word] = []
            if word in review:
                ad_rate_dict[word].append(rates[idx])
    ad_word_rates = {}
    for word in ad_words:
        ad_word_rates[word] = np.mean(ad_rate_dict[word])

    tf_sum = np.sum([value for key, value in ad_corpus])
    ad_word_weight = {}
    for word in ad_words:
        tfidf_v = dict(ad_word_tfidf)[dictionary.token2id[word]]
        tf_v = dict(ad_corpus)[dictionary.token2id[word]] / float(tf_sum)
        rate_v = ad_word_rates[word]
        ad_word_weight[word] = my_weight(tf_v, tfidf_v, rate_v)
    # write for excel
    # from sklearn.decomposition import PCA
    # from sklearn.manifold import TSNE
    #
    # topic_term = np.array([wv_model[word] for word in ad_words])
    # pca = PCA(n_components=50)
    # topic_term = pca.fit_transform(topic_term)
    #
    # tsne_model = TSNE()
    # x_y = tsne_model.fit_transform(topic_term)
    # with open("../result/excel.x_y.case", "w") as fout:
    #     for vec in x_y:
    #         fout.write("%f\t%f\n"%(vec[0],vec[1]))
    # with open("../result/excel.label.case", "w") as fout:
    #     for word in ad_words:
    #         fout.write("%s\n"%word)
    # with open("../result/excel.size.case", "w") as fout:
    #     for word in ad_words:
    #         fout.write("%f\n"%ad_word_weight[word])
    # with open("../result/excel.rate.case", "w") as fout:
    #     for word in ad_words:
    #         fout.write("%f\n"%ad_word_rates[word])

    # ad_counts = []
    # for word in ad_words:
    #     ad_dict = {}
    #     ad_dict["issue"] = word
    #     ad_dict["wv"] = list(wv_model[word])
    #     ad_dict["tfidf"] = dict(ad_word_weights)[dictionary.token2id[word]]
    #     ad_counts.append(ad_dict)
    # return ad_counts
    # topic_term = [5 * wv_model[word] * dict(ad_word_weights)[dictionary.token2id[word]] for word in ad_words]
    # import pyLDAvis
    # pyLDAvis.prepare(topic_term,doc_topic,1,vocab,term_freq)

def my_trans(x):
    rst = 1/(1+np.exp(-x))
    rst /= 100
    return rst

def my_weight(tf, tfidf, rate):
    return tfidf / rate

def my_n_weight(percent, rate):
    return -np.log((rate-0.9)/5) * percent
# def read_rst(filename):
#     import cPickle
#     with open(filename) as fin:
#         ad_counts = cPickle.load(fin)
#     for ad_count in ad_counts:
#         ad_count["issue"] # word
#         ad_count["wv"] # word2vec's vector
#         ad_count["tfidf"] # tfidf
if __name__ == "__main__":
    # define the top n number of words

    # reviews_sents, reviews_docs, rate_sents, rate_docs = get_all_reviews()
    # sent_reviews, doc_reviews = get_test_reviews()

    # reviews = extract_phrases(reviews_sents, reviews_docs, save=True)
    # model = training(reviews)
    bigram, trigram, wv_model = load_model()

    # reviews_sents = trigram[bigram[reviews_sents]]

    reviews_sents, reviews_docs, rate_sents, rate_docs = get_all_reviews()

    issue_reviews = classify_reviews(reviews_sents)
    sample_reviews_labeling(issue_reviews)
    # issue_dict = get_similar_word(wv_model, add_issues, similar_num)
    # save_rst("../result/relevant_add_issues.json", issue_dict)

    # doc_reviews = filter_test_reviews(sent_reviews)
    # save_rst("../result/relevant_ad_count.test.json", doc_reviews)
    # issue_count = count_reviews(doc_reviews)
    # save_rst("../result/relevant_ad_count.json", issue_count)

    # issues_reviews, issues_rates, issues_dist = filter_reviews(reviews_sents, rate_sents)
    # save_rst("../result/ad_issue_reviews", issues_reviews)
    # save_rst("../result/ad_issue_rates", issues_rates)

    # centers = clustering(wv_model)
    # save_rst("../result/centers", centers)

    # train_lda(8)
    # visual_lda()

    # ad_counts = build_ad_issue_vis(wv_model)
    # save_obj("../result/ad_vis_counts", ad_counts)


    # issue_count = count_all_reviews(reviews_docs, rate_docs)
    # save_rst("../result/issue_scores.json", issue_count)