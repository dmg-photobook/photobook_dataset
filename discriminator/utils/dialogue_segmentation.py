import os
import json
import pickle
import argparse
import random as rd
from collections import defaultdict

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils.processor import Log


# Log Loader # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_logs(log_repository, data_path):

    filepath = os.path.join(data_path, log_repository)
    print("Loading logs from {}".format(filepath))

    missing_counter = 0
    file_count = 0
    for _, _, files in os.walk(filepath):
        file_count += len(files)
    logs = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), 'r') as logfile:
                    log = Log(json.load(logfile))
                    if log.complete:
                        logs.append(log)
                    else:
                        missing_counter += 1

    print("Complete. Loaded {} completed game logs.".format(len(logs)))
    return logs


# Dataset Splitter # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def generate_game_sets(sample_size, domain_dict, remaining_total):
    """
    Generates a data set of the given sample size by allocating games relative to their domain's frequencies.
    :param sample_size: int. Number of games to be allocated to the set
    :param domain_dict: dict. Dictionary linking domain IDs and the IDs of all games in that domain
    :param remaining_total: Total number of games remaining in the domain_dict.
    :return: [list, dict, int]. List of game_ids, updated domain_dict and total number of games remaining in the domain_dict
    """
    game_set = []
    sampled_games = 0

    for domain_id, games in domain_dict.items():
        domain_sample_size = int(len(games) / remaining_total * sample_size + 0.5)
        sampled_games += domain_sample_size
        game_set.extend([(domain_id, game_id) for game_id in rd.sample(games, domain_sample_size)])

    for domain_id, game_id in game_set:
        games = domain_dict[domain_id]
        games.remove(game_id)
        domain_dict[domain_id] = games

    game_list = [tup[1] for tup in game_set]
    remaining_total = remaining_total - sampled_games

    return game_list, domain_dict, remaining_total


# Dialogue Segmentation Heuristics # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def clean_clicks(round_data):
    """
    Removes duplicate clicks from the dialogues recorded in the passed round data object
    :param round_data: Round object.
    :return: A Round object that is cleared of multiple clicks
    """
    filtered_data = []
    speaker_selections = defaultdict(lambda: [])
    messages = round_data.messages.copy()
    messages.reverse()

    cleaning_counter = 0

    for message in messages:
        if message.type == 'selection':
            previous_selections = speaker_selections[message.speaker]
            target = message.text.split()[2]
            if target in previous_selections:
                cleaning_counter += 1
                continue
            else:
                speaker_selections[message.speaker].append(target)
                filtered_data.append(message)
        else:
            filtered_data.append(message)

    filtered_data.reverse()
    return filtered_data, cleaning_counter


def is_selection(message):
    """
    Returns True if the passed message is a labeling action
    :param message: Message object.
    :return: True if the passed message is a labeling action
    """
    if len(message.text.split()) == 3 and message.text.split()[0] == '<selection>':
        return True
    return False


def is_common_label(message):
    """
    Returns True if the passed message is a selection and the target image was marked common
    Returns False if the message is no selection or the target was marked different
    :param message: Message object.
    :return: True if the passed message is a selection and the target image was marked common
    """
    if not message.type == 'selection':
        print("Received message not a selection!")
        return False

    if message.text.split()[1] == '<com>':
        return True
    else:
        return False


def get_target(message):
    """
    Returns the target identifier from a message or None if the message was no selection
    :param message: Message object.
    :return: The target identifier of the message or None if the message was no selection
    """
    if not len(message.text.split()) == 3:
        print("Received message not a selection!")
        return None
    return message.text.split()[2]


def dialogue_segmentation(logs, selection, seg_verbose=False):
    """
    sections the dialogues in the game rounds based on a pre-defined heuristics
    :param logs: list. List containing the Log objects created from the log files
    :param selection: list. List containing the set of game indexes to be included in the current split
    :param seg_verbose: bool. Set to True to print the decision structure
    :return: A list of lists containing tuples of dialogue segments and their corresponding targets for the games in the given set
    """
    cleaning_total = 0
    section_counter = 0

    dialogue_sections = []
    for game in logs:
        game_id = game.game_id
        if selection and game_id not in selection:
            continue

        game_sections = []
        for round_data in game.rounds:
            selections = []
            messages = round_data.messages

            if seg_verbose: print("\n")
            for message in round_data.messages:
                if seg_verbose: print("{}: {}".format(message.speaker, message.text))
                if message.type == 'selection':
                    selections.append((message.message_id, message.speaker, message.text))
            if seg_verbose: print("\n")

            if len(selections) > 6:
                messages, cleaning_counter = clean_clicks(round_data)
                cleaning_total += cleaning_counter

            sections = []
            current_section = []
            current_targets = []
            previous_selection = None
            previous_turn = None
            skip = 0
            for i, message in enumerate(messages):
                if skip > 0:
                    skip -= 1
                    continue
                if seg_verbose: print("{}: {}".format(message.speaker, message.text))
                if seg_verbose: print("--> Current section contains {} utterances".format(len(current_section)))
                if message.type == 'text':
                    if previous_turn and seg_verbose: print("--> Previous turn text: ", previous_turn.text)
                    if previous_turn and is_selection(previous_turn):
                        if seg_verbose: print("--> Previous turn was selection")
                        if is_common_label(previous_selection):
                            if seg_verbose: print("--> Previous selection was common")
                            if previous_selection.speaker != message.speaker:
                                if seg_verbose: print("--> Previous selection was from other speaker")
                                next_message = messages[i + 1]
                                if next_message.type == 'selection':
                                    if seg_verbose: print("--> Next turn is a selection")
                                    if next_message.speaker == message.speaker:
                                        if seg_verbose: print("--> Next selection is from same speaker")
                                        if is_common_label(next_message):
                                            if seg_verbose: print("--> Next selection is common")
                                            if previous_selection.text == next_message.text:
                                                if seg_verbose: print("--> Case 1")
                                                # Case: After one speaker selected an image as common, the other speaker makes one utterance and marks the same image as common
                                                # Resolution: The previous section is saved with the common image as referent and a new section is initialised with the trailing utterance of the second speaker
                                                current_targets.append(get_target(previous_selection))
                                                if current_section:
                                                    sections.append((current_section, set(current_targets)))

                                                current_section = [message]
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = message
                                                skip = 1
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 2")
                                                # Case: After one speaker selected an image as common, the other speaker makes one utterance and marks an other image as common
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents. A new section is initialised empty
                                                current_targets.extend(
                                                    [get_target(previous_selection), get_target(next_message)])
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                        else:
                                            if get_target(previous_selection) == get_target(next_message):
                                                if seg_verbose: print("--> Case 3")
                                                # Case: After one speaker selected an image as common, the other speaker makes one utterance and marks the same image as different
                                                # Resolution: The previous section is saved with the common image as referent and a new section is initialised with the trailing utterance of the second speaker
                                                current_targets.append(get_target(previous_selection))
                                                if current_section:
                                                    sections.append((current_section, set(current_targets)))

                                                current_section = [message]
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = message
                                                skip = 1
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 4")
                                                # Case: After one speaker selected an image as common, the other speaker makes one utterance and marks an other image as different
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents. A new section is initialised empty
                                                current_targets.extend(
                                                    [get_target(previous_selection), get_target(next_message)])
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                    else:
                                        if seg_verbose: print("--> Case 5")
                                        # Case: After one speaker selected an image as common, the other speaker makes one utterance and the first speaker marks a second image
                                        # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents
                                        current_targets.extend(
                                            [get_target(previous_selection), get_target(next_message)])
                                        current_section.append(message)
                                        sections.append((current_section, set(current_targets)))

                                        current_section = []
                                        current_targets = []
                                        previous_selection = next_message
                                        previous_turn = next_message
                                        skip = 1
                                        continue
                                else:
                                    if next_message.speaker == message.speaker:
                                        if i + 2 < len(messages):
                                            second_next_message = messages[i + 2]
                                            if second_next_message.speaker == next_message.speaker and second_next_message.type == 'selection' and get_target(
                                                    second_next_message) == get_target(previous_selection):
                                                if seg_verbose: print("--> Case 6")
                                                # Case: After one speaker selected an image as common, the other speaker makes two utterances and marks the same image
                                                # Resolution: The trailing utterances are added to the current section and the current section is saved with the common image as referent
                                                current_targets.append(get_target(previous_selection))
                                                current_section.append(message)
                                                current_section.append(next_message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = second_next_message
                                                previous_turn = second_next_message
                                                skip = 2
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 7")
                                                # Case: After one speaker selected an image as common, the other speaker makes multiple utterances without marking any images
                                                # Resolution: Save the current section with the target marked as common and initialise a new section with the current utterance
                                                current_targets.append(get_target(previous_selection))
                                                if current_section:
                                                    sections.append((current_section, set(current_targets)))

                                                current_section = [message]
                                                current_targets = []
                                                previous_turn = message
                                                continue
                                        else:
                                            pass
                                    else:
                                        if seg_verbose: print("--> Case 9")
                                        # Case: After one speaker selected an image as common, there is an interaction between the speakers
                                        # Resolution: Save the current section with the target marked as common and initialise a new section with the current utterance
                                        current_targets.append(get_target(previous_selection))
                                        if current_section:
                                            sections.append((current_section, set(current_targets)))

                                        current_section = [message]
                                        current_targets = []
                                        previous_turn = message
                                        continue
                            else:
                                next_message = messages[i + 1]
                                if next_message.type == 'selection':
                                    if next_message.speaker != message.speaker:
                                        if is_common_label(next_message):
                                            if previous_selection.text == next_message.text:
                                                if seg_verbose: print("--> Case 10")
                                                # Case: After one speaker selected an image as common, he or she adds something, leading to the other speaker marking the same image as common as well
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with the common image as referent. A new section is initialised empty
                                                current_targets.append(get_target(previous_selection))
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 11")
                                                # Case: After one speaker selected an image as common,  he or she adds something, leading to the other speaker marking a different image as common
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with both disagreed referents. A new section is initialised empty
                                                current_targets.extend(
                                                    [get_target(previous_selection), get_target(next_message)])
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                        else:
                                            if get_target(previous_selection) == get_target(next_message):
                                                if seg_verbose: print("--> Case 12")
                                                # Case: After one speaker selected an image as common, he or she adds something, leading to the other speaker marking the same image as different
                                                # Resolution: The current section is saved with the disagreed image as referent. A new section is initialised with the trailing utterance
                                                current_targets.append(get_target(next_message))
                                                sections.append((current_section, set(current_targets)))

                                                current_section = [message]
                                                current_targets = []
                                                previous_selection = previous_selection
                                                previous_turn = message
                                                skip = 1
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 13")
                                                # Case: After one speaker selected an image as common, he or she adds something, leading to the other speaker marking another image as different
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents. A new section is initialised empty
                                                current_targets.extend(
                                                    [get_target(previous_selection), get_target(next_message)])
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                    else:
                                        if seg_verbose: print("--> Case 14")
                                        # Case: After one speaker selected an image as common, he or she adds something and marks a second image
                                        # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents. A new section is initialised empty
                                        current_targets.extend(
                                            [get_target(previous_selection), get_target(next_message)])
                                        current_section.append(message)
                                        sections.append((current_section, set(current_targets)))

                                        current_section = []
                                        current_targets = []
                                        previous_selection = next_message
                                        previous_turn = next_message
                                        skip = 1
                                        continue
                                else:
                                    if seg_verbose: print("--> Case 15")
                                    # Case: After one speaker selected an image as common, he or she adds multiple utterances
                                    # Resolution: Save the current section with the target marked as common and initialise a new section with the current utterance
                                    current_targets.append(get_target(previous_selection))
                                    if current_section:
                                        sections.append((current_section, set(current_targets)))

                                    current_section = [message]
                                    current_targets = []
                                    previous_turn = message
                                    continue
                        else:
                            if seg_verbose: print("--> Previous selection was different")
                            if previous_selection.speaker != message.speaker:
                                if seg_verbose: print("--> Previous speaker was the other participant")
                                next_message = messages[i + 1]
                                if next_message.type == 'selection':
                                    if seg_verbose: print("--> Next message is selection")
                                    if next_message.speaker == message.speaker:
                                        if not is_common_label(next_message):
                                            if previous_selection.text == next_message.text:
                                                if seg_verbose: print("--> Case 16")
                                                # Case: After one speaker selected an image as different, the other speaker makes one utterance and marks the same image as different
                                                # Resolution: The previous section is saved with the wrongly labeled image as referent and a new section is initialised with the trailing utterance of the second speaker
                                                current_targets.append(get_target(previous_selection))
                                                if current_section:
                                                    sections.append((current_section, set(current_targets)))

                                                current_section = [message]
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = message
                                                skip = 1
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 17")
                                                # Case: After one speaker selected an image as different, the other speaker makes one utterance and marks another image as different
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents
                                                current_targets.extend(
                                                    [get_target(previous_selection), get_target(next_message)])
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                        else:
                                            if get_target(previous_selection) == get_target(next_message):
                                                if seg_verbose: print("--> Case 18")
                                                # Case: After one speaker selected an image as different, the other speaker makes one utterance and marks the same image as common
                                                # Resolution: The previous section is saved with the disagreed image as referent and a new section is initialised with the trailing utterance of the second speaker
                                                current_targets.append(get_target(previous_selection))
                                                if current_section:
                                                    sections.append((current_section, set(current_targets)))

                                                current_section = [message]
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = message
                                                skip = 1
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 19")
                                                # Case: After one speaker selected an image as different, the other speaker makes one utterance and marks a different image as common
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents
                                                current_targets.extend(
                                                    [get_target(previous_selection), get_target(next_message)])
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue

                                    else:
                                        if seg_verbose: print("--> Case 20")
                                        # Case: After one speaker selected an image as different, the other speaker makes one utterance and the first speaker marks a second image
                                        # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents. A new section is initialised empty
                                        current_targets.extend(
                                            [get_target(previous_selection), get_target(next_message)])
                                        current_section.append(message)
                                        sections.append((current_section, set(current_targets)))

                                        current_section = []
                                        current_targets = []
                                        previous_selection = next_message
                                        previous_turn = next_message
                                        skip = 1
                                        continue
                                else:
                                    if seg_verbose: print("--> Next message is regular utterance")
                                    if next_message.speaker == message.speaker:
                                        if seg_verbose: print("--> Next speaker is current speaker")
                                        if i + 2 < len(messages):
                                            second_next_message = messages[i + 2]
                                            if second_next_message.speaker == next_message.speaker and second_next_message.type == 'selection' and get_target(
                                                    second_next_message) == get_target(previous_selection):
                                                if seg_verbose: print("--> Case 21")
                                                # Case: After one speaker selected an image as different, the other speaker makes two utterances and marks the same image
                                                # Resolution: The trailing utterances are added to the current section and the current section is saved with the marked image as referent
                                                current_targets.append(get_target(previous_selection))
                                                current_section.append(message)
                                                current_section.append(next_message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = second_next_message
                                                previous_turn = second_next_message
                                                skip = 2
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 22")
                                                # Case: After one speaker selected an image as different, the other speaker makes multiple utterances without marking any images
                                                # Resolution: Save the current section with the target marked as different and initialise a new section with the current utterance
                                                current_targets.append(get_target(previous_selection))
                                                if current_section:
                                                    sections.append((current_section, set(current_targets)))

                                                current_section = [message]
                                                current_targets = []
                                                previous_turn = message
                                                continue
                                        else:
                                            pass
                                    else:
                                        if seg_verbose: print("--> Case 24")
                                        # Case: After one speaker selected an image as different, there is an interaction between the speakers
                                        # Resolution: Save the current section with the target marked as different and initialise a new section with the current utterance
                                        current_targets.append(get_target(previous_selection))
                                        if current_section:
                                            sections.append((current_section, set(current_targets)))

                                        current_section = [message]
                                        current_targets = []
                                        previous_turn = message
                                        continue
                            else:
                                if seg_verbose: print("--> Previous speaker was the same participant")
                                next_message = messages[i + 1]
                                if next_message.type == 'selection':
                                    if seg_verbose: print("--> Next message is selection")
                                    if next_message.speaker != message.speaker:
                                        if not is_common_label(next_message):
                                            if previous_selection.text == next_message.text:
                                                if seg_verbose: print("--> Case 25")
                                                # Case: After one speaker selected an image as different, he or she adds something, leading to the other speaker marking the same image as different
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with the wrongly labeled image as referent
                                                current_targets.append(get_target(previous_selection))
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 26")
                                                # Case: After one speaker selected an image as different, he or she adds something, leading to the other speaker marking a different image as different
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with both disagreed referents
                                                current_targets.extend(
                                                    [get_target(previous_selection), get_target(next_message)])
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                        else:
                                            if get_target(previous_selection) == get_target(next_message):
                                                if seg_verbose: print("--> Case 27")
                                                # Case: After one speaker selected an image as different, he or she adds something, leading to the other speaker marking the same image as common
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with the disagreed image as referent
                                                current_targets.append(get_target(next_message))
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = [message]
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = message
                                                skip = 1
                                                continue
                                            else:
                                                if seg_verbose: print("--> Case 28")
                                                # Case: After one speaker selected an image as different, he or she adds something, leading to the other speaker marking another image as common
                                                # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents
                                                current_targets.extend(
                                                    [get_target(previous_selection), get_target(next_message)])
                                                current_section.append(message)
                                                sections.append((current_section, set(current_targets)))

                                                current_section = []
                                                current_targets = []
                                                previous_selection = next_message
                                                previous_turn = next_message
                                                skip = 1
                                                continue
                                    else:
                                        if seg_verbose: print("--> Case 29")
                                        # Case: After one speaker selected an image as different, he or she adds something and marks a second image
                                        # Resolution: The trailing utterance is added to the current section and the current section is saved with both referents
                                        current_targets.extend(
                                            [get_target(previous_selection), get_target(next_message)])
                                        current_section.append(message)
                                        sections.append((current_section, set(current_targets)))

                                        current_section = []
                                        current_targets = []
                                        previous_selection = next_message
                                        previous_turn = next_message
                                        skip = 1
                                        continue
                                else:
                                    if seg_verbose: print("--> Case 30")
                                    # Case: After one speaker selected an image as different, he or she adds an utterance
                                    # Resolution: Save the current section with the target marked as different and initialise a new section with the current utterance
                                    if current_section and current_targets:
                                        sections.append((current_section, set(current_targets)))
                                    current_section = [message]
                                    current_targets = []
                                    previous_turn = message
                                    continue
                    else:
                        if seg_verbose: print("--> Case 31")
                        # Case: Regular utterance following another regular utterance
                        # Resolution: Add utterance to current section
                        current_section.append(message)
                        current_targets = []
                        previous_turn = message
                        continue
                elif message.type == 'selection':
                    if seg_verbose: print("--> Selection")
                    if current_section:
                        if seg_verbose: print("--> Case 32")
                        # A speaker marks an image
                        # Resolution: Add the label of the selection to the set of current targets
                        if seg_verbose: print("--> Adding target")
                        current_targets.append(get_target(message))
                        previous_selection = message
                        previous_turn = message
                        continue
                    else:
                        if seg_verbose: print("--> Case 33")
                        if seg_verbose: print("--> Current section is empty. Skipping selection")
                        continue
                else:
                    continue

            if current_section and current_targets:
                sections.append((current_section, set(current_targets)))

            sections = (sections, set(round_data.images["A"] + round_data.images["B"]))
            game_sections.append(sections)
            if seg_verbose: print("{} dialogue sections encountered in round".format(len(sections)))
            section_counter += len(sections)

        dialogue_sections.append((game_id, game_sections))

    if seg_verbose: print("Total of {} duplicate labeling action(s) removed.".format(cleaning_total))
    if seg_verbose: print("Processed {} dialogue(s).".format(len(dialogue_sections)))
    if seg_verbose: print("Generated a total of {} dialogue section(s).".format(section_counter))

    return dialogue_sections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../data")
    parser.add_argument("-new_split", type=bool, default=False)
    parser.add_argument("-split", type=list, default=[15,15])

    args = parser.parse_args()
    data_path = args.data_path
    new_split = args.new_split
    split = list(args.split)
    if len(split) != 2:
        print("Alert: -split argument takes a list of length 2 with validation and test size in %. Using default 15/15/70 split.")
        split = [15,15]

    logs = load_logs("logs", data_path)

    # Create a new split
    if new_split:
        val_size = int(split[0]/100 * len(logs))
        test_size = int(split[1]/100 * len(logs))

        domain_dict = defaultdict(lambda: [])
        for game in logs:
            domain_dict[game.domain_id].append(game.game_id)

        data_split = dict()
        remaining_total = len(logs)
        data_split["dev"], domain_dict, remaining_total = generate_game_sets(60, domain_dict, remaining_total)
        data_split["val"], domain_dict, remaining_total = generate_game_sets(val_size, domain_dict, remaining_total)
        data_split["test"], domain_dict, remaining_total = generate_game_sets(val_size, domain_dict, remaining_total)

        train_set = []
        for domain_id, games in domain_dict.items():
            train_set.extend(games)

        data_split["train"] = train_set

        with open(os.path.join(data_path, "new_data_splits.json"), 'w') as f:
            json.dump(data_split, f)


    # Load a pre-defined split
    else:
        with open(os.path.join(data_path, "data_splits.json"), 'r') as f:
            data_split = json.load(f)

    print("Development set contains {} games".format(len(data_split["dev"])))
    print("Validation set contains {} games".format(len(data_split["val"])))
    print("Test set contains {} games".format(len(data_split["test"])))
    print("Train set contains {} games".format(len(data_split["train"])))

    for set_name in ['dev', 'val', 'test', 'train']:
        set_ids = data_split[set_name]
        dialogue_sections = dialogue_segmentation(logs, set_ids, seg_verbose=False)
        with open(os.path.join(data_path, "{}_sections.pickle".format(set_name)), 'wb') as f:
            pickle.dump(dialogue_sections, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")



